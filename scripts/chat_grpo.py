"""
Group Relative Policy Optimization (GRPO) on GSM8K.

This implements the original DeepSeekMath objective: group-normalized rewards,
an old-policy clipped surrogate, and KL regularization to a frozen SFT model.

1 GPU:
python -m scripts.chat_grpo --output-model-tag=d20-grpo

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- \
    --model-tag=d32-a800-bf16-L \
    --model-step=242 \
    --output-model-tag=d32-a800-grpo \
    --rollout-batch-size=32 \
    --policy-batch-size=16 \
    --device-batch-size=8 \
    --num-samples=16 \
    --temperature=1.0 \
    --top-k=0 \
    --init-lr-frac=0.005 \
    --run=dummy
"""

import argparse
import itertools

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from nanochat.grpo import (
    compute_grpo_loss,
    compute_group_advantages,
    get_rollout_batch_plan,
    reserve_grpo_checkpoint_dir,
    should_save_grpo_checkpoint,
    validate_grpo_batch_config,
    validate_grpo_sampling_config,
)
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="GRPO on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading and saving
parser.add_argument("--model-tag", type=str, default=None, help="SFT model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="SFT model step to load from")
parser.add_argument("--output-model-tag", type=str, required=True, help="new RL checkpoint tag (must not already exist)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K prompts")
parser.add_argument("--num-iterations", type=int, default=-1, help="policy batches to sample (-1 = full epochs)")
# Batch sizes / sampling
parser.add_argument("--rollout-batch-size", type=int, default=32, help="questions sampled per old-policy rollout")
parser.add_argument("--policy-batch-size", type=int, default=16, help="questions per optimizer update")
parser.add_argument("--device-batch-size", type=int, default=8, help="max responses per forward pass")
parser.add_argument("--num-samples", type=int, default=16, help="responses sampled per question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens generated per response")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (must be disabled for exact policy ratios)")
# GRPO
parser.add_argument("--ppo-epochs", type=int, default=1, help="optimization passes over each rollout")
parser.add_argument("--clip-epsilon", type=float, default=0.2, help="old-policy ratio clipping range")
parser.add_argument("--kl-beta", type=float, default=0.04, help="KL coefficient for the frozen SFT reference")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for Muon matrix parameters")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of each base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N policy updates (-1 disables)")
parser.add_argument("--eval-examples", type=int, default=400, help="number of GSM8K examples for pass@k")
parser.add_argument("--save-every", type=int, default=60, help="save every N policy updates (-1 = final only)")
args = parser.parse_args()
user_config = vars(args).copy()

if args.num_epochs <= 0:
    raise ValueError("num epochs must be positive")
if args.ppo_epochs <= 0:
    raise ValueError("ppo epochs must be positive")
if not 0 < args.clip_epsilon < 1:
    raise ValueError("clip epsilon must be between zero and one")
if args.kl_beta < 0:
    raise ValueError("KL beta must be non-negative")
validate_grpo_sampling_config(args.temperature, args.top_k)

# -----------------------------------------------------------------------------
# Init compute, logging, policy, and frozen reference
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

validate_grpo_batch_config(
    args.rollout_batch_size,
    args.policy_batch_size,
    args.num_samples,
    args.device_batch_size,
    ddp_world_size,
)

reservation = [None]
if master_process:
    try:
        reservation[0] = (reserve_grpo_checkpoint_dir(get_base_dir(), args.output_model_tag), None)
    except FileExistsError as exc:
        reservation[0] = (None, str(exc))
if ddp:
    dist.broadcast_object_list(reservation, src=0)
checkpoint_dir, reservation_error = reservation[0]
if reservation_error is not None:
    raise FileExistsError(reservation_error)

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-grpo",
    name=args.run,
    config=user_config,
)

model, tokenizer, meta = load_model(
    "sft",
    device,
    phase="eval",
    model_tag=args.model_tag,
    step=args.model_step,
)
reference_model, _, _ = load_model(
    "sft",
    device,
    phase="eval",
    model_tag=args.model_tag,
    step=args.model_step,
)
reference_model.requires_grad_(False)
reference_model.eval()
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Tasks and training schedule
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
rollout_plan = get_rollout_batch_plan(
    len(train_task),
    args.rollout_batch_size,
    args.policy_batch_size,
    args.num_epochs,
    args.num_iterations,
)
num_policy_batches = sum(rollout_plan)
num_steps = num_policy_batches * args.ppo_epochs
if num_steps == 0:
    raise ValueError("GRPO schedule contains no policy updates")
print0(f"Calculated rollout batches: {len(rollout_plan)}")
print0(f"Calculated policy updates: {num_steps}")
print0(f"Sequences per rollout: {args.rollout_batch_size * args.num_samples}")
print0(f"Sequences per policy update: {args.policy_batch_size * args.num_samples}")

groups_per_policy_batch = args.policy_batch_size // ddp_world_size
rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
example_iterator = itertools.cycle(rank_indices)


@torch.no_grad()
def collect_rollout_group(example_idx, rollout_idx):
    """Generate and score all responses for one question."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    terminal_tokens = {assistant_end, tokenizer.get_bos_token_id()}
    conversation = train_task[example_idx]
    prompt_tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(prompt_tokens)

    model.eval()
    generated_sequences = []
    generated_masks = []
    num_sampling_passes = args.num_samples // args.device_batch_size
    for sampling_pass in range(num_sampling_passes):
        seed = hash((rollout_idx, example_idx, sampling_pass)) & 0x7FFFFFFF
        sequences, masks = engine.generate_batch(
            prompt_tokens,
            num_samples=args.device_batch_size,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=seed,
            include_terminal_tokens=True,
        )
        generated_sequences.extend(sequences)
        generated_masks.extend(masks)

    rewards = []
    for sequence in generated_sequences:
        completion_tokens = sequence[prefix_length:]
        if completion_tokens and completion_tokens[-1] in terminal_tokens:
            completion_tokens = completion_tokens[:-1]
        generated_text = tokenizer.decode(completion_tokens)
        rewards.append(train_task.reward(conversation, generated_text))

    max_length = max(len(sequence) for sequence in generated_sequences)
    padded_sequences = [
        sequence + [assistant_end] * (max_length - len(sequence))
        for sequence in generated_sequences
    ]
    padded_masks = [
        mask + [0] * (max_length - len(mask))
        for mask in generated_masks
    ]
    ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1

    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    advantages = compute_group_advantages(rewards)

    old_logp_chunks = []
    ref_logp_chunks = []
    for start in range(0, args.num_samples, args.device_batch_size):
        end = start + args.device_batch_size
        inputs_chunk = inputs[start:end]
        targets_chunk = targets[start:end]
        old_logp = -model(
            inputs_chunk,
            targets_chunk,
            loss_reduction="none",
        ).view_as(inputs_chunk)
        ref_logp = -reference_model(
            inputs_chunk,
            targets_chunk,
            loss_reduction="none",
        ).view_as(inputs_chunk)
        old_logp_chunks.append(old_logp)
        ref_logp_chunks.append(ref_logp)

    completion_lengths = [
        len(sequence) - prefix_length
        for sequence in generated_sequences
    ]
    return {
        "inputs": inputs,
        "targets": targets,
        "rewards": rewards,
        "advantages": advantages,
        "old_logp": torch.cat(old_logp_chunks).detach(),
        "ref_logp": torch.cat(ref_logp_chunks).detach(),
        "completion_lengths": completion_lengths,
    }


# -----------------------------------------------------------------------------
# GSM8K pass@k evaluation
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        if num_samples > args.device_batch_size:
            raise ValueError("evaluation samples cannot exceed device batch size")
        generated_sequences, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sequence in generated_sequences:
            generated_text = tokenizer.decode(sequence[prefix_length:])
            outcomes.append({"is_correct": task.evaluate(conversation, generated_text)})
        yield {"idx": idx, "outcomes": outcomes}


def run_eval(step):
    model.eval()
    passk = torch.zeros(args.device_batch_size, device=device)
    records = list(run_gsm8k_eval(
        val_task,
        tokenizer,
        engine,
        num_samples=args.device_batch_size,
        max_examples=args.eval_examples,
        temperature=1.0,
    ))
    for k in range(1, args.device_batch_size + 1):
        passk[k - 1] = sum(
            any(outcome["is_correct"] for outcome in record["outcomes"][:k])
            for record in records
        )
    num_records = torch.tensor(len(records), dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)
    passk = passk / num_records.item()
    print_passk = [
        f"Pass@{k}: {passk[k - 1].item():.4f}"
        for k in range(1, args.device_batch_size + 1)
    ]
    print0(f"Step {step} | {', '.join(print_passk)}")
    wandb_run.log({
        "step": step,
        **{f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)},
    })


# -----------------------------------------------------------------------------
# Optimizer and policy updates
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(step):
    return 1.0 - step / num_steps


def reduce_metrics(values):
    tensor = torch.tensor(values, dtype=torch.float, device=device)
    if ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.tolist()


completed_steps = 0
for rollout_idx, policy_batches_in_rollout in enumerate(rollout_plan):
    local_group_count = policy_batches_in_rollout * groups_per_policy_batch
    rollout_groups = [
        collect_rollout_group(next(example_iterator), rollout_idx)
        for _ in range(local_group_count)
    ]

    for ppo_epoch in range(args.ppo_epochs):
        for policy_batch_idx in range(policy_batches_in_rollout):
            if args.eval_every > 0 and completed_steps % args.eval_every == 0:
                run_eval(completed_steps)

            group_start = policy_batch_idx * groups_per_policy_batch
            group_end = group_start + groups_per_policy_batch
            policy_groups = rollout_groups[group_start:group_end]
            num_microbatches = groups_per_policy_batch * (
                args.num_samples // args.device_batch_size
            )

            model.train()
            model.zero_grad(set_to_none=True)
            loss_total = 0.0
            metric_totals = {
                "reference_kl": 0.0,
                "old_policy_kl": 0.0,
                "clip_fraction": 0.0,
                "ratio_mean": 0.0,
            }

            for rollout_group in policy_groups:
                for start in range(0, args.num_samples, args.device_batch_size):
                    end = start + args.device_batch_size
                    inputs = rollout_group["inputs"][start:end]
                    targets = rollout_group["targets"][start:end]
                    current_logp = -model(
                        inputs,
                        targets,
                        loss_reduction="none",
                    ).view_as(inputs)
                    loss, metrics = compute_grpo_loss(
                        current_logp,
                        rollout_group["old_logp"][start:end],
                        rollout_group["ref_logp"][start:end],
                        rollout_group["advantages"][start:end],
                        targets >= 0,
                        clip_epsilon=args.clip_epsilon,
                        kl_beta=args.kl_beta,
                    )
                    (loss / num_microbatches).backward()
                    loss_total += loss.detach().item()
                    for name in metric_totals:
                        metric_totals[name] += metrics[name].item()

            lr_multiplier = get_lr_multiplier(completed_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_multiplier
            optimizer.step()
            model.zero_grad(set_to_none=True)
            completed_steps += 1

            rewards = torch.cat([group["rewards"] for group in policy_groups])
            advantages = torch.cat([group["advantages"] for group in policy_groups])
            lengths = [
                length
                for group in policy_groups
                for length in group["completion_lengths"]
            ]
            local_values = [
                loss_total / num_microbatches,
                metric_totals["reference_kl"] / num_microbatches,
                metric_totals["old_policy_kl"] / num_microbatches,
                metric_totals["clip_fraction"] / num_microbatches,
                metric_totals["ratio_mean"] / num_microbatches,
                rewards.mean().item(),
                rewards.std().item(),
                advantages.mean().item(),
                advantages.std().item(),
                sum(lengths) / len(lengths),
            ]
            (
                policy_loss,
                reference_kl,
                old_policy_kl,
                clip_fraction,
                ratio_mean,
                reward_mean,
                reward_std,
                advantage_mean,
                advantage_std,
                completion_length,
            ) = reduce_metrics(local_values)

            print0(
                f"Step {completed_steps}/{num_steps} | "
                f"loss: {policy_loss:.6f} | reward: {reward_mean:.4f} | "
                f"ref KL: {reference_kl:.6f} | old KL: {old_policy_kl:.6f} | "
                f"clip: {clip_fraction:.4f} | ratio: {ratio_mean:.4f}"
            )
            wandb_run.log({
                "step": completed_steps,
                "policy_loss": policy_loss,
                "reference_kl": reference_kl,
                "old_policy_kl": old_policy_kl,
                "clip_fraction": clip_fraction,
                "ratio_mean": ratio_mean,
                "reward": reward_mean,
                "reward_std": reward_std,
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "completion_length": completion_length,
                "lrm": lr_multiplier,
                "ppo_epoch": ppo_epoch,
            })

            if master_process and should_save_grpo_checkpoint(
                completed_steps,
                num_steps,
                args.save_every,
            ):
                save_checkpoint(
                    checkpoint_dir,
                    completed_steps,
                    model.state_dict(),
                    None,
                    {
                        "model_config": model.config.__dict__,
                        "algorithm": "grpo",
                        "input_model_tag": args.model_tag,
                        "input_model_step": args.model_step,
                        "training_config": user_config,
                    },
                )
                print0(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat GRPO", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
