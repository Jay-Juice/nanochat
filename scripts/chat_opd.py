"""
On-Policy Distillation (OPD) on GSM8K.

The student samples one response per prompt. A frozen teacher scores those same
tokens, and their log-probability gap becomes the token-level policy advantage.

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_opd -- \
    --student-model-tag=d24-a800-bf16-L \
    --student-model-step=466 \
    --teacher-model-tag=d32-a800-bf16-L-math15 \
    --teacher-model-step=180 \
    --output-model-tag=d24-a800-opd-d32math15-v1 \
    --run=dummy
"""

import argparse
import itertools

import torch
import torch.distributed as dist
import wandb
from torch.nn.utils.rnn import pad_sequence

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
from nanochat.grpo import get_rollout_batch_plan
from nanochat.opd import (
    compute_opd_loss,
    get_opd_eval_seed,
    get_opd_lr_multiplier,
    reserve_opd_checkpoint_dir,
    should_save_opd_checkpoint,
    validate_opd_batch_config,
    validate_opd_model_configs,
    validate_opd_sampling_config,
)
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(
    description="On-Policy Distillation on GSM8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading and saving
parser.add_argument("--student-model-tag", type=str, required=True, help="RL student model tag to load from")
parser.add_argument("--student-model-step", type=int, required=True, help="RL student model step to load from")
parser.add_argument("--teacher-model-tag", type=str, required=True, help="frozen RL teacher model tag to load from")
parser.add_argument("--teacher-model-step", type=int, required=True, help="frozen RL teacher model step to load from")
parser.add_argument("--output-model-tag", type=str, required=True, help="new OPD checkpoint tag (must not already exist)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K training prompts")
parser.add_argument("--num-iterations", type=int, default=-1, help="policy batches to sample (-1 = full epochs)")
# Batch sizes
parser.add_argument("--rollout-batch-size", type=int, default=64, help="questions sampled per student rollout")
parser.add_argument("--policy-batch-size", type=int, default=64, help="questions per optimizer update")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens generated per response")
parser.add_argument("--temperature", type=float, default=1.0, help="student rollout temperature")
parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (must be disabled for exact OPD)")
# OPD
parser.add_argument("--ppo-epochs", type=int, default=1, help="optimization passes over each rollout")
parser.add_argument("--clip-epsilon", type=float, default=0.2, help="old-student policy ratio clipping range")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for Muon matrix parameters")
parser.add_argument("--init-lr-frac", type=float, default=0.005, help="initial LR as fraction of each base LR")
parser.add_argument("--lr-schedule", choices=("constant", "linear"), default="constant", help="learning-rate schedule")
# Evaluation / checkpointing
parser.add_argument("--validation-examples", type=int, default=500, help="GSM8K train examples held out from OPD")
parser.add_argument("--eval-every", type=int, default=20, help="evaluate holdout pass@k every N policy updates (-1 disables)")
parser.add_argument("--eval-examples", type=int, default=400, help="number of held-out GSM8K examples for pass@k")
parser.add_argument("--eval-samples", type=int, default=8, help="responses per held-out question")
parser.add_argument("--eval-seed", type=int, default=1337, help="fixed base seed for holdout sampling")
parser.add_argument("--save-every", type=int, default=20, help="save every N policy updates (-1 = final only)")
args = parser.parse_args()
user_config = vars(args).copy()

if args.num_epochs <= 0:
    raise ValueError("num epochs must be positive")
if args.ppo_epochs <= 0:
    raise ValueError("ppo epochs must be positive")
if not 0 < args.clip_epsilon < 1:
    raise ValueError("clip epsilon must be between zero and one")
if args.max_new_tokens <= 0:
    raise ValueError("max new tokens must be positive")
if args.validation_examples <= 0:
    raise ValueError("validation examples must be positive")
if args.eval_examples <= 0 or args.eval_samples <= 0:
    raise ValueError("evaluation examples and samples must be positive")
validate_opd_sampling_config(args.temperature, args.top_k)

# -----------------------------------------------------------------------------
# Init compute, logging, student, and frozen teacher
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

validate_opd_batch_config(
    args.rollout_batch_size,
    args.policy_batch_size,
    ddp_world_size,
)

reservation = [None]
if master_process:
    try:
        reservation[0] = (reserve_opd_checkpoint_dir(get_base_dir(), args.output_model_tag), None)
    except FileExistsError as exc:
        reservation[0] = (None, str(exc))
if ddp:
    dist.broadcast_object_list(reservation, src=0)
checkpoint_dir, reservation_error = reservation[0]
if reservation_error is not None:
    raise FileExistsError(reservation_error)

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-opd",
    name=args.run,
    config=user_config,
)

model, tokenizer, student_meta = load_model(
    "rl",
    device,
    phase="eval",
    model_tag=args.student_model_tag,
    step=args.student_model_step,
)
teacher_model, _, teacher_meta = load_model(
    "rl",
    device,
    phase="eval",
    model_tag=args.teacher_model_tag,
    step=args.teacher_model_step,
)
validate_opd_model_configs(model.config, teacher_model.config)
teacher_model.requires_grad_(False)
teacher_model.eval()
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Tasks and training schedule
train_task = GSM8K(subset="main", split="train")
full_train_size = len(train_task)
if args.validation_examples >= full_train_size:
    raise ValueError("validation holdout must leave GSM8K prompts for training")
validation_start = full_train_size - args.validation_examples
train_task.stop = validation_start
val_task = GSM8K(subset="main", split="train", start=validation_start)

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
    raise ValueError("OPD schedule contains no policy updates")
print0(f"Training prompts: {len(train_task)}")
print0(f"Validation prompts: {len(val_task)}")
print0(f"Calculated rollout batches: {len(rollout_plan)}")
print0(f"Calculated policy updates: {num_steps}")
print0(f"Prompts used per OPD epoch: {(len(train_task) // args.policy_batch_size) * args.policy_batch_size}")

groups_per_policy_batch = args.policy_batch_size // ddp_world_size
rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
example_iterator = itertools.cycle(rank_indices)
assistant_end = tokenizer.encode_special("<|assistant_end|>")
terminal_tokens = {assistant_end, tokenizer.get_bos_token_id()}


@torch.no_grad()
def collect_rollout(example_idx, rollout_idx):
    """Generate one student response and score it under student and teacher."""
    conversation = train_task[example_idx]
    prompt_tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(prompt_tokens)

    model.eval()
    sequences, masks = engine.generate_batch(
        prompt_tokens,
        num_samples=1,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=hash((rollout_idx, example_idx)) & 0x7FFFFFFF,
        include_terminal_tokens=True,
    )
    sequence = sequences[0]
    sequence_mask = masks[0]
    completion_tokens = sequence[prefix_length:]
    if completion_tokens and completion_tokens[-1] in terminal_tokens:
        completion_tokens = completion_tokens[:-1]
    reward = train_task.reward(conversation, tokenizer.decode(completion_tokens))

    ids = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    mask_ids = torch.tensor(sequence_mask, dtype=torch.long, device=device).unsqueeze(0)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1
    old_logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)
    teacher_logp = -teacher_model(inputs, targets, loss_reduction="none").view_as(inputs)
    return {
        "inputs": inputs.squeeze(0),
        "targets": targets.squeeze(0),
        "old_logp": old_logp.squeeze(0).detach(),
        "teacher_logp": teacher_logp.squeeze(0).detach(),
        "reward": reward,
        "completion_length": len(sequence) - prefix_length,
    }


def collate_rollouts(rollouts):
    """Pad variable-length rollout tensors into one local policy batch."""
    return {
        "inputs": pad_sequence(
            [rollout["inputs"] for rollout in rollouts],
            batch_first=True,
            padding_value=assistant_end,
        ),
        "targets": pad_sequence(
            [rollout["targets"] for rollout in rollouts],
            batch_first=True,
            padding_value=-1,
        ),
        "old_logp": pad_sequence(
            [rollout["old_logp"] for rollout in rollouts],
            batch_first=True,
            padding_value=0.0,
        ),
        "teacher_logp": pad_sequence(
            [rollout["teacher_logp"] for rollout in rollouts],
            batch_first=True,
            padding_value=0.0,
        ),
    }


# -----------------------------------------------------------------------------
# GSM8K holdout pass@k evaluation
@torch.no_grad()
def run_eval(step):
    model.eval()
    passk = torch.zeros(args.eval_samples, device=device)
    num_records = 0
    max_examples = min(args.eval_examples, len(val_task))
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = val_task[idx]
        prompt_tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(prompt_tokens)
        sequences, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=args.eval_samples,
            max_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=0,
            seed=get_opd_eval_seed(idx, args.eval_seed),
        )
        outcomes = [
            val_task.evaluate(conversation, tokenizer.decode(sequence[prefix_length:]))
            for sequence in sequences
        ]
        for k in range(1, args.eval_samples + 1):
            passk[k - 1] += any(outcomes[:k])
        num_records += 1

    num_records = torch.tensor(num_records, dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)
    passk = passk / num_records.item()
    print_passk = [
        f"Pass@{k}: {passk[k - 1].item():.4f}"
        for k in range(1, args.eval_samples + 1)
    ]
    print0(f"Step {step} | holdout | {', '.join(print_passk)}")
    wandb_run.log({
        "step": step,
        **{f"holdout_pass@{k}": passk[k - 1].item() for k in range(1, args.eval_samples + 1)},
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

def reduce_policy_metrics(loss, metrics, mask, rollouts):
    """Reduce token-weighted objective metrics and response-level diagnostics."""
    token_count = mask.sum().to(torch.float32)
    token_values = torch.stack([
        loss.detach() * token_count,
        metrics["teacher_kl"] * token_count,
        metrics["old_policy_kl"] * token_count,
        metrics["clip_fraction"] * token_count,
        metrics["ratio_mean"] * token_count,
        metrics["advantage_mean"] * token_count,
        (metrics["advantage_std"].square() + metrics["advantage_mean"].square()) * token_count,
        token_count,
    ])
    response_values = torch.tensor([
        sum(rollout["reward"] for rollout in rollouts),
        sum(rollout["completion_length"] for rollout in rollouts),
        len(rollouts),
    ], dtype=torch.float32, device=device)
    if ddp:
        dist.all_reduce(token_values, op=dist.ReduceOp.SUM)
        dist.all_reduce(response_values, op=dist.ReduceOp.SUM)

    token_total = token_values[-1]
    token_means = token_values[:-1] / token_total
    advantage_std = (token_means[6] - token_means[5].square()).clamp_min(0).sqrt()
    response_means = response_values[:2] / response_values[2]
    return (
        token_means[0].item(),
        token_means[1].item(),
        token_means[2].item(),
        token_means[3].item(),
        token_means[4].item(),
        token_means[5].item(),
        advantage_std.item(),
        response_means[0].item(),
        response_means[1].item(),
    )


completed_steps = 0
for rollout_idx, policy_batches_in_rollout in enumerate(rollout_plan):
    local_rollout_count = policy_batches_in_rollout * groups_per_policy_batch
    rollout_groups = [
        collect_rollout(next(example_iterator), rollout_idx)
        for _ in range(local_rollout_count)
    ]

    for ppo_epoch in range(args.ppo_epochs):
        for policy_batch_idx in range(policy_batches_in_rollout):
            if args.eval_every > 0 and completed_steps % args.eval_every == 0:
                run_eval(completed_steps)

            group_start = policy_batch_idx * groups_per_policy_batch
            group_end = group_start + groups_per_policy_batch
            policy_groups = rollout_groups[group_start:group_end]
            policy_batch = collate_rollouts(policy_groups)
            response_mask = policy_batch["targets"] >= 0

            model.train()
            model.zero_grad(set_to_none=True)
            current_logp = -model(
                policy_batch["inputs"],
                policy_batch["targets"],
                loss_reduction="none",
            ).view_as(policy_batch["inputs"])
            loss, metrics = compute_opd_loss(
                current_logp,
                policy_batch["old_logp"],
                policy_batch["teacher_logp"],
                response_mask,
                clip_epsilon=args.clip_epsilon,
            )

            local_token_count = response_mask.sum().to(torch.float32)
            global_token_count = local_token_count.clone()
            if ddp:
                dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
            gradient_scale = ddp_world_size * local_token_count / global_token_count
            (loss * gradient_scale).backward()

            lr_multiplier = get_opd_lr_multiplier(completed_steps, num_steps, args.lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_multiplier
            optimizer.step()
            model.zero_grad(set_to_none=True)
            completed_steps += 1

            (
                policy_loss,
                teacher_kl,
                old_policy_kl,
                clip_fraction,
                ratio_mean,
                advantage_mean,
                advantage_std,
                reward_mean,
                completion_length,
            ) = reduce_policy_metrics(loss, metrics, response_mask, policy_groups)

            print0(
                f"Step {completed_steps}/{num_steps} | "
                f"loss: {policy_loss:.6f} | reward: {reward_mean:.4f} | "
                f"teacher KL: {teacher_kl:.6f} | old KL: {old_policy_kl:.6f} | "
                f"adv: {advantage_mean:.6f} | clip: {clip_fraction:.4f} | "
                f"ratio: {ratio_mean:.4f}"
            )
            wandb_run.log({
                "step": completed_steps,
                "policy_loss": policy_loss,
                "teacher_kl": teacher_kl,
                "old_policy_kl": old_policy_kl,
                "clip_fraction": clip_fraction,
                "ratio_mean": ratio_mean,
                "reward": reward_mean,
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "completion_length": completion_length,
                "lrm": lr_multiplier,
                "ppo_epoch": ppo_epoch,
            })

            if master_process and should_save_opd_checkpoint(
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
                        "algorithm": "opd",
                        "student_model_tag": args.student_model_tag,
                        "student_model_step": args.student_model_step,
                        "student_model_meta": student_meta,
                        "teacher_model_tag": args.teacher_model_tag,
                        "teacher_model_step": args.teacher_model_step,
                        "teacher_model_config": teacher_model.config.__dict__,
                        "teacher_model_meta": teacher_meta,
                        "validation_examples": args.validation_examples,
                        "training_config": user_config,
                    },
                )
                print0(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat OPD", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
