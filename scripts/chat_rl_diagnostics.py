"""
Diagnose outcome-reward coverage on GSM8K without updating the model.

Evaluate one checkpoint with 8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_diagnostics -- evaluate \
    --source=sft --model-tag=d32-a800-bf16-L --step=242 \
    --label=d32-sft --output=d32_sft_diagnostics.json

Compare two completed runs:
python -m scripts.chat_rl_diagnostics compare d24_sft.json d32_sft.json
"""

import argparse
import json
import os
from pathlib import Path

from nanochat.grpo import compare_reward_records, summarize_reward_groups


def make_sampling_seed(base_seed, question_idx, sampling_pass, num_sampling_passes):
    """Return a stable, distinct seed for one question sampling pass."""
    return (base_seed + question_idx * num_sampling_passes + sampling_pass) % (2**31 - 1)


def write_result(output_path, payload, overwrite=False):
    """Atomically write a diagnostic JSON result."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)


def compare_result_files(left_path, right_path):
    """Load and compare two saved diagnostic runs."""
    left_path = Path(left_path)
    right_path = Path(right_path)
    with left_path.open(encoding="utf-8") as handle:
        left = json.load(handle)
    with right_path.open(encoding="utf-8") as handle:
        right = json.load(handle)
    return {
        "left_label": left.get("label", left_path.stem),
        "right_label": right.get("label", right_path.stem),
        "comparison": compare_reward_records(left["records"], right["records"]),
    }


def validate_evaluate_args(args):
    if args.max_examples <= 0:
        raise ValueError("max examples must be positive")
    if args.num_samples <= 0 or args.device_batch_size <= 0:
        raise ValueError("sample counts must be positive")
    if args.num_samples % args.device_batch_size:
        raise ValueError("num samples must be divisible by device batch size")
    if args.max_new_tokens <= 0:
        raise ValueError("max new tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")


def evaluate_checkpoint(args):
    import torch.distributed as dist

    from nanochat.checkpoint_manager import load_model
    from nanochat.common import (
        autodetect_device_type,
        compute_cleanup,
        compute_init,
        print0,
    )
    from nanochat.engine import Engine
    from tasks.gsm8k import GSM8K

    validate_evaluate_args(args)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    try:
        model, tokenizer, _ = load_model(
            args.source,
            device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step,
        )
        model.eval()
        engine = Engine(model, tokenizer)
        task = GSM8K(subset="main", split="test")
        num_examples = min(args.max_examples, len(task))
        num_sampling_passes = args.num_samples // args.device_batch_size

        local_records = []
        for idx in range(ddp_rank, num_examples, ddp_world_size):
            conversation = task[idx]
            prompt_tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(prompt_tokens)
            rewards = []
            for sampling_pass in range(num_sampling_passes):
                seed = make_sampling_seed(
                    args.seed,
                    idx,
                    sampling_pass,
                    num_sampling_passes,
                )
                sequences, _ = engine.generate_batch(
                    prompt_tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
                for sequence in sequences:
                    completion = tokenizer.decode(sequence[prefix_length:])
                    rewards.append(task.evaluate(conversation, completion))
            local_records.append({"idx": idx, "rewards": rewards})
            if len(local_records) % 10 == 0:
                print(f"Rank {ddp_rank} | {len(local_records)} questions", flush=True)

        if ddp:
            gathered_records = [None] * ddp_world_size if master_process else None
            dist.gather_object(local_records, gathered_records, dst=0)
        else:
            gathered_records = [local_records]

        if master_process:
            records = [
                record
                for rank_records in gathered_records
                for record in rank_records
            ]
            records.sort(key=lambda record: record["idx"])
            if len(records) != num_examples:
                raise RuntimeError(f"expected {num_examples} records, received {len(records)}")
            summary = summarize_reward_groups([record["rewards"] for record in records])
            payload = {
                "label": args.label,
                "config": {
                    "source": args.source,
                    "model_tag": args.model_tag,
                    "step": args.step,
                    "max_examples": num_examples,
                    "num_samples": args.num_samples,
                    "device_batch_size": args.device_batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "seed": args.seed,
                },
                "summary": summary,
                "records": records,
            }
            write_result(args.output, payload, overwrite=args.overwrite)
            print0(json.dumps(summary, indent=2))
            print0(f"Saved diagnostics to {args.output}")
    finally:
        compute_cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="GSM8K reward-group diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="sample one existing checkpoint")
    evaluate.add_argument("--source", choices=["sft", "rl"], required=True)
    evaluate.add_argument("--model-tag", required=True)
    evaluate.add_argument("--step", type=int, required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-examples", type=int, default=400)
    evaluate.add_argument("--num-samples", type=int, default=16)
    evaluate.add_argument("--device-batch-size", type=int, default=8)
    evaluate.add_argument("--max-new-tokens", type=int, default=256)
    evaluate.add_argument("--temperature", type=float, default=1.0)
    evaluate.add_argument("--top-k", type=int, default=0)
    evaluate.add_argument("--seed", type=int, default=1337)
    evaluate.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"])
    evaluate.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare", help="compare two diagnostic JSON files")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--output")
    compare.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "evaluate":
        evaluate_checkpoint(args)
        return

    result = compare_result_files(args.left, args.right)
    if args.output:
        write_result(args.output, result, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
