"""
Small tensor helpers for Group Relative Policy Optimization (GRPO).
"""

import os

import torch


def _validate_reward_groups(reward_groups):
    if not reward_groups:
        raise ValueError("reward groups must contain at least one question")
    num_samples = len(reward_groups[0])
    if num_samples == 0:
        raise ValueError("reward groups must contain at least one sample")
    for rewards in reward_groups:
        if len(rewards) != num_samples:
            raise ValueError("reward groups must have the same number of samples")
        if any(reward not in (0, 1) for reward in rewards):
            raise ValueError("rewards must be binary values")
    return num_samples


def summarize_reward_groups(reward_groups):
    """Summarize binary outcome-reward groups from multiple questions."""
    num_samples = _validate_reward_groups(reward_groups)
    num_questions = len(reward_groups)
    correct_counts = [sum(rewards) for rewards in reward_groups]
    group_counts = {
        "all_wrong": sum(count == 0 for count in correct_counts),
        "mixed": sum(0 < count < num_samples for count in correct_counts),
        "all_correct": sum(count == num_samples for count in correct_counts),
    }
    return {
        "num_questions": num_questions,
        "num_samples": num_samples,
        "group_counts": group_counts,
        "group_fractions": {
            name: count / num_questions
            for name, count in group_counts.items()
        },
        "mean_reward": sum(correct_counts) / (num_questions * num_samples),
        "pass_at_k": {
            str(k): sum(any(rewards[:k]) for rewards in reward_groups) / num_questions
            for k in range(1, num_samples + 1)
        },
    }


def _index_reward_records(records):
    if not records:
        raise ValueError("reward records must contain at least one question")
    indexed = {}
    reward_groups = []
    for record in records:
        idx = record["idx"]
        if idx in indexed:
            raise ValueError(f"duplicate question index: {idx}")
        rewards = record["rewards"]
        indexed[idx] = rewards
        reward_groups.append(rewards)
    num_samples = _validate_reward_groups(reward_groups)
    return indexed, num_samples


def compare_reward_records(left_records, right_records):
    """Compare questions solved at least once in two diagnostic runs."""
    left, left_num_samples = _index_reward_records(left_records)
    right, right_num_samples = _index_reward_records(right_records)
    if left.keys() != right.keys():
        raise ValueError("diagnostic runs must contain the same question indices")
    if left_num_samples != right_num_samples:
        raise ValueError("diagnostic runs must use matching sample counts")

    left_solved = {idx for idx, rewards in left.items() if any(rewards)}
    right_solved = {idx for idx, rewards in right.items() if any(rewards)}
    intersection = left_solved & right_solved
    union = left_solved | right_solved
    return {
        "num_questions": len(left),
        "num_samples": left_num_samples,
        "left_solved": len(left_solved),
        "right_solved": len(right_solved),
        "intersection": len(intersection),
        "union": len(union),
        "left_only": len(left_solved - right_solved),
        "right_only": len(right_solved - left_solved),
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }


def _masked_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_group_advantages(rewards, eps=1e-6):
    """Z-score scalar rewards within one prompt group."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two scalar rewards per group")
    centered = rewards - rewards.mean()
    std = rewards.std()
    if std <= eps:
        return torch.zeros_like(rewards)
    return centered / (std + eps)


def get_rollout_batch_plan(
    dataset_size,
    rollout_batch_size,
    policy_batch_size,
    num_epochs=1,
    num_iterations=-1,
):
    """Return the number of policy updates contained in each rollout batch."""
    if rollout_batch_size % policy_batch_size != 0:
        raise ValueError("rollout batch size must be divisible by policy batch size")
    if num_iterations > 0:
        total_updates = num_iterations
    else:
        total_updates = (dataset_size // policy_batch_size) * num_epochs
    updates_per_rollout = rollout_batch_size // policy_batch_size
    return [
        min(updates_per_rollout, total_updates - start)
        for start in range(0, total_updates, updates_per_rollout)
    ]


def validate_grpo_batch_config(
    rollout_batch_size,
    policy_batch_size,
    num_samples,
    device_batch_size,
    world_size,
):
    """Validate batch relationships required by synchronous distributed GRPO."""
    values = (rollout_batch_size, policy_batch_size, num_samples, device_batch_size, world_size)
    if any(value <= 0 for value in values):
        raise ValueError("GRPO batch sizes and world size must be positive")
    if rollout_batch_size % policy_batch_size != 0:
        raise ValueError("rollout batch size must be divisible by policy batch size")
    if rollout_batch_size % world_size or policy_batch_size % world_size:
        raise ValueError("rollout and policy batch sizes must be divisible by world size")
    if num_samples < 2:
        raise ValueError("GRPO requires at least two samples per question")
    if num_samples % device_batch_size:
        raise ValueError("num samples must be divisible by device batch size")


def validate_grpo_sampling_config(temperature, top_k):
    """Require rollout sampling to match the raw policy log probabilities."""
    if temperature != 1.0:
        raise ValueError("GRPO training temperature must be 1.0 to match recorded policy log probabilities")
    if top_k not in (None, 0):
        raise ValueError("GRPO training top-k must be disabled to match recorded policy log probabilities")


def should_save_grpo_checkpoint(completed_steps, total_steps, save_every):
    """Return whether the policy after completed_steps updates should be saved."""
    return (
        completed_steps == total_steps
        or save_every > 0 and completed_steps > 0 and completed_steps % save_every == 0
    )


def reserve_grpo_checkpoint_dir(base_dir, output_model_tag):
    """Atomically reserve a new GRPO checkpoint directory."""
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_model_tag)
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        raise FileExistsError(f"Output model tag already exists: {output_model_tag}")
    return checkpoint_dir


def compute_grpo_loss(
    current_logp,
    old_logp,
    ref_logp,
    advantages,
    response_mask,
    clip_epsilon=0.2,
    kl_beta=0.04,
):
    """Compute the original sequence-normalized clipped GRPO objective."""
    mask = response_mask.to(current_logp.dtype)
    log_ratio = current_logp - old_logp
    ratio = log_ratio.exp()
    token_advantages = advantages.unsqueeze(-1)
    unclipped = ratio * token_advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * token_advantages
    policy_objective = torch.minimum(unclipped, clipped)

    ref_log_ratio = ref_logp - current_logp
    reference_kl = ref_log_ratio.exp() - ref_log_ratio - 1
    token_loss = -policy_objective + kl_beta * reference_kl

    lengths = mask.sum(dim=-1)
    valid_sequences = lengths > 0
    if not valid_sequences.any():
        raise ValueError("GRPO mini-batch contains no sampled response tokens")
    sequence_loss = (token_loss * mask).sum(dim=-1) / lengths.clamp_min(1)
    loss = sequence_loss[valid_sequences].mean()

    clipped_tokens = (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
    old_policy_kl = ratio - 1 - log_ratio
    metrics = {
        "reference_kl": _masked_token_mean(reference_kl.detach(), mask),
        "old_policy_kl": _masked_token_mean(old_policy_kl.detach(), mask),
        "clip_fraction": _masked_token_mean(clipped_tokens.detach().to(mask.dtype), mask),
        "ratio_mean": _masked_token_mean(ratio.detach(), mask),
    }
    return loss, metrics
