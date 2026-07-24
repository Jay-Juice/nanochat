"""
Small tensor helpers for Group Relative Policy Optimization (GRPO).
"""

import torch


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
    log_ratio = (current_logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    token_advantages = advantages.unsqueeze(-1)
    unclipped = ratio * token_advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * token_advantages
    policy_objective = torch.minimum(unclipped, clipped)

    ref_log_ratio = (ref_logp - current_logp).clamp(-20.0, 20.0)
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
