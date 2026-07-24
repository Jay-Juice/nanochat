"""
Small tensor helpers for Group Relative Policy Optimization (GRPO).
"""

import torch


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
