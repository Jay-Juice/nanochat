"""
Small tensor helpers for On-Policy Distillation (OPD).
"""

import os

import torch


def get_opd_lr_multiplier(step, total_steps, schedule):
    """Return the learning-rate multiplier for an OPD optimizer step."""
    if total_steps <= 0:
        raise ValueError("total OPD steps must be positive")
    if not 0 <= step <= total_steps:
        raise ValueError("OPD step must be between zero and total steps")
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return 1.0 - step / total_steps
    raise ValueError(f"unknown OPD learning-rate schedule: {schedule}")


def get_opd_eval_seed(example_idx, base_seed):
    """Return a stable per-example seed independent of the training step."""
    if example_idx < 0:
        raise ValueError("OPD evaluation example index must be non-negative")
    return (base_seed + 1_000_003 * example_idx + 0xE1A1) & 0x7FFFFFFF


def _masked_token_mean(values, mask):
    return values[mask].mean()


def compute_opd_advantages(old_logp, teacher_logp):
    """Return the detached token advantage for standard reverse-KL OPD."""
    if old_logp.shape != teacher_logp.shape:
        raise ValueError("student and teacher log probabilities must have the same shape")
    return (teacher_logp - old_logp).detach()


def compute_opd_loss(
    current_logp,
    old_logp,
    teacher_logp,
    response_mask,
    clip_epsilon=0.2,
):
    """Compute the token-mean clipped policy objective for standard OPD."""
    shapes = {current_logp.shape, old_logp.shape, teacher_logp.shape, response_mask.shape}
    if len(shapes) != 1:
        raise ValueError("OPD log probabilities and response mask must have the same shape")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be between zero and one")

    mask = response_mask.bool()
    if not mask.any():
        raise ValueError("OPD mini-batch contains no sampled response tokens")

    old_logp = old_logp.float()
    teacher_logp = teacher_logp.float()
    log_ratio = torch.where(mask, current_logp.float() - old_logp, 0.0)
    advantages = torch.where(mask, compute_opd_advantages(old_logp, teacher_logp), 0.0)
    ratio = log_ratio.exp()

    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    token_loss = -torch.minimum(unclipped, clipped)
    loss = _masked_token_mean(token_loss, mask)

    teacher_log_ratio = torch.where(mask, teacher_logp - old_logp, 0.0)
    teacher_kl = teacher_log_ratio.exp() - teacher_log_ratio - 1
    old_policy_kl = ratio - log_ratio - 1
    clipped_tokens = (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
    valid_advantages = advantages[mask]
    metrics = {
        "teacher_kl": _masked_token_mean(teacher_kl.detach(), mask),
        "old_policy_kl": _masked_token_mean(old_policy_kl.detach(), mask),
        "clip_fraction": _masked_token_mean(clipped_tokens.float(), mask),
        "ratio_mean": _masked_token_mean(ratio.detach(), mask),
        "advantage_mean": valid_advantages.mean(),
        "advantage_std": valid_advantages.std(unbiased=False),
    }
    return loss, metrics


def validate_opd_batch_config(rollout_batch_size, policy_batch_size, world_size):
    """Validate batch relationships required by synchronous distributed OPD."""
    if any(value <= 0 for value in (rollout_batch_size, policy_batch_size, world_size)):
        raise ValueError("OPD batch sizes and world size must be positive")
    if rollout_batch_size != policy_batch_size:
        raise ValueError("rollout batch size must equal policy batch size for exact on-policy updates")
    if rollout_batch_size % world_size or policy_batch_size % world_size:
        raise ValueError("rollout and policy batch sizes must be divisible by world size")


def validate_opd_sampling_config(temperature, top_k):
    """Require rollout sampling to match the recorded student log probabilities."""
    if temperature != 1.0:
        raise ValueError("OPD training temperature must be 1.0 to match recorded student log probabilities")
    if top_k not in (None, 0):
        raise ValueError("OPD training top-k must be disabled to match recorded student log probabilities")


def validate_opd_model_configs(student_config, teacher_config):
    """Validate token and context compatibility while allowing different sizes."""
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("student and teacher vocabularies must match")
    if teacher_config.sequence_len < student_config.sequence_len:
        raise ValueError("teacher context window must cover the student context window")


def should_save_opd_checkpoint(completed_steps, total_steps, save_every):
    """Return whether the policy after completed_steps updates should be saved."""
    return (
        completed_steps == total_steps
        or save_every > 0 and completed_steps > 0 and completed_steps % save_every == 0
    )


def reserve_opd_checkpoint_dir(base_dir, output_model_tag):
    """Atomically reserve a new OPD checkpoint directory."""
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_model_tag)
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        raise FileExistsError(f"Output model tag already exists: {output_model_tag}")
    return checkpoint_dir
