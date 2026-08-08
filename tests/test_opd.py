from types import SimpleNamespace

import pytest
import torch

from nanochat.opd import (
    compute_opd_advantages,
    compute_opd_loss,
    get_opd_eval_seed,
    get_opd_lr_multiplier,
    reserve_opd_checkpoint_dir,
    should_save_opd_checkpoint,
    validate_opd_batch_config,
    validate_opd_model_configs,
    validate_opd_sampling_config,
)


@pytest.mark.parametrize("step", [0, 54, 108])
def test_constant_lr_multiplier_stays_at_one(step):
    assert get_opd_lr_multiplier(step, total_steps=108, schedule="constant") == 1.0


def test_linear_lr_multiplier_preserves_v1_schedule():
    assert get_opd_lr_multiplier(0, total_steps=108, schedule="linear") == 1.0
    assert get_opd_lr_multiplier(54, total_steps=108, schedule="linear") == 0.5
    assert get_opd_lr_multiplier(108, total_steps=108, schedule="linear") == 0.0


def test_lr_multiplier_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="learning-rate schedule"):
        get_opd_lr_multiplier(0, total_steps=108, schedule="cosine")


def test_eval_seed_is_stable_for_the_same_example():
    assert get_opd_eval_seed(17, base_seed=1337) == get_opd_eval_seed(17, base_seed=1337)


def test_eval_seed_varies_by_example():
    assert get_opd_eval_seed(17, base_seed=1337) != get_opd_eval_seed(18, base_seed=1337)


def test_opd_advantage_is_teacher_logp_minus_old_student_logp():
    old_logp = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    teacher_logp = torch.tensor([[-1.5, -1.25]], requires_grad=True)

    advantages = compute_opd_advantages(old_logp, teacher_logp)

    assert torch.allclose(advantages, torch.tensor([[0.5, -0.25]]))
    assert not advantages.requires_grad


def test_opd_advantage_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_opd_advantages(torch.zeros(1, 2), torch.zeros(2, 1))


def test_positive_teacher_advantage_increases_sampled_token_probability():
    current_logp = torch.tensor([[-2.0]], requires_grad=True)

    loss, _ = compute_opd_loss(
        current_logp,
        old_logp=torch.tensor([[-2.0]]),
        teacher_logp=torch.tensor([[-1.0]]),
        response_mask=torch.tensor([[True]]),
    )
    loss.backward()

    assert current_logp.grad.item() < 0


def test_negative_teacher_advantage_decreases_sampled_token_probability():
    current_logp = torch.tensor([[-2.0]], requires_grad=True)

    loss, _ = compute_opd_loss(
        current_logp,
        old_logp=torch.tensor([[-2.0]]),
        teacher_logp=torch.tensor([[-3.0]]),
        response_mask=torch.tensor([[True]]),
    )
    loss.backward()

    assert current_logp.grad.item() > 0


def test_equal_student_and_teacher_have_zero_opd_loss():
    current_logp = torch.tensor([[-2.0, -1.0]], requires_grad=True)

    loss, metrics = compute_opd_loss(
        current_logp,
        old_logp=current_logp.detach(),
        teacher_logp=current_logp.detach(),
        response_mask=torch.ones_like(current_logp, dtype=torch.bool),
    )

    assert torch.equal(loss, torch.tensor(0.0))
    assert torch.equal(metrics["teacher_kl"], torch.tensor(0.0))
    assert torch.equal(metrics["advantage_mean"], torch.tensor(0.0))


def test_positive_advantage_clips_large_policy_ratio():
    current_logp = torch.tensor([[0.4]], requires_grad=True)

    loss, metrics = compute_opd_loss(
        current_logp,
        old_logp=torch.tensor([[0.0]]),
        teacher_logp=torch.tensor([[1.0]]),
        response_mask=torch.tensor([[True]]),
    )

    assert torch.allclose(loss, torch.tensor(-1.2))
    assert torch.equal(metrics["clip_fraction"], torch.tensor(1.0))


def test_negative_advantage_clips_small_policy_ratio():
    current_logp = torch.tensor([[-0.4]], requires_grad=True)

    loss, _ = compute_opd_loss(
        current_logp,
        old_logp=torch.tensor([[0.0]]),
        teacher_logp=torch.tensor([[-1.0]]),
        response_mask=torch.tensor([[True]]),
    )

    assert torch.allclose(loss, torch.tensor(0.8))


def test_opd_loss_uses_token_mean_and_ignores_masked_tokens():
    current_logp = torch.tensor([[0.0, 100.0], [0.0, 0.0]], requires_grad=True)
    old_logp = torch.zeros_like(current_logp)
    teacher_logp = torch.tensor([[1.0, -100.0], [2.0, 2.0]])
    response_mask = torch.tensor([[True, False], [True, True]])

    loss, metrics = compute_opd_loss(
        current_logp,
        old_logp,
        teacher_logp,
        response_mask,
    )

    assert torch.allclose(loss, torch.tensor(-5.0 / 3.0))
    assert torch.equal(metrics["clip_fraction"], torch.tensor(0.0))
    assert torch.allclose(metrics["advantage_mean"], torch.tensor(5.0 / 3.0))


def test_opd_loss_rejects_fully_masked_batch():
    logp = torch.zeros((1, 2), requires_grad=True)

    with pytest.raises(ValueError, match="no sampled response tokens"):
        compute_opd_loss(
            logp,
            old_logp=logp.detach(),
            teacher_logp=logp.detach(),
            response_mask=torch.zeros_like(logp, dtype=torch.bool),
        )


def test_opd_loss_requires_matching_tensor_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_opd_loss(
            current_logp=torch.zeros(1, 2),
            old_logp=torch.zeros(1, 2),
            teacher_logp=torch.zeros(1, 2),
            response_mask=torch.ones(2, 1, dtype=torch.bool),
        )


def test_batch_config_accepts_eight_gpu_schedule():
    validate_opd_batch_config(
        rollout_batch_size=16,
        policy_batch_size=16,
        world_size=8,
    )


def test_batch_config_requires_world_size_divisibility():
    with pytest.raises(ValueError, match="world size"):
        validate_opd_batch_config(
            rollout_batch_size=12,
            policy_batch_size=12,
            world_size=8,
        )


def test_batch_config_requires_exact_on_policy_updates():
    with pytest.raises(ValueError, match="must equal"):
        validate_opd_batch_config(
            rollout_batch_size=32,
            policy_batch_size=16,
            world_size=8,
        )


def test_sampling_config_matches_recorded_student_logprobs():
    validate_opd_sampling_config(temperature=1.0, top_k=0)


def test_sampling_config_rejects_temperature_transform():
    with pytest.raises(ValueError, match="temperature"):
        validate_opd_sampling_config(temperature=0.8, top_k=0)


def test_sampling_config_rejects_top_k_truncation():
    with pytest.raises(ValueError, match="top-k"):
        validate_opd_sampling_config(temperature=1.0, top_k=50)


def test_model_configs_allow_different_depth_and_width():
    student = SimpleNamespace(vocab_size=32768, sequence_len=2048, n_layer=24, n_embd=1536)
    teacher = SimpleNamespace(vocab_size=32768, sequence_len=2048, n_layer=32, n_embd=2048)

    validate_opd_model_configs(student, teacher)


def test_model_configs_require_matching_vocabularies():
    student = SimpleNamespace(vocab_size=32768, sequence_len=2048)
    teacher = SimpleNamespace(vocab_size=32000, sequence_len=2048)

    with pytest.raises(ValueError, match="vocabularies"):
        validate_opd_model_configs(student, teacher)


def test_teacher_must_cover_student_context_window():
    student = SimpleNamespace(vocab_size=32768, sequence_len=2048)
    teacher = SimpleNamespace(vocab_size=32768, sequence_len=1024)

    with pytest.raises(ValueError, match="context window"):
        validate_opd_model_configs(student, teacher)


def test_checkpoint_steps_count_completed_policy_updates():
    assert not should_save_opd_checkpoint(completed_steps=29, total_steps=435, save_every=30)
    assert should_save_opd_checkpoint(completed_steps=30, total_steps=435, save_every=30)
    assert should_save_opd_checkpoint(completed_steps=435, total_steps=435, save_every=30)


def test_output_checkpoint_tag_cannot_overwrite_existing_directory(tmp_path):
    checkpoint_dir = tmp_path / "chatrl_checkpoints" / "existing-opd"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="existing-opd"):
        reserve_opd_checkpoint_dir(tmp_path, "existing-opd")


def test_output_checkpoint_tag_atomically_reserves_unused_directory(tmp_path):
    checkpoint_dir = reserve_opd_checkpoint_dir(tmp_path, "new-opd")

    assert checkpoint_dir == str(tmp_path / "chatrl_checkpoints" / "new-opd")
    assert (tmp_path / "chatrl_checkpoints" / "new-opd").is_dir()
