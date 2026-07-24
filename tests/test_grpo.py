import pytest
import torch

from nanochat.grpo import (
    compute_grpo_loss,
    compute_group_advantages,
    get_rollout_batch_plan,
    reserve_grpo_checkpoint_dir,
    should_save_grpo_checkpoint,
    validate_grpo_batch_config,
    validate_grpo_sampling_config,
)


def test_group_advantages_are_standardized():
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])

    advantages = compute_group_advantages(rewards)

    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(), torch.tensor(1.0), atol=1e-5)


def test_group_advantages_are_zero_when_rewards_are_equal():
    advantages = compute_group_advantages(torch.ones(16))

    assert torch.equal(advantages, torch.zeros(16))
    assert torch.isfinite(advantages).all()


def test_rollout_plan_preserves_nanochat_epoch_length():
    plan = get_rollout_batch_plan(
        dataset_size=7473,
        rollout_batch_size=32,
        policy_batch_size=16,
        num_epochs=1,
        num_iterations=-1,
    )

    assert plan == [2] * 233 + [1]
    assert sum(plan) == 467


def test_num_iterations_overrides_dataset_epoch_length():
    plan = get_rollout_batch_plan(
        dataset_size=7473,
        rollout_batch_size=32,
        policy_batch_size=16,
        num_epochs=1,
        num_iterations=3,
    )

    assert plan == [2, 1]


def test_positive_advantage_clips_large_ratio():
    current = torch.tensor([[0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = current.detach().clone()

    loss, metrics = compute_grpo_loss(
        current,
        old,
        ref,
        torch.tensor([1.0]),
        torch.ones_like(current),
        kl_beta=0.0,
    )

    assert torch.allclose(loss, torch.tensor(-1.2))
    assert torch.allclose(metrics["clip_fraction"], torch.tensor(1.0))


def test_negative_advantage_clips_small_ratio():
    current = torch.tensor([[-0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = current.detach().clone()

    loss, _ = compute_grpo_loss(
        current,
        old,
        ref,
        torch.tensor([-1.0]),
        torch.ones_like(current),
        kl_beta=0.0,
    )

    assert torch.allclose(loss, torch.tensor(0.8))


def test_reference_kl_is_zero_for_equal_log_probabilities():
    logp = torch.tensor([[-1.0, -2.0]], requires_grad=True)

    loss, metrics = compute_grpo_loss(
        logp,
        logp.detach(),
        logp.detach(),
        torch.tensor([0.0]),
        torch.ones_like(logp),
    )

    assert torch.equal(loss, torch.tensor(0.0))
    assert torch.equal(metrics["reference_kl"], torch.tensor(0.0))


def test_reference_kl_is_positive_for_different_log_probabilities():
    current = torch.zeros((1, 2), requires_grad=True)
    ref = torch.full_like(current, -1.0)

    loss, metrics = compute_grpo_loss(
        current,
        current.detach(),
        ref,
        torch.tensor([0.0]),
        torch.ones_like(current),
    )

    assert loss > 0
    assert metrics["reference_kl"] > 0


def test_sequence_normalization_weights_responses_equally():
    current = torch.zeros((2, 4), requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float)

    loss, _ = compute_grpo_loss(
        current,
        old,
        ref,
        torch.tensor([1.0, 1.0]),
        mask,
        kl_beta=0.0,
    )

    assert torch.allclose(loss, torch.tensor(-1.0))


def test_masked_tokens_do_not_affect_loss():
    current = torch.tensor([[0.0, 100.0]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    mask = torch.tensor([[1.0, 0.0]])

    loss, metrics = compute_grpo_loss(
        current,
        old,
        ref,
        torch.tensor([1.0]),
        mask,
        kl_beta=0.0,
    )

    assert torch.allclose(loss, torch.tensor(-1.0))
    assert torch.equal(metrics["clip_fraction"], torch.tensor(0.0))


def test_large_negative_policy_log_ratio_keeps_corrective_gradient():
    current = torch.tensor([[-25.0]], requires_grad=True)

    loss, _ = compute_grpo_loss(
        current,
        torch.zeros_like(current),
        current.detach(),
        torch.tensor([1.0]),
        torch.ones_like(current),
        kl_beta=0.0,
    )
    loss.backward()

    assert current.grad.item() < 0


def test_large_reference_log_ratio_keeps_kl_gradient():
    current = torch.tensor([[0.0]], requires_grad=True)

    loss, _ = compute_grpo_loss(
        current,
        current.detach(),
        torch.tensor([[25.0]]),
        torch.tensor([0.0]),
        torch.ones_like(current),
        kl_beta=1.0,
    )
    loss.backward()

    assert current.grad.item() < 0


def test_batch_config_requires_world_size_divisibility():
    with pytest.raises(ValueError, match="world size"):
        validate_grpo_batch_config(
            rollout_batch_size=24,
            policy_batch_size=12,
            num_samples=16,
            device_batch_size=8,
            world_size=8,
        )


def test_batch_config_requires_complete_sampling_microbatches():
    with pytest.raises(ValueError, match="num samples"):
        validate_grpo_batch_config(
            rollout_batch_size=32,
            policy_batch_size=16,
            num_samples=15,
            device_batch_size=8,
            world_size=8,
        )


def test_sampling_config_matches_recorded_policy_logprobs():
    validate_grpo_sampling_config(temperature=1.0, top_k=0)


def test_sampling_config_rejects_temperature_transform():
    with pytest.raises(ValueError, match="temperature"):
        validate_grpo_sampling_config(temperature=0.8, top_k=0)


def test_sampling_config_rejects_top_k_truncation():
    with pytest.raises(ValueError, match="top-k"):
        validate_grpo_sampling_config(temperature=1.0, top_k=50)


def test_checkpoint_steps_count_completed_policy_updates():
    assert not should_save_grpo_checkpoint(completed_steps=29, total_steps=467, save_every=30)
    assert should_save_grpo_checkpoint(completed_steps=30, total_steps=467, save_every=30)
    assert should_save_grpo_checkpoint(completed_steps=467, total_steps=467, save_every=30)


def test_output_checkpoint_tag_cannot_overwrite_existing_model(tmp_path):
    checkpoint_dir = tmp_path / "chatrl_checkpoints" / "existing-grpo"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_30.pt").touch()

    with pytest.raises(FileExistsError, match="existing-grpo"):
        reserve_grpo_checkpoint_dir(tmp_path, "existing-grpo")


def test_output_checkpoint_tag_cannot_reuse_empty_directory(tmp_path):
    checkpoint_dir = tmp_path / "chatrl_checkpoints" / "empty-grpo"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="empty-grpo"):
        reserve_grpo_checkpoint_dir(tmp_path, "empty-grpo")


def test_output_checkpoint_tag_cannot_reuse_metadata_only_directory(tmp_path):
    checkpoint_dir = tmp_path / "chatrl_checkpoints" / "partial-grpo"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "meta_30.json").touch()

    with pytest.raises(FileExistsError, match="partial-grpo"):
        reserve_grpo_checkpoint_dir(tmp_path, "partial-grpo")


def test_output_checkpoint_tag_atomically_reserves_unused_directory(tmp_path):
    checkpoint_dir = reserve_grpo_checkpoint_dir(tmp_path, "new-grpo")

    assert checkpoint_dir == str(tmp_path / "chatrl_checkpoints" / "new-grpo")
    assert (tmp_path / "chatrl_checkpoints" / "new-grpo").is_dir()
