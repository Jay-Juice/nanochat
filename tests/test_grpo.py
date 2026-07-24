import torch

from nanochat.grpo import compute_group_advantages, get_rollout_batch_plan


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
