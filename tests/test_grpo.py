import torch

from nanochat.grpo import compute_grpo_loss, compute_group_advantages, get_rollout_batch_plan


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
