# Nanochat GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an original-objective GRPO trainer for GSM8K with verl-style rollout and policy mini-batches, without changing nanochat's existing RL baseline.

**Architecture:** Pure GRPO tensor math and rollout scheduling live in `nanochat/grpo.py` and are tested independently. `scripts/chat_grpo.py` keeps nanochat's single-file training style, loads a trainable SFT policy plus frozen SFT reference, collects each rollout under one policy snapshot, and performs clipped policy mini-batch updates. GRPO checkpoints remain compatible with the existing `rl` evaluation and chat interfaces.

**Tech Stack:** Python 3.10, PyTorch 2.6, torch.distributed, pytest, nanochat Engine and checkpoint manager.

**Remote Environment:** Every A800 Python command first runs
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate nanochat`.
System Python is never used.

---

### Task 1: Group Advantages And Rollout Schedule

**Files:**
- Create: `nanochat/grpo.py`
- Create: `tests/test_grpo.py`

- [ ] **Step 1: Write failing tests for group normalization and rollout planning**

```python
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
```

- [ ] **Step 2: Sync the test to A800 and verify RED**

Run from PowerShell:

```powershell
wsl rsync -avz `
  -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe" `
  "/mnt/d/ai project/nanochat/.worktrees/nanochat-grpo/tests/test_grpo.py" `
  "A800:/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/tests/test_grpo.py"

ssh A800 "source ~/anaconda3/etc/profile.d/conda.sh && conda activate nanochat && cd '/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat' && python -m pytest tests/test_grpo.py -q"
```

Expected: collection fails because `nanochat.grpo` does not exist.

- [ ] **Step 3: Implement the minimal helpers**

```python
import math

import torch


def compute_group_advantages(rewards, eps=1e-6):
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
```

- [ ] **Step 4: Sync implementation and verify GREEN**

Run the focused A800 test again. Expected: `4 passed`.

- [ ] **Step 5: Commit**

```powershell
git add nanochat/grpo.py tests/test_grpo.py
git commit -m "test: define GRPO advantages and rollout schedule"
git push
```

### Task 2: Clipped GRPO Loss And Metrics

**Files:**
- Modify: `nanochat/grpo.py`
- Modify: `tests/test_grpo.py`

- [ ] **Step 1: Add failing tests for clipping, KL, sequence weighting, and masks**

```python
from nanochat.grpo import compute_grpo_loss


def test_positive_advantage_clips_large_ratio():
    current = torch.tensor([[0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = current.detach().clone()
    loss, metrics = compute_grpo_loss(
        current, old, ref, torch.tensor([1.0]), torch.ones_like(current), kl_beta=0.0
    )
    assert torch.allclose(loss, torch.tensor(-1.2))
    assert torch.allclose(metrics["clip_fraction"], torch.tensor(1.0))


def test_negative_advantage_clips_small_ratio():
    current = torch.tensor([[-0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = current.detach().clone()
    loss, _ = compute_grpo_loss(
        current, old, ref, torch.tensor([-1.0]), torch.ones_like(current), kl_beta=0.0
    )
    assert torch.allclose(loss, torch.tensor(0.8))


def test_reference_kl_is_zero_for_equal_log_probabilities():
    logp = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    loss, metrics = compute_grpo_loss(
        logp, logp.detach(), logp.detach(), torch.tensor([0.0]), torch.ones_like(logp)
    )
    assert torch.equal(loss, torch.tensor(0.0))
    assert torch.equal(metrics["reference_kl"], torch.tensor(0.0))


def test_sequence_normalization_weights_responses_equally():
    current = torch.zeros((2, 4), requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float)
    loss, _ = compute_grpo_loss(
        current, old, ref, torch.tensor([1.0, 1.0]), mask, kl_beta=0.0
    )
    assert torch.allclose(loss, torch.tensor(-1.0))


def test_masked_tokens_do_not_affect_loss():
    current = torch.tensor([[0.0, 100.0]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    mask = torch.tensor([[1.0, 0.0]])
    loss, metrics = compute_grpo_loss(
        current, old, ref, torch.tensor([1.0]), mask, kl_beta=0.0
    )
    assert torch.allclose(loss, torch.tensor(-1.0))
    assert torch.equal(metrics["clip_fraction"], torch.tensor(0.0))
```

- [ ] **Step 2: Verify RED on A800**

Expected: import fails because `compute_grpo_loss` is not defined.

- [ ] **Step 3: Implement the original sequence-normalized GRPO objective**

```python
def _masked_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_grpo_loss(
    current_logp,
    old_logp,
    ref_logp,
    advantages,
    response_mask,
    clip_epsilon=0.2,
    kl_beta=0.04,
):
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
```

- [ ] **Step 4: Verify focused and full tests**

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanochat
python -m pytest tests/test_grpo.py -q
python -m pytest -q
```

Expected: focused tests pass; full baseline remains `13 passed, 10 skipped` plus the new GRPO tests.

- [ ] **Step 5: Commit**

```powershell
git add nanochat/grpo.py tests/test_grpo.py
git commit -m "feat: add clipped GRPO objective"
git push
```

### Task 3: GRPO Training Script

**Files:**
- Create: `scripts/chat_grpo.py`
- Modify: `tests/test_grpo.py`

- [ ] **Step 1: Add failing validation tests**

```python
import pytest

from nanochat.grpo import validate_grpo_batch_config


def test_batch_config_requires_world_size_divisibility():
    with pytest.raises(ValueError, match="world size"):
        validate_grpo_batch_config(32, 15, 16, 8, 8)


def test_batch_config_requires_complete_sampling_microbatches():
    with pytest.raises(ValueError, match="num samples"):
        validate_grpo_batch_config(32, 16, 15, 8, 8)
```

- [ ] **Step 2: Verify RED, then implement validation**

```python
def validate_grpo_batch_config(
    rollout_batch_size,
    policy_batch_size,
    num_samples,
    device_batch_size,
    world_size,
):
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
```

- [ ] **Step 3: Create `scripts/chat_grpo.py` using nanochat's existing style**

The script will:

```python
model, tokenizer, meta = load_model(
    "sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
reference_model, _, _ = load_model(
    "sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
reference_model.requires_grad_(False)
reference_model.eval()

rollout_plan = get_rollout_batch_plan(
    len(train_task),
    args.rollout_batch_size,
    args.policy_batch_size,
    args.num_epochs,
    args.num_iterations,
)
```

For every rollout group, it generates responses and stores masked `old_logp`
and `ref_logp` before updates. Each policy mini-batch accumulates all
device-sized backward passes, scales each loss by the number of micro-batches,
then calls one `optimizer.step()`. Evaluation, LR scheduling, W&B metrics, and
checkpoint step numbers are keyed to policy updates.

Checkpoint initialization must reject an existing output tag:

```python
checkpoint_dir = os.path.join(get_base_dir(), "chatrl_checkpoints", args.output_model_tag)
if glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    raise FileExistsError(f"Output model tag already contains checkpoints: {args.output_model_tag}")
```

- [ ] **Step 4: Sync source and run static/CLI verification on A800**

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanochat
python -m py_compile nanochat/grpo.py scripts/chat_grpo.py
python -m scripts.chat_grpo --help
```

Expected: compilation succeeds and help lists rollout, policy, clipping, KL,
input checkpoint, and output checkpoint arguments without loading a model.

- [ ] **Step 5: Run focused and full tests**

Expected: all tests pass with no new warnings.

- [ ] **Step 6: Commit**

```powershell
git add nanochat/grpo.py scripts/chat_grpo.py tests/test_grpo.py
git commit -m "feat: train nanochat with GRPO"
git push
```

### Task 4: A800 Runbook And Final Verification

**Files:**
- Create: `docs/GRPO_A800.md`

- [ ] **Step 1: Document the D32 training and evaluation commands**

The runbook records a unique output tag, low D32 learning-rate fraction,
terminal logging, checkpoint locations, metric interpretation, and the exact
post-training `chat_eval.py` and `chat_cli.py` commands.

- [ ] **Step 2: Run local repository checks**

```powershell
git diff --check
git status --short
python -m compileall -q nanochat scripts tests
```

Local pytest is not used because the Windows Python environment has no
PyTorch. Compilation and all PyTorch tests run with the A800 Python.

- [ ] **Step 3: Preview and deploy only implementation files**

```powershell
wsl rsync -anvz `
  -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe" `
  "/mnt/d/ai project/nanochat/.worktrees/nanochat-grpo/nanochat/grpo.py" `
  "A800:/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/nanochat/grpo.py"

wsl rsync -anvz `
  -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe" `
  "/mnt/d/ai project/nanochat/.worktrees/nanochat-grpo/scripts/chat_grpo.py" `
  "A800:/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/scripts/chat_grpo.py"

wsl rsync -anvz `
  -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe" `
  "/mnt/d/ai project/nanochat/.worktrees/nanochat-grpo/tests/test_grpo.py" `
  "A800:/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/tests/test_grpo.py"

wsl rsync -anvz `
  -e "/mnt/c/Windows/System32/OpenSSH/ssh.exe" `
  "/mnt/d/ai project/nanochat/.worktrees/nanochat-grpo/docs/GRPO_A800.md" `
  "A800:/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/docs/GRPO_A800.md"
```

After reviewing the dry run, deploy with `-avz` and explicit destination
subdirectories so files retain their repository paths. Do not use
`--delete`.

- [ ] **Step 4: Run final A800 verification**

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /media/ubuntu/D1/zsj/nanochat/nanochat/nanochat
python -m pytest -q
python -m py_compile nanochat/grpo.py scripts/chat_grpo.py
python -m scripts.chat_grpo --help
```

- [ ] **Step 5: Commit and push documentation**

```powershell
git add docs/GRPO_A800.md
git commit -m "docs: add A800 GRPO runbook"
git push
```

- [ ] **Step 6: Hand off the full training command**

The user starts the eight-GPU job. Codex does not launch the full training
process.
