# Nanochat GRPO Design

## Goal

Add a complete, compact implementation of the original DeepSeekMath GRPO
objective to nanochat while preserving the existing `scripts/chat_rl.py` as a
REINFORCE baseline. The training layout should follow the common verl pattern:
collect a rollout batch, freeze its old-policy log probabilities, split it into
policy mini-batches, and update each mini-batch once by default.

## Scope

The first implementation trains on the existing GSM8K task and reward function.
It does not add a learned reward model, process rewards, iterative reward-model
training, vLLM, FSDP, or asynchronous rollout workers. Those systems are not
needed to implement or evaluate GRPO for nanochat's current scale.

The existing RL script, checkpoints, and evaluation interface remain intact.
GRPO checkpoints continue to use `chatrl_checkpoints`, under a required,
distinct output model tag, so existing `chat_eval.py` and `chat_cli.py` can load
them with `-i rl`.

## Terminology And Batch Layout

All example counts refer to GSM8K questions, not generated sequences.

- `rollout_batch_size=32`: questions sampled from one old-policy snapshot.
- `policy_batch_size=16`: questions contributing to one optimizer update.
- `num_samples=16`: generated responses per question.
- `device_batch_size=8`: responses processed by one forward/backward pass.
- `ppo_epochs=1`: each generated response is used for one policy pass.

On eight GPUs, a full rollout contains four questions per rank. Each policy
mini-batch contains two questions per rank and therefore preserves nanochat's
existing effective update batch of 16 global questions. The first policy
mini-batch starts at ratio one. The second uses responses sampled before the
first update, so the old-policy ratio and clipping constraint are active.

One GSM8K epoch still contains:

```text
floor(7473 / 16) = 467 policy updates
467 * 16 = 7472 consumed questions
7472 * 16 = 119552 generated responses
```

Policy updates are grouped into rollout batches two at a time. The final
rollout contains one policy mini-batch, avoiding duplicated questions and
preserving the original 467-step schedule.

## GRPO Objective

For each question, rewards from its 16 sampled responses are normalized using
the original group-relative outcome advantage:

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
```

The sample standard deviation is used to match verl. If all rewards in a group
are identical, every advantage is explicitly set to zero.

Before any update from a rollout batch, the trainer records:

- `old_logp`: sampled-token log probabilities from the rollout policy.
- `ref_logp`: sampled-token log probabilities from the frozen SFT reference.

For each sampled token:

```text
ratio = exp(current_logp - old_logp)
policy_objective = min(
    ratio * advantage,
    clip(ratio, 1 - epsilon, 1 + epsilon) * advantage,
)

log_ratio_ref = ref_logp - current_logp
reference_kl = exp(log_ratio_ref) - log_ratio_ref - 1

token_loss = -policy_objective + beta * reference_kl
```

`epsilon` defaults to `0.2`. `beta` defaults to `0.04`, matching the original
DeepSeekMath GRPO experiment. Log-ratios are clamped before exponentiation for
numerical stability.

The original sequence-level GRPO normalization is used: token losses are
averaged within each response, then averaged across responses. Forced tool
tokens, prompt tokens, and padding remain masked exactly as in `chat_rl.py`.

## Components

### `nanochat/grpo.py`

Contains small, framework-independent tensor functions:

- group-relative advantage normalization;
- clipped GRPO policy and reference-KL loss;
- scalar metrics for ratio, clip fraction, old-policy KL, and reference KL.

Keeping the mathematics outside the training script makes it directly
unit-testable on CPU and prevents the main script from accumulating framework
abstractions.

### `scripts/chat_grpo.py`

Follows the structure and naming style of `scripts/chat_rl.py`:

1. Initialize distributed compute and logging.
2. Load the trainable SFT policy.
3. Load a second SFT model as the frozen reference.
4. Generate a rollout batch before any optimizer update.
5. Compute rewards, group advantages, old log probabilities, and reference log
   probabilities.
6. Split the rollout into policy mini-batches.
7. Accumulate device-sized micro-batches and execute one optimizer update per
   policy mini-batch.
8. Evaluate, log, and checkpoint by policy-update step.

The rollout buffer is a short list of per-question tensors. It contains token
IDs, masks, rewards, advantages, old log probabilities, and reference log
probabilities. No vocabulary-sized logits are retained.

## Command-Line Interface

The script keeps nanochat's existing model, optimizer, generation, evaluation,
and logging arguments. It adds:

```text
--output-model-tag       required destination tag
--rollout-batch-size     default 32 questions
--policy-batch-size      default 16 questions
--ppo-epochs             default 1
--clip-epsilon           default 0.2
--kl-beta                default 0.04
```

The script validates:

- batch sizes are positive and divisible by the distributed world size;
- rollout batch size is divisible by policy batch size;
- number of samples is divisible by device batch size;
- the output tag does not already contain checkpoints;
- input and output tags are logged separately.

Optimizer defaults remain consistent with `chat_rl.py`. The D32 A800 command
will explicitly retain the previously stable low learning-rate fraction rather
than silently restoring the high-LR configuration that collapsed.

## Logging And Checkpoints

Terminal and optional W&B logging include:

- pass@k;
- reward mean and standard deviation;
- response length;
- advantage mean and standard deviation;
- policy loss;
- reference KL;
- old-policy approximate KL;
- clip fraction;
- learning-rate multiplier.

Checkpoint metadata records the model config, algorithm name, input SFT
checkpoint, and complete training configuration. Optimizer state remains
omitted, matching nanochat's existing short RL stage.

## Testing

`tests/test_grpo.py` covers:

- normalized advantages have group mean zero and sample standard deviation one;
- all-equal rewards produce zero advantages without NaN or infinity;
- positive and negative advantages select the correct clipped objective;
- reference KL is zero for equal policies and non-negative otherwise;
- sequence-level normalization gives equal response weight despite different
  response lengths;
- masked prompt, forced, and padding positions do not affect loss or metrics.

Local verification consists of the focused unit tests, the existing test suite,
Python compilation, and CLI help. A short remote CPU or single-GPU smoke run is
performed after deployment; the user starts the full eight-GPU training run.

## Version Control And Deployment

Implementation commits contain only GRPO source, tests, and documentation.
Existing local A800 edits, generated logs, data, wheel archives, caches, and
checkpoints are not staged implicitly.

The local Windows checkout remains authoritative. After verification, commits
are pushed to `Jay-Juice/nanochat`, then source files are synchronized to the
A800 target using the documented rsync workflow without `--delete`. The exact
remote project directory must be confirmed before synchronization.
