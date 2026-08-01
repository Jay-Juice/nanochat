# GSM8K RL Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible inference-only diagnostic that measures GSM8K reward-group composition and solved-question overlap for existing SFT and RL checkpoints.

**Architecture:** Pure aggregation and comparison functions live in `nanochat/grpo.py`. A self-contained distributed CLI in `scripts/chat_rl_diagnostics.py` loads one checkpoint, samples strided GSM8K examples on each rank, gathers compact reward records to rank 0, and writes JSON; the same CLI compares two saved JSON files without initializing CUDA.

**Tech Stack:** Python, PyTorch distributed, nanochat Engine, pytest, JSON.

---

### Task 1: Reward-group summaries

**Files:**
- Modify: `nanochat/grpo.py`
- Modify: `tests/test_grpo.py`

- [ ] Add failing tests asserting all-wrong/mixed/all-correct counts, fractions, mean reward, Pass@1/8/16, and rejection of empty, ragged, or non-binary groups.
- [ ] Run `python -m pytest tests/test_grpo.py -q` and confirm the new import or function assertions fail.
- [ ] Implement `summarize_reward_groups(reward_groups)` using pure Python data so JSON records need no tensor conversion.
- [ ] Re-run `python -m pytest tests/test_grpo.py -q` and confirm all GRPO tests pass.

### Task 2: Solved-set comparisons

**Files:**
- Modify: `nanochat/grpo.py`
- Modify: `tests/test_grpo.py`

- [ ] Add failing tests for intersection, union, Jaccard, left-only/right-only counts, mismatched index sets, duplicate indices, and mismatched sample counts.
- [ ] Run the focused tests and verify failure for the missing comparison function.
- [ ] Implement `compare_reward_records(left_records, right_records)` with explicit record validation.
- [ ] Re-run the focused and full GRPO test file.

### Task 3: Distributed diagnostic CLI

**Files:**
- Create: `scripts/chat_rl_diagnostics.py`

- [ ] Add an `evaluate` subcommand accepting source, model tag, step, output, max examples, samples, device batch size, generation settings, and seed.
- [ ] Generate each question in device-sized passes with distinct deterministic seeds, retain only `{idx, rewards}`, gather records with `dist.gather_object`, sort on rank 0, and write configuration plus `summarize_reward_groups` output atomically.
- [ ] Add a `compare` subcommand that reads two outputs, calls `compare_reward_records`, and prints JSON without initializing distributed compute.
- [ ] Validate divisibility, positive limits, required model arguments, and output overwrite behavior.
- [ ] Run `python -m scripts.chat_rl_diagnostics --help`, subcommand help, and `python -m py_compile scripts/chat_rl_diagnostics.py`.

### Task 4: Verification and publication

**Files:**
- Verify all files above.

- [ ] Run `python -m pytest tests/test_grpo.py -q` and the repository's available focused tests.
- [ ] Run `git diff --check` and inspect the scoped diff without altering unrelated working-tree changes.
- [ ] Commit only the diagnostic implementation and push `master` to `origin`.
- [ ] Synchronize the committed files to the A800 checkout.

### Task 5: A800 inference and analysis

**Files:**
- Generate remotely: four JSON result files and terminal logs under the repository directory.

- [ ] Activate conda environment `nanochat` and evaluate D24 SFT step 483 on 400 questions with 16 samples.
- [ ] Evaluate D24 RL step 466 with identical settings.
- [ ] Evaluate D32 SFT step 242 with identical settings.
- [ ] Evaluate D32 GRPO tag `d32-a800-grpo-v1` step 180 with identical settings.
- [ ] Compare D24 versus D32 at SFT and RL stages, plus each model's SFT-to-RL transition.
- [ ] Report all group counts/fractions, Pass@1/8/16, solved-set overlap, runtime failures, and the resulting diagnosis.
