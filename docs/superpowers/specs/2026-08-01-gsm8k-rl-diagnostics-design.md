# GSM8K RL Diagnostics Design

## Goal

Measure why D24 and D32 reach similar GSM8K RL accuracy without retraining. The diagnostic must quantify how often each checkpoint produces all-wrong, mixed, and all-correct response groups on the same questions, then compare which questions each model can solve at least once.

## Scope

Evaluate four existing checkpoints on the first 400 examples of the shuffled GSM8K test split:

- D24 SFT step 483
- D24 RL step 466
- D32 SFT step 242
- D32 GRPO step 180

Each question receives 16 stochastic samples with temperature 1.0, top-k disabled, and at most 256 completion tokens. This is inference only; no optimizer or checkpoint writes are involved.

## Interface

Add `scripts/chat_rl_diagnostics.py` with two modes:

1. Evaluation mode loads one `sft` or `rl` checkpoint under `torchrun`, samples GSM8K responses, gathers records on rank 0, and writes one JSON result.
2. Comparison mode reads result JSON files without CUDA and reports pairwise solved-set overlap for matching question indices.

The JSON stores configuration, aggregate counts, Pass@1/8/16, and one record per question containing its index and 16 binary rewards. It does not store completion text, keeping output small and avoiding unnecessary training-data duplication.

## Statistics

For an `N x K` binary reward matrix:

- all-wrong: row sum equals 0
- mixed: row sum is between 1 and K-1
- all-correct: row sum equals K
- Pass@k: fraction of rows with any success among the first k samples
- mean reward: total successes divided by `N * K`
- solved set: question indices with at least one success among all K samples

Pairwise comparison reports intersection, union, Jaccard similarity, left-only, and right-only solved counts. Inputs must contain identical question-index sets and sample counts.

## Reproducibility And Errors

Sampling seeds derive from a user seed and question index, so all checkpoints see the same per-question random seed. The script validates positive sample counts, binary rewards, matching records, and refuses to overwrite an existing output unless explicitly requested. Distributed failures do not leave a misleading completed summary.

## Testing

Pure aggregation and comparison functions live in `nanochat/grpo.py` and receive unit tests for normal groups, edge cases, Pass@k, and mismatched comparison inputs. The GPU script is additionally checked through CLI help and Python compilation before remote execution.
