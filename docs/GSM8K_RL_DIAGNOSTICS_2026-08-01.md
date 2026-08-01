# GSM8K RL Diagnostics (2026-08-01)

## Method

Four existing checkpoints were evaluated without training on the same first 400 examples of nanochat's shuffled GSM8K test split. Each question received 16 samples with temperature 1.0, top-k disabled, a 256-token completion limit, and deterministic per-question seeds derived from seed 1337.

The raw JSON and terminal logs are stored on the A800 server under `nanochat/diagnostics/`.

## Group Statistics

| Checkpoint | All wrong | Mixed | All correct | Mean reward | Pass@1 | Pass@8 | Pass@16 | Correct samples per solved question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D24 SFT step483 | 252 (63.00%) | 148 (37.00%) | 0 (0.00%) | 6.53% | 7.00% | 28.75% | 37.00% | 2.82 / 16 |
| D24 RL step466 | 220 (55.00%) | 172 (43.00%) | 8 (2.00%) | 16.16% | 14.75% | 36.00% | 45.00% | 5.74 / 16 |
| D32 SFT step242 | 260 (65.00%) | 140 (35.00%) | 0 (0.00%) | 6.34% | 6.25% | 25.75% | 35.00% | 2.90 / 16 |
| D32 GRPO step180 | 187 (46.75%) | 213 (53.25%) | 0 (0.00%) | 13.05% | 11.50% | 43.50% | 53.25% | 3.92 / 16 |

For binary group-relative rewards, only mixed groups provide a task-directed policy advantage. On this test sample, 63-65% of questions at the SFT starting points produced no correct response in 16 attempts. This does not measure the exact historical training batches, but it directly measures the checkpoints' reward-signal coverage under the same rollout policy.

## Solved-Set Comparisons

"Solved" means at least one correct response among 16 samples. The p-values below are two-sided exact binomial tests over discordant paired questions.

| Comparison | Intersection | Left only | Right only | Jaccard | Paired p-value |
|---|---:|---:|---:|---:|---:|
| D24 SFT vs D32 SFT | 92 | 56 | 48 | 0.469 | 0.493 |
| D24 RL vs D32 GRPO | 141 | 39 | 72 | 0.560 | 0.00224 |
| D24 SFT vs D24 RL | 100 | 48 | 80 | 0.439 | 0.00593 |
| D32 SFT vs D32 GRPO | 115 | 25 | 98 | 0.483 | 2.08e-11 |

At SFT, D32 did not expose a broader GSM8K solved set than D24 under this sampling protocol. After RL, D32 GRPO solved 213 questions at Pass@16 versus 180 for D24 RL, and its paired coverage advantage is unlikely to be sampling noise alone.

## Interpretation

The similar post-RL Pass@1 values do not mean D24 and D32 have the same GSM8K capability. Their probability distributions differ:

- D24 RL concentrates more probability on a narrower solved set. It averages 5.74 correct samples per solved question and reaches 14.75% Pass@1, but covers only 180 questions at Pass@16.
- D32 GRPO spreads lower correct probability across a broader solved set. It averages 3.92 correct samples per solved question and reaches 11.50% Pass@1, but covers 213 questions at Pass@16.
- D32 therefore has more latent correct trajectories after GRPO, but the current objective and one-epoch schedule have not concentrated enough of that probability into the first response.

The comparison also contains an algorithm confound: D24 step466 used nanochat's original centered policy-gradient RL, while D32 step180 used the new clipped GRPO objective with reference KL. The broader D32 coverage cannot be attributed to model size alone without running the same RL algorithm and schedule on both sizes.

## Conclusion

The evidence supports a reward-signal and probability-concentration bottleneck, not a shared model-capacity ceiling:

1. D24 and D32 begin RL with statistically indistinguishable Pass@16 coverage on this sample.
2. A majority of SFT questions are all-wrong groups and provide no outcome-reward learning direction.
3. D32 GRPO expands the solved set substantially more than D24 RL.
4. D32's remaining weakness is converting low-probability correct trajectories into reliable Pass@1 answers while preserving its broader coverage.
