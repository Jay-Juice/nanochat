# D32 Math SFT and RL Experiment

This report records the D32 math-data SFT experiment and its follow-up RL run
on an offline 8x NVIDIA A800 80 GB server. The goal was to test whether the
similar D24 and D32 GSM8K results came from insufficient math supervision in
the original SFT mixture.

## Result

| Checkpoint | GSM8K | ARC-Easy | ARC-Challenge | MMLU | HumanEval | SpellingBee | ChatCORE |
|---|---:|---:|---:|---:|---:|---:|---:|
| D32 original SFT, step 242 | 9.63% | 72.14% | 57.85% | 40.17% | 12.20% | 100.00% | 0.4145 |
| D32 Math15 SFT, step 273 | 19.56% | 72.77% | 57.94% | 40.44% | 7.32% | 99.22% | 0.4238 |
| D32 Math15 RL, step 180 | **27.29%** | **72.81%** | **58.79%** | 40.34% | 5.49% | 99.22% | **0.4354** |

The final recommended math checkpoint is `d32-a800-bf16-L-math15`, RL step
180. The RL run continued to step 466, but its fixed-subset GSM8K curve peaked
at step 180 and declined afterwards. Use step 180 for this experiment's final
evaluation. HumanEval declined during math SFT and RL; this is the main measured
trade-off.

## What “15% Math” Means

The 15% target is a share of **assistant loss tokens**, not a share of all
rendered tokens and not the external-data-only share. “Math” counts the
original nanochat GSM8K examples plus the added Orca Math and MetaMathQA data.

| Quantity | Count |
|---|---:|
| Original mixture assistant loss tokens | 311,389,610 |
| Native GSM8K assistant loss tokens in original mixture | 3,028,396 (0.9725% of original assistant tokens) |
| Added Orca Math assistant loss tokens | 49,304,434 |
| Added MetaMathQA assistant loss tokens | 2,084,345 |
| Added external math assistant loss tokens | 51,388,779 |
| Final mixture assistant loss tokens | 362,778,389 |
| Final math assistant loss tokens, including native GSM8K | 54,417,175 (15.0001% of final assistant tokens) |

The external-data builder targeted the total math fraction with:

```text
external_tokens = ceil((target_fraction * original_total - original_math_tokens)
                       / (1 - target_fraction))
```

The actual selected token count can exceed the requested budget slightly
because whole examples are selected. That produced 15.0001% in this run.

The added JSONL contained 190,278 examples: 177,084 Orca Math and 13,194
MetaMathQA. The final training mixture contained 1,262,037 examples, so the
added rows were 15.0771% of rows. Rendered tokens and assistant loss tokens are
different measures: the external JSONL had 63,774,477 rendered tokens but
51,388,779 assistant loss tokens. The GSM8K evaluation split was checked for
normalized exact question overlap; none was found. This check does not rule out
semantic near-duplicates.

The training JSONL SHA256 was:

```text
04e0b201a9b0aeb87ac4642d4f55ae907849f29cf046f35ec8307471c9fa677a
```

The dataset itself is not stored in this repository. It was kept under
`$NANOCHAT_BASE_DIR/math_sft_v1/training_mix/math_sft_train.jsonl` on the A800
server.

## Training Recipe

The base D32 pretraining checkpoint was step 6400. Original SFT produced step
242, and the math SFT initialized from that checkpoint and saved to a separate
model tag at step 273. The math dataset is opt-in through `--math-sft-data`;
without it, the original SFT mixture is unchanged.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /media/ubuntu/D1/zsj/nanochat/nanochat/nanochat

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat_cache
export HF_HOME=/media/ubuntu/D1/zsj/nanochat/nanochat/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
  --model-tag=d32-a800-bf16-L \
  --model-step=6400 \
  --output-model-tag=d32-a800-bf16-L-math15 \
  --device-batch-size=8 \
  --math-sft-data=$NANOCHAT_BASE_DIR/math_sft_v1/training_mix/math_sft_train.jsonl \
  --run=dummy \
  2>&1 | tee chat_sft_d32_math15.log
```

Math SFT completed 273 optimization steps in 78.85 minutes. The minimum
validation BPB was 0.2467 and peak memory was about 61,091 MiB. The resulting
checkpoint is:

```text
$NANOCHAT_BASE_DIR/chatsft_checkpoints/d32-a800-bf16-L-math15/model_000273.pt
```

RL used nanochat's original `scripts/chat_rl.py` objective with a lower
learning-rate fraction. It was not the separate clipped GRPO implementation.
Each question generated 16 responses; there were 467 planned updates. The
lower `--init-lr-frac=0.005` reduced the learning rate by 10x relative to the
script default, but did not prevent later regression.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- \
  --model-tag=d32-a800-bf16-L-math15 \
  --model-step=273 \
  --num-epochs=1 \
  --device-batch-size=8 \
  --examples-per-step=16 \
  --num-samples=16 \
  --max-new-tokens=256 \
  --temperature=1.0 \
  --top-k=50 \
  --init-lr-frac=0.005 \
  --eval-every=30 \
  --eval-examples=400 \
  --save-every=30 \
  --run=dummy \
  2>&1 | tee chat_rl_d32_math15_lr0005_original.log
```

The fixed 400-question training evaluation peaked at Pass@1 21.75% at step 180
and Pass@8 57.25% at step 150. Full-test greedy GSM8K was 26.46% at step 150
and 27.29% at step 180. Step 180 was selected; the small 0.83-point difference
between those full-test results should not be overinterpreted. Full GSM8K
Pass@8 was not run for this final checkpoint.

## Reproduction Logs

Selected raw logs are included in [`experiment-logs/d32-math15`](experiment-logs/d32-math15/):

- math SFT training and evaluation;
- original RL training;
- step 150 and step 180 GSM8K evaluations;
- step 180 full evaluation.

They are terminal records, not additional source of truth; the summary tables
above report the metric counts and denominators for easier comparison.
