# GRPO on the Offline 8xA800 Server

This runbook trains the D32 SFT checkpoint with clipped GRPO. It does not
replace `scripts/chat_rl.py`; GRPO uses the separate `scripts/chat_grpo.py`
entry point and a new output model tag.

## Environment

Run every command from the repository root in the existing `nanochat` Conda
environment:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanochat

cd /media/ubuntu/D1/zsj/nanochat/nanochat/nanochat

export NANOCHAT_BASE_DIR=/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat_cache
export HF_HOME=/media/ubuntu/D1/zsj/nanochat/nanochat/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TIKTOKEN_CACHE_DIR=/media/ubuntu/D1/zsj/nanochat/nanochat/tiktoken_cache

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
```

`OMP_NUM_THREADS=1` limits CPU threads per torchrun worker. It does not reduce
the GPU count: torchrun still starts eight workers, one for each A800.

## Preflight

Confirm that the shell is using the intended environment, all GPUs are
available, the input SFT checkpoint exists, and the output tag is unused:

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

test -f "$NANOCHAT_BASE_DIR/chatsft_checkpoints/d32-a800-bf16-L/model_000242.pt"
test ! -e "$NANOCHAT_BASE_DIR/chatrl_checkpoints/d32-a800-grpo-v1"

python -m pytest -q
python -m scripts.chat_grpo --help
```

The output tag is intentionally different from the existing REINFORCE
checkpoints. Rank 0 atomically reserves the output directory, and the GRPO
script refuses to start if that tag already exists, including an empty or
partially written directory.

## Train

Keep `temperature=1.0` and `top-k=0` for training. The stored old-policy log
probabilities use the model's full, untempered distribution, so changing these
values would invalidate the importance ratios and is rejected at startup.

```bash
set -o pipefail

torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- \
  --model-tag=d32-a800-bf16-L \
  --model-step=242 \
  --output-model-tag=d32-a800-grpo-v1 \
  --num-epochs=1 \
  --rollout-batch-size=32 \
  --policy-batch-size=16 \
  --device-batch-size=8 \
  --num-samples=16 \
  --ppo-epochs=1 \
  --temperature=1.0 \
  --top-k=0 \
  --clip-epsilon=0.2 \
  --kl-beta=0.04 \
  --init-lr-frac=0.005 \
  --eval-every=30 \
  --eval-examples=400 \
  --save-every=30 \
  --run=dummy \
  2>&1 | tee chat_grpo_d32_a800.log
```

This schedule consumes 7,472 of the 7,473 GSM8K training questions:

- 234 rollout batches;
- 467 optimizer updates;
- 32 questions per full rollout, split into two 16-question policy updates;
- 16 responses per question;
- 4 rollout questions and 2 update questions per GPU.

Step numbers are completed optimizer updates. The initial evaluation is step 0.
Evaluations and periodic checkpoints are produced at steps 30, 60, ..., 450.
The final checkpoint is step 467.

## Outputs

Terminal output is written to:

```text
/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat/chat_grpo_d32_a800.log
```

Model and metadata checkpoints are written to:

```text
/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat_cache/chatrl_checkpoints/d32-a800-grpo-v1/
```

Expected files include:

```text
model_000030.pt
meta_000030.json
...
model_000467.pt
meta_000467.json
```

The script does not save optimizer state, matching nanochat's existing RL
workflow. A stopped run cannot be resumed exactly from an optimizer checkpoint;
start a new output tag for a new run.

## Monitoring

Each full rollout produces two policy updates. On the first update, the policy
still matches the rollout snapshot, so `ratio` should be about 1,
`old KL` about 0, and `clip` about 0. The second update reuses the same snapshot,
so non-zero old-policy KL and clipping are expected.

Stop and inspect the log if any loss or KL becomes non-finite, reference KL
grows rapidly, clip fraction remains near 1, or reward and pass@k collapse
together. Do not judge the run from loss alone; compare reward, pass@k,
reference KL, old-policy KL, and clip fraction at the same checkpoint step.

## Evaluate

To reproduce the training-time GSM8K pass@8 settings for checkpoint 30:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
  -i rl \
  -g d32-a800-grpo-v1 \
  -s 30 \
  -a GSM8K \
  -t 1.0 \
  -n 8 \
  -k 50 \
  -m 256 \
  -x 400 \
  2>&1 | tee chat_eval_grpo_d32_step030_gsm8k_pass8.log
```

For the full deterministic nanochat evaluation of the final checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
  -i rl \
  -g d32-a800-grpo-v1 \
  -s 467 \
  2>&1 | tee chat_eval_grpo_d32_step467_full.log
```

Use the same commands with steps 30, 60, and later checkpoints to compare the
learning curve. Compare them against the SFT and low-LR REINFORCE logs using
identical evaluation arguments.

## Chat

Interactive chat uses one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.chat_cli \
  -i rl \
  -g d32-a800-grpo-v1 \
  -s 467
```

For a one-shot prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.chat_cli \
  -i rl \
  -g d32-a800-grpo-v1 \
  -s 467 \
  -p "Solve: If 12 notebooks cost 36 dollars, how much do 5 notebooks cost?"
```
