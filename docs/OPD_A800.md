# Native OPD on A800

This run distills the D32 Math15 RL teacher into the best fully evaluated D24
RL checkpoint using standard On-Policy Distillation on GSM8K.

## Checkpoints

Student:

```text
chatrl_checkpoints/d24-a800-bf16-L/model_000466.pt
GSM8K greedy accuracy: 16.98%
```

Teacher:

```text
chatrl_checkpoints/d32-a800-bf16-L-math15/model_000180.pt
GSM8K greedy accuracy: 27.29%
SHA256: 2b3e6ddcf69c1bb1e08b2eb304fff0ee189d23f9133b551a39484538384d7d49
```

The teacher and student share the nanochat tokenizer, 32,768-token vocabulary,
2,048-token context window, and chat format. Their depth and width may differ.

## Preflight

Run from the deployed source directory:

```bash
cd /media/ubuntu/D1/zsj/nanochat/nanochat/nanochat
source ~/anaconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=~/.conda/envs:~/anaconda3/envs
conda activate nanochat
export NANOCHAT_BASE_DIR=/media/ubuntu/D1/zsj/nanochat/nanochat/nanochat_cache
export OMP_NUM_THREADS=1
```

Verify the environment, checkpoints, checksum, output tag, and free space:

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
test -f "$NANOCHAT_BASE_DIR/chatrl_checkpoints/d24-a800-bf16-L/model_000466.pt"
test -f "$NANOCHAT_BASE_DIR/chatrl_checkpoints/d32-a800-bf16-L-math15/model_000180.pt"
sha256sum "$NANOCHAT_BASE_DIR/chatrl_checkpoints/d32-a800-bf16-L-math15/model_000180.pt"
test ! -e "$NANOCHAT_BASE_DIR/chatrl_checkpoints/d24-a800-opd-d32math15-v2-b64-constant"
df -h "$NANOCHAT_BASE_DIR"
```

Free at least 300 GB before the full run. The v2 schedule saves six D24
checkpoints, and the filesystem must also accommodate logs and temporary
training state.

## One-Step Smoke Run

The smoke run performs one real optimizer update and saves one D24 checkpoint.
The user starts this command after reviewing the output tag:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_opd -- \
  --student-model-tag=d24-a800-bf16-L \
  --student-model-step=466 \
  --teacher-model-tag=d32-a800-bf16-L-math15 \
  --teacher-model-step=180 \
  --output-model-tag=d24-a800-opd-d32math15-smoke \
  --num-iterations=1 \
  --rollout-batch-size=16 \
  --policy-batch-size=16 \
  --ppo-epochs=1 \
  --temperature=1.0 \
  --top-k=0 \
  --init-lr-frac=0.005 \
  --lr-schedule=constant \
  --eval-every=-1 \
  --save-every=-1 \
  --run=dummy \
  2>&1 | tee chat_opd_d24_d32math15_smoke.log
```

Expected structural output includes:

```text
Training prompts: 6973
Validation prompts: 500
Calculated policy updates: 1
Step 1/1
Saved model checkpoint
```

Stop before the full run if any rank reports OOM, a non-finite loss, mismatched
vocabularies/context windows, or a missing checkpoint.

## V1 Full Run

This consumes 6,960 of the 6,973 non-holdout prompts. The rollout and policy
batches are both 16, so every student rollout is used by the immediately
following update. One epoch contains 435 optimizer updates. Each prompt has
one student rollout; there is no GRPO response group.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_opd -- \
  --student-model-tag=d24-a800-bf16-L \
  --student-model-step=466 \
  --teacher-model-tag=d32-a800-bf16-L-math15 \
  --teacher-model-step=180 \
  --output-model-tag=d24-a800-opd-d32math15-v1 \
  --num-epochs=1 \
  --rollout-batch-size=16 \
  --policy-batch-size=16 \
  --ppo-epochs=1 \
  --max-new-tokens=256 \
  --temperature=1.0 \
  --top-k=0 \
  --clip-epsilon=0.2 \
  --init-lr-frac=0.005 \
  --lr-schedule=linear \
  --validation-examples=500 \
  --eval-examples=400 \
  --eval-samples=8 \
  --eval-every=30 \
  --save-every=30 \
  --run=dummy \
  2>&1 | tee chat_opd_d24_d32math15_v1.log
```

Checkpoints are expected at steps 30, 60, ..., 420, and 435. The output tag is
reserved before model loading and cannot overwrite an existing directory.

## GSM8K Control V2

This controlled run keeps the v1 student, teacher, data, sampling, and OPD
objective unchanged. It increases the global prompt batch from 16 to 64, keeps
the initial learning rates constant, and evaluates every checkpoint with the
same per-example sampling seeds.

The 6,973 non-holdout prompts produce 108 complete updates of 64 prompts. This
uses 6,912 prompts and skips the final incomplete group of 61 prompts. The
batch size is global across all eight GPUs, so each rank contributes eight
prompts per optimizer update.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_opd -- \
  --student-model-tag=d24-a800-bf16-L \
  --student-model-step=466 \
  --teacher-model-tag=d32-a800-bf16-L-math15 \
  --teacher-model-step=180 \
  --output-model-tag=d24-a800-opd-d32math15-v2-b64-constant \
  --num-epochs=1 \
  --rollout-batch-size=64 \
  --policy-batch-size=64 \
  --ppo-epochs=1 \
  --max-new-tokens=256 \
  --temperature=1.0 \
  --top-k=0 \
  --clip-epsilon=0.2 \
  --init-lr-frac=0.005 \
  --lr-schedule=constant \
  --validation-examples=500 \
  --eval-examples=400 \
  --eval-samples=8 \
  --eval-seed=1337 \
  --eval-every=20 \
  --save-every=20 \
  --run=dummy \
  2>&1 | tee chat_opd_d24_d32math15_v2_b64_constant.log
```

Holdout evaluations run at steps 0, 20, 40, 60, 80, and 100 using identical
seeds for every checkpoint. Checkpoints are saved at steps 20, 40, 60, 80,
100, and the final step 108. The constant schedule reports `lrm: 1.0` for all
updates.

## Reading The Metrics

- `teacher KL`: non-negative reverse-KL estimate; it should stay finite and
  generally fall as the student approaches the teacher.
- `adv`: `log p_teacher - log p_old_student`; its expectation under student
  rollouts is normally non-positive. A negative mean is not itself a failure.
- `old KL`: change from the rollout student to the updated student.
- `clip`: fraction of sampled tokens outside the PPO clipping interval.
- `ratio`: current/rollout student probability ratio; it should remain near 1.
- `reward`: GSM8K correctness of sampled training responses. It is diagnostic
  only and is not used in the OPD objective.
- `holdout Pass@k`: fixed GSM8K-train holdout used to select candidate steps.

Stop and inspect if metrics become non-finite, teacher KL rises continuously,
clip fraction remains high, completion length collapses, or holdout Pass@1 and
Pass@8 decline together across several evaluations.

## Final Evaluation

Do not use the official GSM8K test set to select every intermediate checkpoint.
Select two or three candidates from holdout Pass@1/Pass@8, then evaluate them on
the full 1,319-example test set.

Greedy GSM8K Pass@1 for a selected step:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
  -i rl \
  -g d24-a800-opd-d32math15-v1 \
  -s 180 \
  -a GSM8K \
  -t 0.0 \
  -n 1 \
  -k 50 \
  -m 256 \
  2>&1 | tee chat_eval_opd_d24_step180_gsm8k_greedy.log
```

Full GSM8K Pass@8 with the same untruncated sampling distribution as OPD:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
  -i rl \
  -g d24-a800-opd-d32math15-v1 \
  -s 180 \
  -a GSM8K \
  -t 1.0 \
  -n 8 \
  -k 0 \
  -m 256 \
  2>&1 | tee chat_eval_opd_d24_step180_gsm8k_pass8.log
```

Replace step 180 with each selected checkpoint. After choosing the GSM8K
candidate, run the normal complete evaluation to detect regressions:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
  -i rl \
  -g d24-a800-opd-d32math15-v1 \
  -s 180 \
  2>&1 | tee chat_eval_opd_d24_step180_full.log
```

Compare against D24 RL step 466 on GSM8K, ARC-Easy, ARC-Challenge, MMLU,
HumanEval, and SpellingBee. The OPD checkpoint is only preferable if the math
gain is not accompanied by unacceptable additional forgetting.
