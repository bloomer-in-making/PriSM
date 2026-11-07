#!/usr/bin/env bash
# PriSM — precompute + train + eval for one of the six generation tasks
# covered by this release. Single GPU unless noted below.
#
# Usage:
#   bash scripts/run_task.sh LaMP_4
#   bash scripts/run_task.sh LongLaMP_topic_writing_user
#
# Required env vars (raw LaMP / LongLaMP JSONs — see README "Data"). Only
# needed when cache/<task> does not exist yet:
#   TRAIN_QUESTIONS  path to train_questions.json
#   TRAIN_OUTPUTS    path to train_outputs.json
#   DEV_QUESTIONS    path to dev_questions.json
#   DEV_OUTPUTS      path to dev_outputs.json
#
# Optional env vars:
#   GPU_IDS=0                   CUDA device id(s)
#   BASE_MODEL=Qwen/Qwen3-8B    base LM
#   CACHE_DIR=cache/<task>      cache directory
#   OUT_ROOT=runs               checkpoint root
#   SEED                        master seed (per-task default below)
#   CLUSTER_SEED                k-means seed (per-task default below)
#   FORCE_PRECOMPUTE=0          1 = rebuild the cache even if it exists
#   FORCE_SWEEP=0               1 = choose the cluster seed by silhouette +
#                               balance sweep instead of the paper value
#   SWEEP_LOW=0 SWEEP_HIGH=30   sweep range (only with FORCE_SWEEP=1)
#   BALANCE_THRESHOLD=2.0       sweep filter (max/min cluster size ratio)
#   GPU0_LAYERS=-1              >=0 splits the model over two GPUs with this
#                               many decoder layers on GPU 0. The paper's
#                               LongLaMP topic-writing run used 4 (two GPUs);
#                               it is a memory accommodation and does not
#                               change the computation.
#   BUFFER_SEEDING=deterministic
#                               evaluation-buffer policy, passed to both steps.
#                               "random" redraws the buffers every run, so the
#                               numbers stop being reproducible; use it only to
#                               measure how much the draw itself moves a metric.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:?usage: bash scripts/run_task.sh <task_name>}"

# ── Per-task settings used for the numbers reported in the paper ───────────
#
#  SEED_D         master seed (rep-div buffers, parameter init, data order)
#  CSEED_D        k-means cluster seed. Obtained by a silhouette + balance
#                 (max/min <= 2.0) sweep over seeds 0..30 of the user-level
#                 k-means (K=4) on the TRAIN split only — dev users are never
#                 used to pick it. Rerun it with FORCE_SWEEP=1.
#  IN_LEN         --max_input_len
#  TRAIN_MAX      --max_target_tokens during training
#  EVAL_MAX       --max_target_tokens during evaluation. For the three
#                 LongLaMP tasks this is deliberately smaller than TRAIN_MAX:
#                 long-form generation otherwise runs into repetition loops.
#                 --max_target_tokens caps the training target, the generation
#                 length at evaluation, AND the prompt budget reserved for the
#                 answer, so the two values can only be set by running the two
#                 steps as separate processes. Do not "fix" them to match.
#  ACC            --grad_accum
#  NGRAM          --eval_no_repeat_ngram_size
#
# | task                              | domain                   |
# |-----------------------------------|--------------------------|
# | LaMP_4                            | news headline generation |
# | LaMP_5                            | paper title generation   |
# | LaMP_7                            | tweet paraphrasing       |
# | LongLaMP_abstract_generation_user | paper abstract generation|
# | LongLaMP_topic_writing_user       | reddit topic writing     |
# | LongLaMP_product_review_user      | product review generation|
case "${TASK}" in
  LaMP_4)
    SEED_D=42; CSEED_D=9;  IN_LEN=8192; TRAIN_MAX=24;   EVAL_MAX=24;  ACC=8; NGRAM=0 ;;
  LaMP_5)
    SEED_D=42; CSEED_D=15; IN_LEN=8192; TRAIN_MAX=24;   EVAL_MAX=24;  ACC=8; NGRAM=0 ;;
  LaMP_7)
    SEED_D=42; CSEED_D=22; IN_LEN=8192; TRAIN_MAX=64;   EVAL_MAX=64;  ACC=8; NGRAM=0 ;;
  LongLaMP_abstract_generation_user)
    SEED_D=42; CSEED_D=42; IN_LEN=4096; TRAIN_MAX=512;  EVAL_MAX=300; ACC=4; NGRAM=0 ;;
  LongLaMP_topic_writing_user)
    SEED_D=6;  CSEED_D=6;  IN_LEN=4096; TRAIN_MAX=1024; EVAL_MAX=720; ACC=4; NGRAM=4 ;;
  LongLaMP_product_review_user)
    SEED_D=0;  CSEED_D=0;  IN_LEN=4096; TRAIN_MAX=1024; EVAL_MAX=720; ACC=4; NGRAM=4 ;;
  *)
    echo "[ERROR] unknown task: ${TASK}"
    echo "        valid: LaMP_4, LaMP_5, LaMP_7,"
    echo "               LongLaMP_abstract_generation_user,"
    echo "               LongLaMP_topic_writing_user,"
    echo "               LongLaMP_product_review_user"
    exit 1 ;;
esac

mkdir -p logs runs

GPU_IDS="${GPU_IDS:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
CACHE_DIR="${CACHE_DIR:-cache/${TASK}}"
OUT_ROOT="${OUT_ROOT:-runs}"
SEED="${SEED:-${SEED_D}}"
CLUSTER_SEED="${CLUSTER_SEED:-${CSEED_D}}"
EVAL_REP_PENALTY="${EVAL_REP_PENALTY:-1.1}"

OUT_DIR="${OUT_ROOT}/${TASK,,}"
EVAL_OUT_DIR="${OUT_DIR}_eval"

# Method hyperparameters (identical across the six tasks).
NUM_CLUSTERS=4   # K
BASIS_M=4        # number of LoRA bases per module
LORA_R=16
ALPHA_ON=2.0     # asymmetric Dirichlet prior: diagonal entry
ALPHA_OFF=1.0    # off-diagonal entry

FORCE_PRECOMPUTE="${FORCE_PRECOMPUTE:-0}"
FORCE_SWEEP="${FORCE_SWEEP:-0}"
SWEEP_LOW="${SWEEP_LOW:-0}"
SWEEP_HIGH="${SWEEP_HIGH:-30}"
BALANCE_THRESHOLD="${BALANCE_THRESHOLD:-2.0}"
GPU0_LAYERS="${GPU0_LAYERS:--1}"
BUFFER_SEEDING="${BUFFER_SEEDING:-deterministic}"

# ── 1. precompute (skipped when the cache already exists) ─────────────────
if [[ ! -d "${CACHE_DIR}" || "${FORCE_PRECOMPUTE}" == "1" ]]; then
  : "${TRAIN_QUESTIONS:?need TRAIN_QUESTIONS (path to train_questions.json)}"
  : "${TRAIN_OUTPUTS:?need TRAIN_OUTPUTS (path to train_outputs.json)}"
  : "${DEV_QUESTIONS:?need DEV_QUESTIONS (path to dev_questions.json)}"
  : "${DEV_OUTPUTS:?need DEV_OUTPUTS (path to dev_outputs.json)}"

  PRE_ARGS=(
      --task_name       "${TASK}"
      --train_questions "${TRAIN_QUESTIONS}"
      --train_outputs   "${TRAIN_OUTPUTS}"
      --dev_questions   "${DEV_QUESTIONS}"
      --dev_outputs     "${DEV_OUTPUTS}"
      --out_dir         "${CACHE_DIR}"
      --num_clusters    "${NUM_CLUSTERS}"
      --seed            "${SEED}"
  )
  if [[ "${FORCE_SWEEP}" == "1" ]]; then
    echo "=================================================="
    echo "[PRECOMPUTE] ${TASK}  out=${CACHE_DIR}"
    echo "             cluster seed sweep ${SWEEP_LOW}..${SWEEP_HIGH}, balance<=${BALANCE_THRESHOLD}"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python precompute.py "${PRE_ARGS[@]}" \
        --seed_sweep_low    "${SWEEP_LOW}" \
        --seed_sweep_high   "${SWEEP_HIGH}" \
        --balance_threshold "${BALANCE_THRESHOLD}"
  else
    echo "=================================================="
    echo "[PRECOMPUTE] ${TASK}  out=${CACHE_DIR}  cluster_seed=${CLUSTER_SEED} (paper value)"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python precompute.py "${PRE_ARGS[@]}" \
        --cluster_seed "${CLUSTER_SEED}"
  fi
else
  echo "[PRECOMPUTE] cache exists (${CACHE_DIR}) — skipped (FORCE_PRECOMPUTE=1 to redo)"
fi

# ── 2. train ──────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "[TRAIN] ${TASK}  GPU=${GPU_IDS}  cache=${CACHE_DIR}  out=${OUT_DIR}"
echo "        seed=${SEED}  max_input_len=${IN_LEN}  max_target=${TRAIN_MAX}  grad_accum=${ACC}"
echo "=================================================="

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python main.py \
    --task_name      "${TASK}" \
    --base_model     "${BASE_MODEL}" \
    --cache_dir      "${CACHE_DIR}" \
    --out_dir        "${OUT_DIR}" \
    --seed           "${SEED}" \
    --eval_buffer_seeding "${BUFFER_SEEDING}" \
    --num_clusters   "${NUM_CLUSTERS}" \
    --basis_m        "${BASIS_M}" \
    --lora_r         "${LORA_R}" \
    --alpha_on       "${ALPHA_ON}" \
    --alpha_off      "${ALPHA_OFF}" \
    --grad_accum        "${ACC}" \
    --gpu0_layers       "${GPU0_LAYERS}" \
    --max_input_len     "${IN_LEN}" \
    --max_target_tokens "${TRAIN_MAX}" \
    --no_final_eval

# ── 3. eval ───────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "[EVAL] ${TASK}  ckpt=${OUT_DIR}  out=${EVAL_OUT_DIR}"
echo "       max_target=${EVAL_MAX}  no_repeat_ngram=${NGRAM}  rep_penalty=${EVAL_REP_PENALTY}"
echo "=================================================="

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python main.py \
    --task_name      "${TASK}" \
    --base_model     "${BASE_MODEL}" \
    --cache_dir      "${CACHE_DIR}" \
    --ckpt_dir       "${OUT_DIR}" \
    --out_dir        "${EVAL_OUT_DIR}" \
    --eval_only \
    --seed           "${SEED}" \
    --eval_buffer_seeding "${BUFFER_SEEDING}" \
    --num_clusters   "${NUM_CLUSTERS}" \
    --basis_m        "${BASIS_M}" \
    --lora_r         "${LORA_R}" \
    --alpha_on       "${ALPHA_ON}" \
    --alpha_off      "${ALPHA_OFF}" \
    --grad_accum        "${ACC}" \
    --gpu0_layers       "${GPU0_LAYERS}" \
    --max_input_len     "${IN_LEN}" \
    --max_target_tokens "${EVAL_MAX}" \
    --eval_repetition_penalty   "${EVAL_REP_PENALTY}" \
    --eval_no_repeat_ngram_size "${NGRAM}"

echo ""
echo "[DONE] ${TASK}  eval results: ${EVAL_OUT_DIR}/dev_eval_only.json"
