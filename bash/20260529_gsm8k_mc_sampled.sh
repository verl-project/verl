#!/usr/bin/env bash
set -euo pipefail

# Current implementation:
#   on-policy loop = live GSM8K candidate generation -> generated parquet -> VERL/GRPO train
#   -> roll out the saved HF actor checkpoint -> repeat.
#
# This is intentionally different from the older static sampled-parquet launcher:
#   bash/20260327_gsm8k_mc_sampled.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILDER_DIR="$REPO_ROOT/reliable-gsm8k-builder"

DATE_TAG=${DATE_TAG:-20260529}
GPU_IDS=${GPU_IDS:-}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
GENERATOR_PROFILE=${GENERATOR_PROFILE:-qwen25_3b}
INFERENCE_PROFILE=${INFERENCE_PROFILE:-sample_balanced}
TRAIN_DATASET=${TRAIN_DATASET:-mc_onecorrect}  # mc_onecorrect | mc_allwrong | oe

ITERATIONS=${ITERATIONS:-3}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SPLIT=${SPLIT:-train}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-multiple_choice_question_study}
IMPLEMENTATION_TAG=${IMPLEMENTATION_TAG:-gsm8k_mc_sampled_onpolicy_livegen_parseronly_grpo}
RUN_PREFIX=${RUN_PREFIX:-${DATE_TAG}_${IMPLEMENTATION_TAG}_${GENERATOR_PROFILE}_${TRAIN_DATASET}_n${NUM_SAMPLES}_it${ITERATIONS}}

OUTPUT_ROOT=${OUTPUT_ROOT:-$BUILDER_DIR/runs_on_policy}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/on_policy}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${RUN_PREFIX}.log}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-999999}  # positive; VERL still saves on final step
TEST_FREQ=${TEST_FREQ:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}

USE_JUDGE=${USE_JUDGE:-0}
JUDGE_PROFILE=${JUDGE_PROFILE:-judgelm_7b}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$LOG_DIR"

CMD=(
  python "$BUILDER_DIR/run_on_policy_loop.py"
  --run-prefix "$RUN_PREFIX"
  --iterations "$ITERATIONS"
  --num-samples "$NUM_SAMPLES"
  --split "$SPLIT"
  --base-model "$BASE_MODEL"
  --generator-profile "$GENERATOR_PROFILE"
  --inference-profile "$INFERENCE_PROFILE"
  --train-dataset "$TRAIN_DATASET"
  --output-root "$OUTPUT_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --val-file "$VAL_FILE"
  --project-name "$PROJECT_NAME"
  --total-epochs "$TOTAL_EPOCHS"
  --save-freq "$SAVE_FREQ"
  --test-freq "$TEST_FREQ"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE"
  --actor-micro-batch-size "$ACTOR_MICRO_BATCH_SIZE"
  --rollout-n "$ROLLOUT_N"
  --learning-rate "$LEARNING_RATE"
)

if [[ -n "$GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  CMD+=(--gpu-ids "$GPU_IDS")
fi

if [[ "$USE_JUDGE" == "1" ]]; then
  CMD+=(--use-judge --judge-profile "$JUDGE_PROFILE")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "[20260529_gsm8k_mc_sampled] implementation=$IMPLEMENTATION_TAG"
echo "[20260529_gsm8k_mc_sampled] run_prefix=$RUN_PREFIX"
echo "[20260529_gsm8k_mc_sampled] base_model=$BASE_MODEL"
echo "[20260529_gsm8k_mc_sampled] generator_profile=$GENERATOR_PROFILE inference_profile=$INFERENCE_PROFILE"
echo "[20260529_gsm8k_mc_sampled] train_dataset=$TRAIN_DATASET iterations=$ITERATIONS num_samples=$NUM_SAMPLES"
echo "[20260529_gsm8k_mc_sampled] output_root=$OUTPUT_ROOT"
echo "[20260529_gsm8k_mc_sampled] checkpoint_root=$CHECKPOINT_ROOT"
echo "[20260529_gsm8k_mc_sampled] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[20260529_gsm8k_mc_sampled] log_file=$LOG_FILE"

PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=1 "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
