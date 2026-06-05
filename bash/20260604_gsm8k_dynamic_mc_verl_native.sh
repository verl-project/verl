#!/usr/bin/env bash
set -euo pipefail

# Option C:
#   Start VERL once, keep the actor/rollout model resident on GPU, generate
#   Stage 1 candidates inside VERL rollouts, queue verified Stage 2 MC prompts
#   through the dataset hooks, and update the actor with GRPO.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILDER_DIR="$REPO_ROOT/reliable-gsm8k-builder"

export PYTHONPATH="$BUILDER_DIR/src:$REPO_ROOT:${PYTHONPATH:-}"

DATE_TAG=${DATE_TAG:-20260604}
RUN_ID=${RUN_ID:-${DATE_TAG}_gsm8k_dynamic_mc_verl_native_grpo}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
SPLIT=${SPLIT:-train}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
GPU_IDS=${GPU_IDS:-}

OUTPUT_ROOT=${OUTPUT_ROOT:-$BUILDER_DIR/runs_verl_native_dynamic_mc}
SEED_DIR=${SEED_DIR:-$OUTPUT_ROOT/seed}
SEED_FILE=${SEED_FILE:-$SEED_DIR/${RUN_ID}_${SPLIT}_stage1.parquet}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints/dynamic_mc/$RUN_ID}
if [[ -n "${VAL_FILE:-}" ]]; then
  VAL_FILE_PROVIDED=1
else
  VAL_FILE_PROVIDED=0
fi
VAL_FILE=${VAL_FILE:-$SEED_DIR/${RUN_ID}_test_stage1.parquet}
VAL_NUM_SAMPLES=${VAL_NUM_SAMPLES:-128}
if [[ -z "${VAL_STAGE1_PROMPT_COUNT:-}" ]]; then
  if [[ -n "${VAL_INCORRECT_TARGET_COUNT:-}" ]]; then
    VAL_STAGE1_PROMPT_COUNT=$((VAL_INCORRECT_TARGET_COUNT + 1))
  else
    VAL_STAGE1_PROMPT_COUNT=1
  fi
fi
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${RUN_ID}.log}

if [[ -z "${STAGE1_PROMPT_COUNT:-}" ]]; then
  if [[ -n "${INCORRECT_PROMPT_COUNT:-}" ]]; then
    STAGE1_PROMPT_COUNT=$((INCORRECT_PROMPT_COUNT + 1))
  elif [[ -n "${INCORRECT_TARGET_COUNT:-}" ]]; then
    STAGE1_PROMPT_COUNT=$((INCORRECT_TARGET_COUNT + 1))
  else
    STAGE1_PROMPT_COUNT=4
  fi
fi
STAGE1_PROMPT_MODE=${STAGE1_PROMPT_MODE:-neutral}
STAGE2_INCORRECT_COUNT=${STAGE2_INCORRECT_COUNT:-3}
MAX_STAGE2_PER_QUESTION=${MAX_STAGE2_PER_QUESTION:-4}
MAX_NEW_STAGE2_PER_BATCH=${MAX_NEW_STAGE2_PER_BATCH:-256}
STAGE2_CANDIDATE_MAX_CHARS=${STAGE2_CANDIDATE_MAX_CHARS:-2000}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-multiple_choice_question_study}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

DRY_RUN=${DRY_RUN:-0}

if [[ "$STAGE2_INCORRECT_COUNT" != "3" ]]; then
  echo "[dynamic-mc-verl-native] STAGE2_INCORRECT_COUNT must be 3 because Stage 2 uses four options with one correct answer." >&2
  exit 1
fi

if [[ "$STAGE1_PROMPT_MODE" != "neutral" && "$STAGE1_PROMPT_MODE" != "role" ]]; then
  echo "[dynamic-mc-verl-native] STAGE1_PROMPT_MODE must be neutral or role." >&2
  exit 1
fi

if ! [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[dynamic-mc-verl-native] NUM_SAMPLES must be a positive integer." >&2
  exit 1
fi

if ! [[ "$STAGE1_PROMPT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[dynamic-mc-verl-native] STAGE1_PROMPT_COUNT must be a positive integer." >&2
  exit 1
fi

if ! [[ "$ROLLOUT_N" =~ ^[1-9][0-9]*$ ]]; then
  echo "[dynamic-mc-verl-native] ROLLOUT_N must be a positive integer." >&2
  exit 1
fi

if ! [[ "$GEN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[dynamic-mc-verl-native] GEN_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi

if ! [[ "$STAGE2_CANDIDATE_MAX_CHARS" =~ ^[0-9]+$ ]]; then
  echo "[dynamic-mc-verl-native] STAGE2_CANDIDATE_MAX_CHARS must be a non-negative integer." >&2
  exit 1
fi

INITIAL_STAGE1_ROWS=$((NUM_SAMPLES * STAGE1_PROMPT_COUNT))
if [[ "$INITIAL_STAGE1_ROWS" -lt "$GEN_BATCH_SIZE" ]]; then
  echo "[dynamic-mc-verl-native] Initial Stage 1 seed rows ($INITIAL_STAGE1_ROWS) must be >= GEN_BATCH_SIZE ($GEN_BATCH_SIZE)." >&2
  echo "[dynamic-mc-verl-native] Increase NUM_SAMPLES/STAGE1_PROMPT_COUNT or lower GEN_BATCH_SIZE for tiny smoke runs." >&2
  exit 1
fi

if ! [[ "$TOTAL_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$TOTAL_EPOCHS" -lt 2 ]]; then
  echo "[dynamic-mc-verl-native] TOTAL_EPOCHS must be at least 2 so queued Stage 2 rows can be consumed on a later dataloader pass." >&2
  exit 1
fi

mkdir -p "$SEED_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

if [[ "$DRY_RUN" != "1" ]]; then
  exec > >(tee "$LOG_FILE") 2>&1
fi

if [[ -n "$GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

echo "[dynamic-mc-verl-native] run_id=$RUN_ID"
echo "[dynamic-mc-verl-native] model_path=$MODEL_PATH"
echo "[dynamic-mc-verl-native] seed_file=$SEED_FILE"
echo "[dynamic-mc-verl-native] val_file=$VAL_FILE"
echo "[dynamic-mc-verl-native] checkpoint_dir=$CHECKPOINT_DIR"
echo "[dynamic-mc-verl-native] stage1_prompt_count=$STAGE1_PROMPT_COUNT rollout_n=$ROLLOUT_N stage1_attempts_per_question=$((STAGE1_PROMPT_COUNT * ROLLOUT_N)) initial_stage1_rows=$INITIAL_STAGE1_ROWS"
echo "[dynamic-mc-verl-native] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

CREATE_SEED_CMD=(
  python "$BUILDER_DIR/create_dynamic_mc_seed.py"
  --split "$SPLIT"
  --num-samples "$NUM_SAMPLES"
  --stage1-prompt-count "$STAGE1_PROMPT_COUNT"
  --prompt-mode "$STAGE1_PROMPT_MODE"
  --output "$SEED_FILE"
)

CREATE_VAL_CMD=()
if [[ "$VAL_FILE_PROVIDED" == "0" ]]; then
  CREATE_VAL_CMD=(
    python "$BUILDER_DIR/create_dynamic_mc_seed.py"
    --split test
    --num-samples "$VAL_NUM_SAMPLES"
    --stage1-prompt-count "$VAL_STAGE1_PROMPT_COUNT"
    --prompt-mode "$STAGE1_PROMPT_MODE"
    --output "$VAL_FILE"
  )
fi

VERL_CMD=(
  python -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  data.train_files="$SEED_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.gen_batch_size="$GEN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.prompt_key=prompt
  data.reward_fn_key=data_source
  data.shuffle=False
  data.dataloader_num_workers=0
  data.custom_cls.path=pkg://reliable_gsm8k.verl_dynamic_mc
  data.custom_cls.name=GSM8KDynamicMCDataset
  +data.dynamic_mc.stage1_prompt_count="$STAGE1_PROMPT_COUNT"
  +data.dynamic_mc.stage1_prompt_mode="$STAGE1_PROMPT_MODE"
  +data.dynamic_mc.stage2_incorrect_count="$STAGE2_INCORRECT_COUNT"
  +data.dynamic_mc.max_stage2_per_question="$MAX_STAGE2_PER_QUESTION"
  +data.dynamic_mc.max_new_stage2_per_batch="$MAX_NEW_STAGE2_PER_BATCH"
  +data.dynamic_mc.stage2_candidate_max_chars="$STAGE2_CANDIDATE_MAX_CHARS"
  +data.dynamic_mc.stage2_insert_strategy=prepend
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_SIZE"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.val_before_train=False
  trainer.n_gpus_per_node="${GPUS:-1}"
  trainer.nnodes=1
  trainer.resume_mode=disable
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.logger="$TRAINER_LOGGER"
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
  trainer.default_local_dir="$CHECKPOINT_DIR"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$RUN_ID"
  custom_reward_function.path="$BUILDER_DIR/src/reliable_gsm8k/verl_dynamic_mc.py"
  custom_reward_function.name=compute_score
  val_custom_reward_function.path="$BUILDER_DIR/src/reliable_gsm8k/verl_dynamic_mc.py"
  val_custom_reward_function.name=compute_score
)

if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
  VERL_CMD+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

echo "[dynamic-mc-verl-native] ${CREATE_SEED_CMD[*]}"
if [[ "${#CREATE_VAL_CMD[@]}" -gt 0 ]]; then
  echo "[dynamic-mc-verl-native] ${CREATE_VAL_CMD[*]}"
else
  echo "[dynamic-mc-verl-native] using caller-provided VAL_FILE=$VAL_FILE"
fi
echo "[dynamic-mc-verl-native] ${VERL_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${CREATE_SEED_CMD[@]}"
if [[ "${#CREATE_VAL_CMD[@]}" -gt 0 ]]; then
  "${CREATE_VAL_CMD[@]}"
fi
PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=1 "${VERL_CMD[@]}"
