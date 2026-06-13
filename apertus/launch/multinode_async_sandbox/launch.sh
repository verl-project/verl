#!/usr/bin/env bash
#
# Submit a multi-node VERL training job with either a code-gym scheduler or a
# Kubernetes sandbox service configured for code reward evaluation.
# 
# Credits: https://github.com/swiss-ai/code-gym/tree/main

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED_SCRIPT="${SCRIPT_DIR}/_sandbox_scheduler.sbatch"
TRAIN_SCRIPT="${SCRIPT_DIR}/_verl_training.sbatch"
USERNAME="$(whoami)"

###############################################################################
# Experiment configuration
###############################################################################
PROJECT_NAME="${PROJECT_NAME:-apertus-rl-tests}"
SCRATCH_HOME="${SCRATCH_HOME:-/iopsstor/scratch/cscs/${USER}}"
WORKING_DIR="${WORKING_DIR:-${SCRATCH_HOME}/projects/verl}"
HOME="${SCRATCH_HOME}"
HF_HOME="${HF_HOME:-${SCRATCH_HOME}/huggingface}"
ENVIRONMENT_PATH="${ENVIRONMENT_PATH:-/capstor/store/cscs/swissai/infra01/reasoning/raas/docker/vs:251215-degenstop/env.toml}"
PY_DEPS_ROOT="${PY_DEPS_ROOT:-}"
PY_DEPS_DIR="${PY_DEPS_DIR:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/capstor/store/cscs/swissai/infra01/reasoning/models/Apertus-1p5-8B-sft-capfilter-linear-it8816}"
TOKENIZERS_ROOT="${TOKENIZERS_ROOT:-/capstor/store/cscs/swissai/infra01/reasoning/models/tokenizers}"
MULTIMODAL="${MULTIMODAL:-false}"
if [[ -z "${TOKENIZER_NAME_OR_PATH:-}" ]]; then
  if [[ "${MULTIMODAL}" == "true" ]]; then
    TOKENIZER_NAME_OR_PATH="${TOKENIZERS_ROOT}/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
  else
    TOKENIZER_NAME_OR_PATH="${TOKENIZERS_ROOT}/apertus_emu3.5_wavtok_text_only"
  fi
fi
CONFIG_NAME="${CONFIG_NAME:-async}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
TRAIN_NNODES="${TRAIN_NNODES:-4}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-2}"
NNODES="${NNODES:-$((TRAIN_NNODES + ROLLOUT_NNODES))}"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/apertus_demo_rl}"
# TRAINING_DATA_DIR=/users/jgarcagi/iopsstor/projects/verl/apertus/data/apertus_demo_rl_tools
FORCE_THINKING="${FORCE_THINKING:-false}"
THINK_PREFIX_TOKEN="${THINK_PREFIX_TOKEN:-<|inner_prefix|>}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
SEED="${SEED:-85}"
ROLLOUT_N="${ROLLOUT_N:-8}"
USE_GROUP_FILTERING="${USE_GROUP_FILTERING:-true}"
JOB_NAME="${JOB_NAME:-debug}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

WANDB_ENTITY="${WANDB_ENTITY:-apertus}"
WANDB_BACKGROUND_SYNC="${WANDB_BACKGROUND_SYNC:-false}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_SYNC_INTERVAL_SECONDS="${WANDB_SYNC_INTERVAL_SECONDS:-60}"
WANDB_REQUIRE_SERVICE="${WANDB_REQUIRE_SERVICE:-}"
WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-}"
WANDB_SYNC_UPLOAD_MODE="${WANDB_SYNC_UPLOAD_MODE:-}"
WANDB_DIR="${WANDB_DIR:-}"

ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-}"
ROLLOUT_TOTAL_ROLLOUT_STEPS="${ROLLOUT_TOTAL_ROLLOUT_STEPS:-}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-}"
ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-}"
ASYNC_TRIGGER_PARAMETER_SYNC_STEP="${ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-}"
ASYNC_STALENESS_THRESHOLD="${ASYNC_STALENESS_THRESHOLD:-}"
ASYNC_STEADY_WARMUP_STEPS="${ASYNC_STEADY_WARMUP_STEPS:-}"

###############################################################################
# Sandbox configuration
###############################################################################

# Set REASONING_GYM_DIR="" to install reasoning-gym from PyPI.
REASONING_GYM_DIR="${REASONING_GYM_DIR:-${SCRATCH_HOME}/projects/r-gym}"
TOOL_GYM_DIR="${TOOL_GYM_DIR:-${SCRATCH_HOME}/projects/tool-gym}"
TOOL_GYM_FUNCTION_TOOL_PATH="${TOOL_GYM_FUNCTION_TOOL_PATH:-/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/toolgym_test_v2/apertus_function_tools.py}"
SANDBOX_BACKEND="kubernetes"  # kubernetes, codegym, or none
KUBERNETES_SANDBOX_URL="https://sandbox-dev.swissai.svc.cscs.ch"
CODE_GYM_DIR="" # ${SCRATCH_HOME}/projects/code-gym}  # Not needed if using kubernetes
PORT="${PORT:-8000}"
POLL_SECS="${POLL_SECS:-3}"
MAX_WAIT="${MAX_WAIT:-$((60 * 10))}"
GIVEN_URL="${SCHEDULER_URL:-}"  # potentially reuse a running code-gym scheduler
NO_FORMAT="${NO_FORMAT:-false}"  # disable tool-formatting (legacy plain-text rollouts)
SANDBOX_REWARD_CONTINUOUS="${SANDBOX_REWARD_CONTINUOUS:-false}" # default is binary reward

log(){ echo -e "$*" >&2; }

sanitize_job_name() {
  local value="$1"
  sed -E 's#[/:.]#-#g; s#[^[:alnum:]_-]#-#g; s/^-+//; s/-+$//; s/-+/-/g' <<<"${value}"
}

resolve_run_name_and_dir() {
  local expected_nnodes
  local group_filtering_tag
  local model_tag
  local thinking_tag

  expected_nnodes=$((TRAIN_NNODES + ROLLOUT_NNODES))
  if [[ "${NNODES}" -ne "${expected_nnodes}" ]]; then
    echo "NNODES=${NNODES} must equal TRAIN_NNODES + ROLLOUT_NNODES (${expected_nnodes})." >&2
    exit 1
  fi

  model_tag="$(sanitize_job_name "$(basename "${MODEL_NAME_OR_PATH}")")"
  group_filtering_tag=""
  if [[ "${USE_GROUP_FILTERING}" == "true" ]]; then
    group_filtering_tag="dapo-"
  fi
  thinking_tag=""
  if [[ "${ENABLE_THINKING}" == "true" ]]; then
    thinking_tag="${thinking_tag}-think"
  fi
  if [[ "${FORCE_THINKING}" == "true" ]]; then
    thinking_tag="-force${thinking_tag}"
  fi

  if [[ -z "${JOB_NAME}" ]]; then
    JOB_NAME="async__${CONFIG_NAME}_${group_filtering_tag}${model_tag}_${TRAIN_NNODES}tn-${ROLLOUT_NNODES}rn__s${SEED}${thinking_tag}"
  fi
  JOB_NAME="$(sanitize_job_name "${JOB_NAME}")"
  RUN_NAME="${JOB_NAME}__$(date +%Y%m%d-%H%M%S)"
  RUN_DIR="${WORKING_DIR}/outputs/${PROJECT_NAME}/${RUN_NAME}"
  SCHED_JOB_NAME="${JOB_NAME}_sched"
  TRAIN_JOB_NAME="${JOB_NAME}_train"

  mkdir -p "${RUN_DIR}"
}

probe_ok() {
  local host="$1"
  local port="$2"
  local url="$3"
  if command -v nc >/dev/null 2>&1; then
    timeout 2s nc -z "${host}" "${port}"
  elif command -v curl >/dev/null 2>&1; then
    timeout 2s curl -sS --max-time 2 "${url}" >/dev/null
  else
    (exec 3<>/dev/tcp/"${host}"/"${port}") 2>/dev/null
  fi
}

resolve_sandbox_backend() {
  case "${SANDBOX_BACKEND}" in
    codegym|kubernetes)
      ;;
    none)
      ;;
    *)
      echo "Unsupported SANDBOX_BACKEND=${SANDBOX_BACKEND}. Use codegym, kubernetes, or none." >&2
      exit 1
      ;;
  esac

  if [[ "${SANDBOX_BACKEND}" == "kubernetes" ]]; then
    if [[ -z "${KUBERNETES_SANDBOX_URL}" ]]; then
      echo "KUBERNETES_SANDBOX_URL must be set when using SANDBOX_BACKEND=kubernetes." >&2
      exit 1
    fi
    KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL%/}"
  elif [[ "${SANDBOX_BACKEND}" == "codegym" && -z "${CODE_GYM_DIR}" && -z "${GIVEN_URL}" ]]; then
    echo "CODE_GYM_DIR or SCHEDULER_URL must be set when using SANDBOX_BACKEND=codegym." >&2
    exit 1
  fi
}

resolve_run_name_and_dir
resolve_sandbox_backend

# ==========================================
# STEP 1 & 2: Sandbox setup
# ==========================================
if [[ "${SANDBOX_BACKEND}" == "none" ]]; then
  log "\n[1/4 & 2/4] Code sandbox disabled for this run"
  URL=""
  NODE_CLEAN="n/a"
  SCHED_ID="skipped"
elif [[ "${SANDBOX_BACKEND}" == "kubernetes" ]]; then
  log "\n[1/4 & 2/4] Check Kubernetes sandbox"
  if ! curl -fsS --max-time 10 "${KUBERNETES_SANDBOX_URL}/" >/dev/null; then
    echo "Kubernetes sandbox is not reachable at ${KUBERNETES_SANDBOX_URL}" >&2
    exit 1
  fi
  URL="${KUBERNETES_SANDBOX_URL}"
  NODE_CLEAN="n/a"
  SCHED_ID="skipped"
elif [[ -z "${GIVEN_URL}" ]]; then
  [[ -f "${SCHED_SCRIPT}" ]] || { echo "Missing ${SCHED_SCRIPT}" >&2; exit 1; }

  log "\n[1/4] Submit code-gym sandbox scheduler"
  log "  -> job-name=${SCHED_JOB_NAME} time=${SLURM_TIME}"
  SCHED_SUBMIT="$(sbatch \
    --job-name="${SCHED_JOB_NAME}" \
    --time="${SLURM_TIME}" \
    --output="${RUN_DIR}/sandbox_scheduler_%j.out" \
    --error="${RUN_DIR}/sandbox_scheduler_%j.err" \
    --export=ALL,CODE_GYM_DIR="${CODE_GYM_DIR}",PORT="${PORT}" \
    "${SCHED_SCRIPT}")"
  SCHED_ID="$(awk '{print $NF}' <<<"${SCHED_SUBMIT}")"
  [[ "${SCHED_ID}" =~ ^[0-9]+$ ]] || { echo "Failed to parse scheduler job id: ${SCHED_SUBMIT}" >&2; exit 1; }
  log "  -> Scheduler JobID: ${SCHED_ID}"

  log "\n[2/4] Wait for scheduler node"
  elapsed=0
  state=""
  node=""
  while :; do
    read -r state node < <(squeue -h -j "${SCHED_ID}" -o "%T %N" || true)
    [[ -n "${state}" ]] || state="PENDING"
    [[ -n "${node}" ]] || node="n/a"
    log "  state=${state} node=${node}"
    if [[ "${state}" == "RUNNING" && "${node}" != "n/a" ]]; then
      break
    fi
    if [[ "${state}" =~ (FAILED|CANCELLED|TIMEOUT|COMPLETED) ]]; then
      echo "Scheduler ended early: ${state}" >&2
      exit 1
    fi
    (( elapsed += POLL_SECS ))
    (( elapsed > MAX_WAIT )) && { echo "Timeout waiting for scheduler RUNNING" >&2; exit 1; }
    sleep "${POLL_SECS}"
  done

  NODE_CLEAN="$(sed -E 's/[\[\],]//g; s/ .*//g' <<<"${node}")"
  URL="http://${NODE_CLEAN}:${PORT}"
  SCHEDULER_URL="${URL}"
else
  log "\n[1/4 & 2/4] Reusing code-gym scheduler ${GIVEN_URL}"
  URL="${GIVEN_URL%/}"
  SCHEDULER_URL="${URL}"
  SCHED_ID="skipped"
  HOST_PORT="${URL#*://}"
  NODE_CLEAN="${HOST_PORT%:*}"
  PORT_FROM_URL="${HOST_PORT##*:}"
  if [[ "${PORT_FROM_URL}" != "${HOST_PORT}" ]]; then
    PORT="${PORT_FROM_URL}"
  fi
fi

# ==========================================
# STEP 3: Report sandbox endpoint
# ==========================================
if [[ -n "${URL}" && "${SANDBOX_BACKEND}" == "codegym" ]]; then
  log "\n[3/4] Probe code-gym scheduler ${NODE_CLEAN}:${PORT}"
  elapsed=0
  until probe_ok "${NODE_CLEAN}" "${PORT}" "${URL}"; do
    (( elapsed += POLL_SECS ))
    (( elapsed > MAX_WAIT )) && { echo "Port never opened: ${NODE_CLEAN}:${PORT}" >&2; exit 1; }
    sleep "${POLL_SECS}"
  done
  log "  -> Scheduler reachable at ${URL}"
elif [[ -n "${URL}" ]]; then
  log "\n[3/4] Kubernetes sandbox reachable at ${URL}"
else
  log "\n[3/4] No sandbox probe needed"
fi

# ==========================================
# STEP 4: Submit VeRL Job
# ==========================================
log "\n[4/4] Submit multi-node async VERL training"
log "  -> job-name=${TRAIN_JOB_NAME} time=${SLURM_TIME} train=${TRAIN_NNODES} rollout=${ROLLOUT_NNODES} total=${NNODES}"
log "  -> config=${CONFIG_NAME} model=${MODEL_NAME_OR_PATH}"
log "  -> data=${TRAINING_DATA_DIR} seed=${SEED} rollout_n=${ROLLOUT_N}"
log "  -> group_filtering=${USE_GROUP_FILTERING} enable_thinking=${ENABLE_THINKING} force_thinking=${FORCE_THINKING}"
log "  -> no_format=${NO_FORMAT}"
if [[ "${WANDB_BACKGROUND_SYNC}" == "true" ]]; then
  log "  -> output=${RUN_DIR} wandb_mode=${WANDB_MODE} wandb_sync_interval=${WANDB_SYNC_INTERVAL_SECONDS}s"
else
  log "  -> output=${RUN_DIR}"
fi
if [[ -n "${URL}" ]]; then
  log "  -> sandbox_backend=${SANDBOX_BACKEND} sandbox_url=${URL} continuous=${SANDBOX_REWARD_CONTINUOUS}"
else
  log "  -> sandbox_backend=${SANDBOX_BACKEND} sandbox_url=disabled continuous=${SANDBOX_REWARD_CONTINUOUS}"
fi
log "  -> reasoning-gym=${REASONING_GYM_DIR:-PyPI reasoning-gym}"
log "  -> tool-gym=${TOOL_GYM_DIR}"
log "  -> tool-gym function tools=${TOOL_GYM_FUNCTION_TOOL_PATH}"

join_export_vars() {
  local out=""
  local item=""
  for item in "$@"; do
    if [[ -z "${out}" ]]; then
      out="${item}"
    else
      out+=",${item}"
    fi
  done
  printf '%s' "${out}"
}

EXPORT_VARS=(
  "ALL"
  "MULTIMODAL=${MULTIMODAL}"
  "SANDBOX_BACKEND=${SANDBOX_BACKEND}"
  "SCHEDULER_URL=${SCHEDULER_URL:-}"
  "KUBERNETES_SANDBOX_URL=${KUBERNETES_SANDBOX_URL:-}"
  "SANDBOX_REWARD_CONTINUOUS=${SANDBOX_REWARD_CONTINUOUS}"
  "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
  "TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH}"
  "CONFIG_NAME=${CONFIG_NAME}"
  "NNODES=${NNODES}"
  "TRAIN_NNODES=${TRAIN_NNODES}"
  "ROLLOUT_NNODES=${ROLLOUT_NNODES}"
  "TRAINING_DATA_DIR=${TRAINING_DATA_DIR}"
  "NO_FORMAT=${NO_FORMAT}"
  "ENABLE_THINKING=${ENABLE_THINKING}"
  "FORCE_THINKING=${FORCE_THINKING}"
  "THINK_PREFIX_TOKEN=${THINK_PREFIX_TOKEN}"
  "DEGENERATION_EARLY_STOP=${DEGENERATION_EARLY_STOP:-}"
  "DEGENERATION_EARLY_STOP_STRIDE=${DEGENERATION_EARLY_STOP_STRIDE:-}"
  "SEED=${SEED}"
  "ROLLOUT_N=${ROLLOUT_N}"
  "USE_GROUP_FILTERING=${USE_GROUP_FILTERING}"
  "PROJECT_NAME=${PROJECT_NAME}"
  "RUN_NAME=${RUN_NAME}"
  "RUN_DIR=${RUN_DIR}"
  "WANDB_BACKGROUND_SYNC=${WANDB_BACKGROUND_SYNC}"
  "WANDB_ENTITY=${WANDB_ENTITY}"
  "WANDB_MODE=${WANDB_MODE}"
  "WANDB_SYNC_INTERVAL_SECONDS=${WANDB_SYNC_INTERVAL_SECONDS}"
  "WANDB_REQUIRE_SERVICE=${WANDB_REQUIRE_SERVICE}"
  "WANDB_DISABLE_SERVICE=${WANDB_DISABLE_SERVICE}"
  "WANDB_SYNC_UPLOAD_MODE=${WANDB_SYNC_UPLOAD_MODE}"
  "WANDB_DIR=${WANDB_DIR}"
  "VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
  "PY_DEPS_ROOT=${PY_DEPS_ROOT}"
  "PY_DEPS_DIR=${PY_DEPS_DIR}"
  "ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE}"
  "ROLLOUT_TOTAL_ROLLOUT_STEPS=${ROLLOUT_TOTAL_ROLLOUT_STEPS}"
  "TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ}"
  "TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ}"
  "ASYNC_REQUIRE_BATCHES=${ASYNC_REQUIRE_BATCHES}"
  "ASYNC_TRIGGER_PARAMETER_SYNC_STEP=${ASYNC_TRIGGER_PARAMETER_SYNC_STEP}"
  "ASYNC_STALENESS_THRESHOLD=${ASYNC_STALENESS_THRESHOLD}"
  "ASYNC_STEADY_WARMUP_STEPS=${ASYNC_STEADY_WARMUP_STEPS}"
  "WORKING_DIR=${WORKING_DIR}"
  "HOME=${HOME}"
  "HF_HOME=${HF_HOME}"
  "ENVIRONMENT_PATH=${ENVIRONMENT_PATH}"
  "REASONING_GYM_DIR=${REASONING_GYM_DIR}"
  "TOOL_GYM_DIR=${TOOL_GYM_DIR}"
  "TOOL_GYM_FUNCTION_TOOL_PATH=${TOOL_GYM_FUNCTION_TOOL_PATH}"
  "JOB_NAME=${JOB_NAME}"
)

EXPORT_SPEC="$(join_export_vars "${EXPORT_VARS[@]}")"

TRAIN_SUBMIT="$(sbatch \
  --job-name="${TRAIN_JOB_NAME}" \
  --nodes="${NNODES}" \
  --time="${SLURM_TIME}" \
  --output="${RUN_DIR}/multinode_async_sandbox_%j.out" \
  --error="${RUN_DIR}/multinode_async_sandbox_%j.err" \
  --export="${EXPORT_SPEC}" \
  "${TRAIN_SCRIPT}" "$@")"
TRAIN_ID="$(awk '{print $NF}' <<<"${TRAIN_SUBMIT}")"
[[ "${TRAIN_ID}" =~ ^[0-9]+$ ]] || { echo "Failed to parse training job id: ${TRAIN_SUBMIT}" >&2; exit 1; }
log "  -> Training JobID: ${TRAIN_ID}"

# ==========================================
echo
echo "Monitor:"
if [[ "${SCHED_ID}" != "skipped" ]]; then
  echo "  squeue -j ${SCHED_ID},${TRAIN_ID}"
  echo "  tail -f ${RUN_DIR}/sandbox_scheduler_${SCHED_ID}.out"
else
  echo "  squeue -j ${TRAIN_ID}"
fi
echo "  tail -f ${RUN_DIR}/multinode_async_sandbox_${TRAIN_ID}.out"
