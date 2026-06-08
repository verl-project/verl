#!/usr/bin/env bash
#
# Submit a code-gym sandbox scheduler job and, once reachable, submit the
# multi-node VERL training job with SCHEDULER_URL injected.
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
PROJECT_NAME=apertus-rl-tests
WORKING_DIR="/iopsstor/scratch/cscs/${USER}/projects/verl"
HOME=/iopsstor/scratch/cscs/${USER}
HF_HOME=/iopsstor/scratch/cscs/${USER}/huggingface
ENVIRONMENT_PATH=/capstor/store/cscs/swissai/infra01/reasoning/raas/docker/vs:251215-patched/env.toml

MODEL_NAME_OR_PATH=/capstor/store/cscs/swissai/infra01/reasoning/models/Apertus-1p5-8B-sft-capfilter-linear-it8816
TOKENIZER_NAME_OR_PATH=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed
CONFIG_NAME=async
SLURM_TIME=04:00:00
TRAIN_NNODES=2
ROLLOUT_NNODES=2
NNODES=$((TRAIN_NNODES + ROLLOUT_NNODES))
TRAINING_DATA_DIR=/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/apertus_demo_rl
ENABLE_THINKING=false
FORCE_THINKING=false
THINK_PREFIX_TOKEN="<|inner_prefix|>"
SEED=85
ROLLOUT_N=8
USE_GROUP_FILTERING=true
JOB_NAME=""

###############################################################################
# Sandbox configuration
###############################################################################

CODE_GYM_DIR=/iopsstor/scratch/cscs/${USER}/projects/code-gym
PORT=8000
POLL_SECS=3
MAX_WAIT=$((60 * 10))
GIVEN_URL="${SCHEDULER_URL:-}"  # potentially reuse running scheduler
CODEGYM_REWARD_CONTINUOUS=false # default is binary reward

log(){ echo -e "$*" >&2; }

clear_inherited_pyxis_options() {
  local name
  while IFS='=' read -r name _; do
    case "${name}" in
      SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_*) unset "${name}" ;;
    esac
  done < <(env)
}

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
    thinking_tag="-think"
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

resolve_run_name_and_dir
clear_inherited_pyxis_options

# ==========================================
# STEP 1 & 2: Scheduler logic (or skip)
# ==========================================
if [[ -z "${GIVEN_URL}" ]]; then
  [[ -f "${SCHED_SCRIPT}" ]] || { echo "Missing ${SCHED_SCRIPT}" >&2; exit 1; }

  log "\n[1/4] Submit sandbox scheduler"
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
else
  log "\n[1/4 & 2/4] Reusing scheduler ${GIVEN_URL}"
  URL="${GIVEN_URL%/}"
  SCHED_ID="skipped"
  HOST_PORT="${URL#*://}"
  NODE_CLEAN="${HOST_PORT%:*}"
  PORT_FROM_URL="${HOST_PORT##*:}"
  if [[ "${PORT_FROM_URL}" != "${HOST_PORT}" ]]; then
    PORT="${PORT_FROM_URL}"
  fi
fi

# ==========================================
# STEP 3: Probe TCP (Runs regardless)
# ==========================================
log "\n[3/4] Probe scheduler ${NODE_CLEAN}:${PORT}"
elapsed=0
until probe_ok "${NODE_CLEAN}" "${PORT}" "${URL}"; do
  (( elapsed += POLL_SECS ))
  (( elapsed > MAX_WAIT )) && { echo "Port never opened: ${NODE_CLEAN}:${PORT}" >&2; exit 1; }
  sleep "${POLL_SECS}"
done
log "  -> Scheduler reachable at ${URL}"

# ==========================================
# STEP 4: Submit VeRL Job
# ==========================================
log "\n[4/4] Submit multi-node async VERL training"
log "  -> job-name=${TRAIN_JOB_NAME} time=${SLURM_TIME} train=${TRAIN_NNODES} rollout=${ROLLOUT_NNODES} total=${NNODES}"
log "  -> config=${CONFIG_NAME} model=${MODEL_NAME_OR_PATH}"
log "  -> data=${TRAINING_DATA_DIR} seed=${SEED} rollout_n=${ROLLOUT_N}"
log "  -> group_filtering=${USE_GROUP_FILTERING} enable_thinking=${ENABLE_THINKING} force_thinking=${FORCE_THINKING}"
log "  -> output=${RUN_DIR}"
log "  -> scheduler=${URL} code-gym continuous=${CODEGYM_REWARD_CONTINUOUS}"

TRAIN_SUBMIT="$(sbatch \
  --job-name="${TRAIN_JOB_NAME}" \
  --nodes="${NNODES}" \
  --time="${SLURM_TIME}" \
  --output="${RUN_DIR}/multinode_async_sandbox_%j.out" \
  --error="${RUN_DIR}/multinode_async_sandbox_%j.err" \
  --export=ALL,SCHEDULER_URL="${URL}",CODEGYM_REWARD_CONTINUOUS="${CODEGYM_REWARD_CONTINUOUS}",MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}",TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH}",CONFIG_NAME="${CONFIG_NAME}",NNODES="${NNODES}",TRAIN_NNODES="${TRAIN_NNODES}",ROLLOUT_NNODES="${ROLLOUT_NNODES}",TRAINING_DATA_DIR="${TRAINING_DATA_DIR}",ENABLE_THINKING="${ENABLE_THINKING}",FORCE_THINKING="${FORCE_THINKING}",THINK_PREFIX_TOKEN="${THINK_PREFIX_TOKEN}",SEED="${SEED}",ROLLOUT_N="${ROLLOUT_N}",USE_GROUP_FILTERING="${USE_GROUP_FILTERING}",PROJECT_NAME="${PROJECT_NAME}",RUN_NAME="${RUN_NAME}",RUN_DIR="${RUN_DIR}",WORKING_DIR="${WORKING_DIR}",HOME="${HOME}",HF_HOME="${HF_HOME}",ENVIRONMENT_PATH="${ENVIRONMENT_PATH}",JOB_NAME="${JOB_NAME}" \
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
