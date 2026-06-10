#!/usr/bin/env bash
#
# Submit a multi-node VERL training job using the Kubernetes sandbox service for
# code reward evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
TRAINING_DATA_DIR=/iopsstor/scratch/cscs/${USER}/apertus_rl/data/code
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

KUBERNETES_SANDBOX_URL=https://sandbox-dev.swissai.svc.cscs.ch/harness-test
SANDBOX_REWARD_CONTINUOUS=false # default is binary reward

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
  TRAIN_JOB_NAME="${JOB_NAME}"

  mkdir -p "${RUN_DIR}"
}

resolve_run_name_and_dir
clear_inherited_pyxis_options
KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL%/}"

log "\nCheck Kubernetes sandbox"
if ! curl -fsS --max-time 10 "${KUBERNETES_SANDBOX_URL}/" >/dev/null; then
  echo "Kubernetes sandbox is not reachable at ${KUBERNETES_SANDBOX_URL}" >&2
      exit 1
    fi
log "  -> reachable at ${KUBERNETES_SANDBOX_URL}"

# ==========================================
# Submit VERL Job
# ==========================================
log "\nSubmit multi-node async VERL training"
log "  -> job-name=${TRAIN_JOB_NAME} time=${SLURM_TIME} train=${TRAIN_NNODES} rollout=${ROLLOUT_NNODES} total=${NNODES}"
log "  -> config=${CONFIG_NAME} model=${MODEL_NAME_OR_PATH}"
log "  -> data=${TRAINING_DATA_DIR} seed=${SEED} rollout_n=${ROLLOUT_N}"
log "  -> group_filtering=${USE_GROUP_FILTERING} enable_thinking=${ENABLE_THINKING} force_thinking=${FORCE_THINKING}"
log "  -> output=${RUN_DIR}"
log "  -> kubernetes_sandbox=${KUBERNETES_SANDBOX_URL} continuous=${SANDBOX_REWARD_CONTINUOUS}"

TRAIN_SUBMIT="$(sbatch \
  --job-name="${TRAIN_JOB_NAME}" \
  --nodes="${NNODES}" \
  --time="${SLURM_TIME}" \
  --output="${RUN_DIR}/multinode_async_sandbox_%j.out" \
  --error="${RUN_DIR}/multinode_async_sandbox_%j.err" \
  --export=ALL,KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL}",SCHEDULER_URL="${KUBERNETES_SANDBOX_URL}",SANDBOX_REWARD_CONTINUOUS="${SANDBOX_REWARD_CONTINUOUS}",MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}",TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH}",CONFIG_NAME="${CONFIG_NAME}",NNODES="${NNODES}",TRAIN_NNODES="${TRAIN_NNODES}",ROLLOUT_NNODES="${ROLLOUT_NNODES}",TRAINING_DATA_DIR="${TRAINING_DATA_DIR}",ENABLE_THINKING="${ENABLE_THINKING}",FORCE_THINKING="${FORCE_THINKING}",THINK_PREFIX_TOKEN="${THINK_PREFIX_TOKEN}",SEED="${SEED}",ROLLOUT_N="${ROLLOUT_N}",USE_GROUP_FILTERING="${USE_GROUP_FILTERING}",PROJECT_NAME="${PROJECT_NAME}",RUN_NAME="${RUN_NAME}",RUN_DIR="${RUN_DIR}",WORKING_DIR="${WORKING_DIR}",HOME="${HOME}",HF_HOME="${HF_HOME}",ENVIRONMENT_PATH="${ENVIRONMENT_PATH}",JOB_NAME="${JOB_NAME}" \
  "${TRAIN_SCRIPT}" "$@")"
TRAIN_ID="$(awk '{print $NF}' <<<"${TRAIN_SUBMIT}")"
[[ "${TRAIN_ID}" =~ ^[0-9]+$ ]] || { echo "Failed to parse training job id: ${TRAIN_SUBMIT}" >&2; exit 1; }
log "  -> Training JobID: ${TRAIN_ID}"

# ==========================================
echo
echo "Monitor:"
  echo "  squeue -j ${TRAIN_ID}"
echo "  tail -f ${RUN_DIR}/multinode_async_sandbox_${TRAIN_ID}.out"
