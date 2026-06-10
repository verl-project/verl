#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -xeuo pipefail

clear_inherited_pyxis_options() {
  local name
  while IFS='=' read -r name _; do
    case "${name}" in
      SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_*) unset "${name}" ;;
    esac
  done < <(env)
}

clear_inherited_pyxis_options

ulimit -c 0

USERNAME="$(whoami)"
WORKING_DIR="${WORKING_DIR:-/iopsstor/scratch/cscs/${USERNAME}/apertus_rl_async}"
HF_HOME="${HF_HOME:-/capstor/scratch/cscs/${USERNAME}/huggingface}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HF_HOME}/hub}"

cd "${WORKING_DIR}"

PROJECT_NAME="${PROJECT_NAME:-apertus-rl-tests}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-swiss-ai/Apertus-8B-Instruct-2509}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"
CONFIG_NAME="${CONFIG_NAME:-async}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
ROLLOUT_N="${ROLLOUT_N:-8}"
SEED="${SEED:-85}"
USE_GROUP_FILTERING="${USE_GROUP_FILTERING:-true}"
FORCE_THINKING="${FORCE_THINKING:-false}"
THINK_PREFIX_TOKEN="${THINK_PREFIX_TOKEN:-<think>}"
WANDB_ENTITY="${WANDB_ENTITY:-apertus}"
NO_CODE="${NO_CODE:-false}"

TRAIN_NNODES="${TRAIN_NNODES:-1}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
NNODES="${NNODES:-$((TRAIN_NNODES + ROLLOUT_NNODES))}"
if [ "${NNODES}" -ne "$((TRAIN_NNODES + ROLLOUT_NNODES))" ]; then
  echo "NNODES=${NNODES} must equal TRAIN_NNODES + ROLLOUT_NNODES." >&2
  exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"
ENVIRONMENT_PATH="${ENVIRONMENT_PATH:-/users/${USERNAME}/.edf/async.toml}"
SRUN_ENV=(--environment="${ENVIRONMENT_PATH}" --container-writable)

# Sandbox and code reward settings
DEFAULT_MEMORY_LIMIT_MB="${DEFAULT_MEMORY_LIMIT_MB:-512}"
CODEGYM_REWARD_MAX_CONCURRENT="${CODEGYM_REWARD_MAX_CONCURRENT:-64}"
CODEGYM_REWARD_CONTINUOUS="${CODEGYM_REWARD_CONTINUOUS:-false}"

# Extra packages needed for scoring are installed once into a target dir 
# so every node can use them and avoid repeated installations
# TODO: for production run, think about a better solution!
PY_DEPS_ROOT="${PY_DEPS_ROOT:-/iopsstor/scratch/cscs/${USERNAME}/python_deps}"
PY_DEPS_DIR="${PY_DEPS_DIR:-${PY_DEPS_ROOT}/apertus_reward_py312}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-/iopsstor/scratch/cscs/${USERNAME}/nltk_data}"

DEPS_STAMP_DIR="${PY_DEPS_ROOT}/.stamps"
DEPS_STAMP="${DEPS_STAMP_DIR}/apertus_reward_py312.ready"
DEPS_LOCK="${DEPS_STAMP_DIR}/apertus_reward_py312.lock"

EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-immutabledict nltk langdetect absl-py emoji syllapy antlr4-python3-runtime==4.9.3 requests math_verify}"
REASONING_GYM_DIR="${REASONING_GYM_DIR:-}"


DATA_DIR="${TRAINING_DATA_DIR:-${WORKING_DIR}/data/apertus_demo_rl}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
TEST_FILE="${DATA_DIR}/val.parquet"
APERTUS_DIR="${APERTUS_DIR:-${WORKING_DIR}/apertus}"
# display_answers is a stateless @function_tool, so it is loaded through
# function_tool_path. tool_config_path is only for BaseTool subclasses.
APERTUS_FUNCTION_TOOL_PATH="${APERTUS_FUNCTION_TOOL_PATH:-${APERTUS_DIR}/display_answer_tools.py}"
OUTPUT_ROOT="${WORKING_DIR}/outputs/${PROJECT_NAME}"
RUN_NAME="${RUN_NAME:-}"
RUN_DIR="${RUN_DIR:-}"

resolve_run_name() {
  local model_tag
  local date_time
  local filtering_tag
  if [ -n "${RUN_DIR}" ]; then
    if [ -z "${RUN_NAME}" ]; then
      RUN_NAME="$(basename "${RUN_DIR}")"
    fi
    mkdir -p "${RUN_DIR}"
    return
  fi
  model_tag="$(basename "${MODEL_NAME_OR_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  date_time="$(date +'%Y%m%dT%H%M%S')"
  filtering_tag=""
  if [ "${USE_GROUP_FILTERING}" = "true" ]; then
    filtering_tag="dapo__"
  fi

  if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="async_sandbox__${filtering_tag}${model_tag}__train${TRAIN_NNODES}x${TRAIN_GPUS_PER_NODE}__rollout${ROLLOUT_NNODES}x${ROLLOUT_GPUS_PER_NODE}__seed${SEED}__${date_time}"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"
}

resolve_hf_path() {
  local input_path="$1"
  local hub_ref
  local commit
  if [ -d "${input_path}" ]; then
    printf '%s' "${input_path}"
    return
  fi
  hub_ref="${HF_HUB_CACHE_DIR}/models--${input_path//\//--}/refs/main"
  if [ -f "${hub_ref}" ]; then
    commit="$(cat "${hub_ref}")"
    printf '%s' "${HF_HUB_CACHE_DIR}/models--${input_path//\//--}/snapshots/${commit}"
    return
  fi

  printf '%s' "${input_path}"
}

build_overrides() {
  overrides=(
    "--config-name=${CONFIG_NAME}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TEST_FILE}']"
    "data.seed=${SEED}"
    "+data.force_thinking_prefix=${FORCE_THINKING}"
    "+data.thinking_prefix_token='${THINK_PREFIX_TOKEN}'"
    "actor_rollout_ref.model.path=${RESOLVED_MODEL}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.multi_turn.enable=True"
    "actor_rollout_ref.rollout.multi_turn.format=apertus2509"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1"
    "actor_rollout_ref.rollout.multi_turn.terminal_tool_names=[display_answers]"
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=null"
    "actor_rollout_ref.rollout.multi_turn.function_tool_path=${APERTUS_FUNCTION_TOOL_PATH}"
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
    "data.tool_config_path=null"
    "data.function_tool_path=${APERTUS_FUNCTION_TOOL_PATH}"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=8192"
    "actor_rollout_ref.rollout.max_num_seqs=256"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8"
    "actor_rollout_ref.rollout.skip_tokenizer_init=False"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=llguidance"
    "rollout.nnodes=${ROLLOUT_NNODES}"
    "rollout.n_gpus_per_node=${ROLLOUT_GPUS_PER_NODE}"
    "reward.sandbox_fusion.memory_limit_mb=${DEFAULT_MEMORY_LIMIT_MB}"
    "reward.sandbox_fusion.max_concurrent=${CODEGYM_REWARD_MAX_CONCURRENT}"
    "+reward.sandbox_fusion.continuous=${CODEGYM_REWARD_CONTINUOUS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.nnodes=${TRAIN_NNODES}"
    "trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE}"
    "+ray_kwargs.ray_init.address=${ip_head}"
    "ray_kwargs.ray_init.num_cpus=null"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HOME=${HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NLTK_DATA=${NLTK_DATA_DIR}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MATH_VERIFY_PYTHONPATH=${PY_DEPS_DIR}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY=${WANDB_ENTITY}"
  )

  if [ -n "${SCHEDULER_URL:-}" ]; then
    overrides+=(
      "reward.sandbox_fusion.url='${SCHEDULER_URL}'"
      "+ray_kwargs.ray_init.runtime_env.env_vars.SCHEDULER_URL='${SCHEDULER_URL}'"
    )
  fi

  if [ "${NO_CODE}" = "true" ]; then
    overrides+=("+data.exclude_abilities=[code]")
  fi

  if [ -n "${RESOLVED_TOKENIZER}" ]; then
    overrides+=("actor_rollout_ref.model.tokenizer_path=${RESOLVED_TOKENIZER}")
  fi

  if [ "${USE_GROUP_FILTERING}" = "true" ]; then
    overrides+=(
      "algorithm.filter_groups.enable=true"
      "algorithm.filter_groups.metric=acc"
      "algorithm.filter_groups.min=0.0"
      "algorithm.filter_groups.max=1.0"
    )
  else
    overrides+=(
      "algorithm.filter_groups.enable=false"
      "algorithm.filter_groups.metric=null"
    )
  fi
}

cleanup_ray() {
  set +e
  srun --overlap "${SRUN_ENV[@]}" --nodes="${NNODES}" --ntasks="${NNODES}" \
    bash -lc "cd '${WORKING_DIR}' && ray stop --force"
}
trap cleanup_ray EXIT

wait_for_ray_cluster() {
  local attempt
  for attempt in {1..60}; do
    if srun --overlap "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
      bash -lc 'python3 -c '"'"'import os, ray
ray.init(address=os.environ["ip_head"], ignore_reinit_error=True)
alive = sum(1 for node in ray.nodes() if node.get("Alive"))
expected = int(os.environ["NNODES"])
print(f"Ray nodes alive: {alive}/{expected}")
ray.shutdown()
raise SystemExit(0 if alive >= expected else 1)'"'"''; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for Ray workers to join ${ip_head}" >&2
  return 1
}

release_deps_lock() {
  rmdir "${DEPS_LOCK}" 2>/dev/null || true
  trap cleanup_ray EXIT
}

prepare_shared_python_deps() {
  srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" bash -lc '
    set -euo pipefail
    release_deps_lock() {
      rmdir "${DEPS_LOCK}" 2>/dev/null || true
    }
    trap release_deps_lock EXIT

    if [ -f "${DEPS_STAMP}" ]; then
      exit 0
    fi

    mkdir -p "${PY_DEPS_DIR}" "${NLTK_DATA_DIR}" "${DEPS_STAMP_DIR}"

    while ! mkdir "${DEPS_LOCK}" 2>/dev/null; do
      if [ -f "${DEPS_STAMP}" ]; then
        exit 0
      fi
      echo "Waiting for shared Python deps to be prepared by another process..."
      sleep 10
    done

    if [ -f "${DEPS_STAMP}" ]; then
      exit 0
    fi

    python3 -m pip install --no-cache-dir --upgrade --target "${PY_DEPS_DIR}" ${EXTRA_PIP_PACKAGES}
    if [ -n "${REASONING_GYM_DIR}" ]; then
      if [ ! -d "${REASONING_GYM_DIR}" ]; then
        echo "REASONING_GYM_DIR=${REASONING_GYM_DIR} does not exist. Clone https://github.com/EduardDurech/r-gym.git and check out the translate branch there, or set REASONING_GYM_DIR=\"\" to install reasoning-gym from PyPI." >&2
        exit 1
      fi
      python3 -m pip install --no-cache-dir --upgrade --force-reinstall --target "${PY_DEPS_DIR}" "${REASONING_GYM_DIR}"
    else
      python3 -m pip install --no-cache-dir --upgrade --force-reinstall --target "${PY_DEPS_DIR}" reasoning-gym
    fi
    python3 -c "import nltk; [nltk.download(pkg, download_dir=\"${NLTK_DATA_DIR}\", quiet=True) for pkg in (\"punkt\", \"punkt_tab\", \"stopwords\", \"averaged_perceptron_tagger_eng\")]"
    python3 -c "import sys; sys.path[:0] = [\"${PY_DEPS_DIR}\", \"${WORKING_DIR}\"]; import absl, emoji, immutabledict, langdetect, math_verify, nltk, reasoning_gym, requests, syllapy, verl; print(\"Using shared deps from\", \"${PY_DEPS_DIR}\"); print(\"Using verl from\", verl.__file__)"
    touch "${DEPS_STAMP}"
  '
}

resolve_run_name
RESOLVED_MODEL="$(resolve_hf_path "${MODEL_NAME_OR_PATH}")"
RESOLVED_TOKENIZER=""
if [ -n "${TOKENIZER_NAME_OR_PATH}" ]; then
  RESOLVED_TOKENIZER="$(resolve_hf_path "${TOKENIZER_NAME_OR_PATH}")"
fi

# Shared HuggingFace cache prevents repeated downloads across runs/workers.
export HOME
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export NLTK_DATA="${NLTK_DATA_DIR}"
export PYTHONPATH="${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
export MATH_VERIFY_PYTHONPATH="${PY_DEPS_DIR}"
export ENVIRONMENT_PATH
export PY_DEPS_ROOT
export PY_DEPS_DIR
export NLTK_DATA_DIR
export REASONING_GYM_DIR
export DEPS_STAMP_DIR
export DEPS_STAMP
export DEPS_LOCK
export EXTRA_PIP_PACKAGES
export SCHEDULER_URL
export DEFAULT_MEMORY_LIMIT_MB
export CODEGYM_REWARD_MAX_CONCURRENT
export PYTHONUNBUFFERED="1"
# export RAY_DEDUP_LOGS="0"
export NNODES

export WANDB_RUN_GROUP="${JOB_NAME:-${RUN_NAME}}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_ENTITY

nodes_array=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [ "${#nodes_array[@]}" -lt "${NNODES}" ]; then
  echo "Expected ${NNODES} Slurm nodes, found ${#nodes_array[@]} in ${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

# ====================
# Start Ray cluster
# ====================
head_node="${nodes_array[0]}"
prepare_shared_python_deps
srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "python3 -c \"import sys; sys.path[:0] = ['${PY_DEPS_DIR}', '${WORKING_DIR}']; import reasoning_gym, requests; print('Using requests from', requests.__file__)\""

head_node_ip="$(srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && hostname --ip-address")"

if [[ "${head_node_ip}" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"${head_node_ip}"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using IPV4 address ${head_node_ip}"
fi

port=6379
ip_head=${head_node_ip}:${port}
export ip_head
export RAY_ADDRESS="${ip_head}"
export MASTER_ADDR="${head_node_ip}"
export MASTER_PORT="${port}"

echo "Starting Ray head at ${head_node} (${ip_head})"
srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && ray start --head --node-ip-address='${head_node_ip}' --port='${port}' --dashboard-host=0.0.0.0 --num-cpus '${CPUS_PER_TASK}' --num-gpus '${GPUS_PER_NODE}' --block" &

sleep 10

worker_num=$((NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i="${nodes_array[$i]}"
  echo "Starting Ray worker ${i} at ${node_i}"
  srun "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${node_i}" \
    bash -lc "cd '${WORKING_DIR}' && ray start --address '${ip_head}' --num-cpus '${CPUS_PER_TASK}' --num-gpus '${GPUS_PER_NODE}' --block" &
  sleep 5
done

wait_for_ray_cluster
build_overrides

srun --overlap "${SRUN_ENV[@]}" --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && HYDRA_FULL_ERROR=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \"\$@\"" -- \
  "${overrides[@]}" \
  "$@"
