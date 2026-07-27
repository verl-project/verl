#!/usr/bin/env bash
#
# Run one TMEM LoCoMo seed as two independent question shards. Each GPU hosts
# both a PEFT trainer and an SGLang+DFlash engine.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DRAFT_OPD_ROOT="${DRAFT_OPD_ROOT:-${REPO_ROOT}/../Draft-OPD}"
CONDA_ENV="${CONDA_ENV:-${DRAFT_OPD_ROOT}/.conda/draft-opd}"
PYTHON_BIN="${CONDA_ENV}/bin/python"

SEED="${1:-${SEED:-1}}"
DATA_PATH="${2:-${DATA_PATH:-/tmp/locomo10.json}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-${REPO_ROOT}/outputs/tmem_locomo_qwen3_4b_dflash_seed_${SEED}}}"
MODEL_PATH="${MODEL_PATH:-${DRAFT_OPD_ROOT}/models/Qwen3-4B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${DRAFT_OPD_ROOT}/models/Qwen3-4B-Thinking-Draft-OPD}"
GPU_IDS="${GPU_IDS:-4,5}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.60}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-12}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
CUDA_TOOLKIT_ROOT="${TMEM_CUDA_HOME:-/usr/local/cuda-12.2}"

for required_path in \
  "${PYTHON_BIN}" \
  "${DATA_PATH}" \
  "${MODEL_PATH}" \
  "${DRAFT_MODEL_PATH}" \
  "${CUDA_TOOLKIT_ROOT}/bin/nvcc"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

IFS=, read -r GPU_A GPU_B GPU_EXTRA <<< "${GPU_IDS}"
if [[ -z "${GPU_A}" || -z "${GPU_B}" || -n "${GPU_EXTRA:-}" ]]; then
  echo "GPU_IDS must contain two comma-separated physical GPU IDs; got: ${GPU_IDS}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-12}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${CONDA_ENV}/bin:${PATH}"

CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/usr/local/cuda-12.8/compat}"
CONDA_CUDA_RUNTIME="${CONDA_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if [[ -d "${CUDA_COMPAT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${CONDA_CUDA_RUNTIME}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${REPO_ROOT}"
echo "Seed ${SEED}: sharding independent questions across physical GPUs ${GPU_A} and ${GPU_B}"
echo "Each GPU runs both its own PEFT trainer and SGLang+DFlash engine."
echo "Output: ${OUTPUT_DIR}"

SHARD_A_DIR="${OUTPUT_DIR}/shard_a"
SHARD_B_DIR="${OUTPUT_DIR}/shard_b"
mkdir -p "${SHARD_A_DIR}" "${SHARD_B_DIR}"

# These official LoCoMo-10 conversations contain 999 and 987 questions,
# respectively. Since every question resets LoRA B, the shards are independent.
SHARD_A_IDS=(conv-26 conv-30 conv-41 conv-42 conv-43)
SHARD_B_IDS=(conv-44 conv-47 conv-48 conv-49 conv-50)

run_shard() {
  local gpu_id="$1"
  local shard_dir="$2"
  shift 2
  local sample_args=()
  local sample_id
  for sample_id in "$@"; do
    sample_args+=(--sample-id "${sample_id}")
  done
  local optional_args=()
  if [[ -n "${MAX_QUESTIONS}" ]]; then
    optional_args+=(--max-questions "${MAX_QUESTIONS}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m examples.tmem.run_locomo \
    --model "${MODEL_PATH}" \
    --dflash-draft-model "${DRAFT_MODEL_PATH}" \
    --rollout-backend dflash \
    --data "${DATA_PATH}" \
    --output-dir "${shard_dir}" \
    --trainer-device cuda:0 \
    --rollout-device cuda:0 \
    --sglang-mem-fraction "${SGLANG_MEM_FRACTION}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE}" \
    --max-extraction-tokens 1024 \
    --seeds "${SEED}" \
    --resume \
    "${sample_args[@]}" \
    "${optional_args[@]}"
}

run_shard "${GPU_A}" "${SHARD_A_DIR}" "${SHARD_A_IDS[@]}" \
  > >(sed -u 's/^/[shard-a] /' | tee "${OUTPUT_DIR}/shard_a.log") 2>&1 &
PID_A=$!
run_shard "${GPU_B}" "${SHARD_B_DIR}" "${SHARD_B_IDS[@]}" \
  > >(sed -u 's/^/[shard-b] /' | tee "${OUTPUT_DIR}/shard_b.log") 2>&1 &
PID_B=$!

terminate_shards() {
  kill "${PID_A}" "${PID_B}" 2>/dev/null || true
}
trap terminate_shards INT TERM

set +e
wait "${PID_A}"
STATUS_A=$?
wait "${PID_B}"
STATUS_B=$?
set -e
trap - INT TERM

if ((STATUS_A != 0 || STATUS_B != 0)); then
  echo "Shard failure: shard-a=${STATUS_A}, shard-b=${STATUS_B}. Re-run the same command to resume." >&2
  exit 1
fi

if [[ -n "${MAX_QUESTIONS}" ]]; then
  echo "Smoke shards completed; skipping full-dataset merge because MAX_QUESTIONS is set."
  exit 0
fi

"${PYTHON_BIN}" -m examples.tmem.merge_shards \
  --data "${DATA_PATH}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${SHARD_A_DIR}" \
  "${SHARD_B_DIR}"
