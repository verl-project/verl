#!/usr/bin/env bash
#
# Run one TMEM LoCoMo seed with PEFT training and SGLang+DFlash rollout on
# separate GPUs. Override any path or runtime setting through the environment.

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
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.72}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-25}"
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

if [[ "${GPU_IDS}" != *,* ]]; then
  echo "GPU_IDS must contain two comma-separated physical GPU IDs; got: ${GPU_IDS}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
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
echo "Seed ${SEED}: trainer=physical GPU ${GPU_IDS%%,*}; SGLang+DFlash=physical GPU ${GPU_IDS#*,}"
echo "Output: ${OUTPUT_DIR}"

RUN_ARGS=(
  --model "${MODEL_PATH}"
  --dflash-draft-model "${DRAFT_MODEL_PATH}"
  --rollout-backend dflash
  --data "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --trainer-device cuda:0
  --rollout-device cuda:1
  --sglang-mem-fraction "${SGLANG_MEM_FRACTION}"
  --generation-batch-size "${GENERATION_BATCH_SIZE}"
  --max-extraction-tokens 1024
  --seeds "${SEED}"
  --resume
)
if [[ -n "${MAX_QUESTIONS}" ]]; then
  RUN_ARGS+=(--max-questions "${MAX_QUESTIONS}")
fi

exec "${PYTHON_BIN}" -m examples.tmem.run_locomo "${RUN_ARGS[@]}"
