#!/usr/bin/env bash
set -euo pipefail

# Run both:
# - cross-eval for your trained OE+MC models
# - eval for base Qwen model
#
# This script:
# - creates/uses a venv
# - installs Python deps (eval-only)
# - logs into W&B (if WANDB_API_KEY is set)
# - runs evals and logs JSON outputs as W&B artifacts (handled in Python)
#
# Usage (recommended):
#   export OE_MODEL="/path/or/hf_id"
#   export MC_MODEL="/path/or/hf_id"
#   export WANDB_PROJECT="gsm8k-evaluation"
#   export WANDB_ENTITY="your_entity"            # optional
#   export WANDB_API_KEY="..."                  # recommended
#   ./evals/oe_mc_eval_05_02_26/run_evals.sh
#
# Optional:
#   export NUM_SAMPLES=200
#   unset NUM_SAMPLES                       # full split (default)
#   export PROMPT_STYLE="train"                 # or "raw"
#   export NO_CHAT_TEMPLATE=0                   # 1 disables chat template
#   export NO_COT_PHRASE=0                      # 1 removes "Let's think step by step" in train prompts
#   export TORCH_DTYPE="float16"                # or bfloat16/float32
#   export ATTN_IMPL="flash_attention_2"        # eager|sdpa|flash_attention_2
#   export USE_FLASH_ATTN=1                     # attempts to install flash-attn (may fail depending on env)
#   export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
#   export WANDB_MODE="online"                  # online|offline|disabled
#   export SKIP_BASE=1                          # skip base model eval
#   export SKIP_CROSS=1                         # skip OE/MC cross-eval
#   export MULTI_GPU=1                          # use torchrun when >1 GPU is available
#   export NUM_PROCESSES=4                      # optional torchrun nproc override

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-evals}"
CACHE_BASE="${CACHE_BASE:-$REPO_ROOT/.cache}"

export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_BASE/wandb}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"
export VERL_RUN_DIR="${VERL_RUN_DIR:-$CACHE_BASE/verl}"
export WANDB_API_KEY="PUR_YOUR_API_KEY"
export SKIP_BASE=0
export OE_MODEL="tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_gsm8k:v0"
export MC_MODEL="tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_mc_gsm8k:v0"
export WANDB_PROJECT="gsm8k-evaluation"
export WANDB_ENTITY="tommaso-bendinelli-eth-zurich"
export USE_FLASH_ATTN=1
export ATTN_IMPL="flash_attention_2"

export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"

# Avoid HF hub rate limit warnings if you have a token.
# export HF_TOKEN="..."

OE_MODEL="${OE_MODEL:-}"
MC_MODEL="${MC_MODEL:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

NUM_SAMPLES="${NUM_SAMPLES:-}"
PROMPT_STYLE="${PROMPT_STYLE:-train}"
NO_CHAT_TEMPLATE="${NO_CHAT_TEMPLATE:-0}"
NO_COT_PHRASE="${NO_COT_PHRASE:-0}"
PARSE_METHODS="${PARSE_METHODS:-strict flexible}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

SKIP_BASE="${SKIP_BASE:-0}"
SKIP_CROSS="${SKIP_CROSS:-0}"
MULTI_GPU="${MULTI_GPU:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-}"

mkdir -p "$CACHE_BASE" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$VERL_RUN_DIR"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Venv: $VENV_DIR"
echo "[INFO] Cache base: $CACHE_BASE"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Installing eval dependencies (without torch)"
python -m pip install -U pip wheel setuptools
python -m pip install -U \
  transformers \
  accelerate \
  datasets \
  tqdm \
  numpy \
  wandb \
  ray \
  tensordict

echo "[INFO] WANDB_MODE=$WANDB_MODE"

echo "[INFO] If OE_MODEL/MC_MODEL are W&B artifacts, the eval will download them to: ${VERL_RUN_DIR}/models/"
echo "[INFO] Tip: export HF_TOKEN=... to avoid HF Hub unauthenticated rate limits."

if [[ "${USE_FLASH_ATTN:-0}" == "1" ]]; then
  if [[ -z "$ATTN_IMPL" ]]; then
    ATTN_IMPL="flash_attention_2"
  fi
  echo "[INFO] USE_FLASH_ATTN=1 -> attempting to install flash-attn (may require CUDA toolchain)"
  python -m pip install flash-attn==2.5.7 --no-build-isolation -v || {
    echo "[WARN] flash-attn install failed; continuing without it."
  }
fi

# If the user forces flash_attention_2 but flash-attn isn't installed, fall back.
if [[ "$ATTN_IMPL" == "flash_attention_2" ]]; then
  python - <<'PY' || {
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("flash_attn") is not None else 1)
PY
    echo "[WARN] ATTN_IMPL=flash_attention_2 but flash-attn is not installed. Falling back to ATTN_IMPL=sdpa."
    ATTN_IMPL="sdpa"
  }
fi

python - <<'PY'
import os
try:
  import torch
  print("[INFO] torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
except Exception as e:
  raise SystemExit("[ERROR] torch is not importable in this venv. Install a CUDA-enabled torch build for your system.") from e
PY

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

LAUNCHER=(python -u)
if [[ "$MULTI_GPU" == "1" && "$GPU_COUNT" -gt 1 ]]; then
  NPROC="${NUM_PROCESSES:-$GPU_COUNT}"
  if [[ "$NPROC" -gt "$GPU_COUNT" ]]; then
    NPROC="$GPU_COUNT"
  fi
  LAUNCHER=(torchrun --standalone --nproc_per_node "$NPROC")
  echo "[INFO] Multi-GPU enabled: $NPROC processes across $GPU_COUNT GPUs (batch_size per process: $BATCH_SIZE)"
else
  echo "[INFO] Single-process mode (batch_size: $BATCH_SIZE)"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] Logging into W&B (WANDB_API_KEY is set)"
  python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("[INFO] W&B login OK")
PY
else
  echo "[WARN] WANDB_API_KEY is not set. If you are not already logged in, W&B artifact logging may fail."
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$REPO_ROOT/evals/oe_mc_eval_05_02_26/outputs/$RUN_TAG"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --prompt_style "$PROMPT_STYLE"
  --batch_size "$BATCH_SIZE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --max_length "$MAX_LENGTH"
  --torch_dtype "$TORCH_DTYPE"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_mode "$WANDB_MODE"
)

if [[ -n "$NUM_SAMPLES" ]]; then
  if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_SAMPLES must be a positive integer when set. Unset it for full split."
    exit 1
  fi
  COMMON_ARGS+=(--num_samples "$NUM_SAMPLES")
fi

if [[ -n "$WANDB_RUN_GROUP" ]]; then
  COMMON_ARGS+=(--wandb_group "$WANDB_RUN_GROUP")
fi

if [[ -n "$ATTN_IMPL" ]]; then
  COMMON_ARGS+=(--attn_implementation "$ATTN_IMPL")
fi
if [[ "$NO_CHAT_TEMPLATE" == "1" ]]; then
  COMMON_ARGS+=(--no_chat_template)
fi
if [[ "$NO_COT_PHRASE" == "1" ]]; then
  COMMON_ARGS+=(--no_cot_phrase)
fi

# parse_methods is a multi-arg flag; split PARSE_METHODS safely
read -r -a PARSE_METHODS_ARR <<<"$PARSE_METHODS"
COMMON_ARGS+=(--parse_methods "${PARSE_METHODS_ARR[@]}")

cd "$REPO_ROOT"

if [[ "$SKIP_CROSS" != "1" ]]; then
  if [[ -z "$OE_MODEL" || -z "$MC_MODEL" ]]; then
    echo "[ERROR] OE_MODEL and MC_MODEL must be set unless SKIP_CROSS=1"
    exit 1
  fi

  echo "[INFO] Running cross-eval (OE+MC models)"
  "${LAUNCHER[@]}" "$REPO_ROOT/evals/oe_mc_eval_05_02_26/evaluate_oe_mc_models_dual_parse.py" \
    --oe_model "$OE_MODEL" \
    --mc_model "$MC_MODEL" \
    --out_json "$OUT_DIR/cross_eval_oe_mc.json" \
    --wandb_run_name "cross_eval_${RUN_TAG}" \
    "${COMMON_ARGS[@]}"
fi

if [[ "$SKIP_BASE" != "1" ]]; then
  echo "[INFO] Running base Qwen eval"
  "${LAUNCHER[@]}" "$REPO_ROOT/evals/oe_mc_eval_05_02_26/evaluate_base_qwen_dual_parse.py" \
    --model "$BASE_MODEL" \
    --out_json "$OUT_DIR/base_qwen.json" \
    --wandb_run_name "base_eval_${RUN_TAG}" \
    "${COMMON_ARGS[@]}"
fi

echo "[DONE] Outputs saved in: $OUT_DIR"
