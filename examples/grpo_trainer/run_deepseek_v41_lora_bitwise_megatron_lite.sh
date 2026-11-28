#!/usr/bin/env bash
# DeepSeek-V4.1-Flash LoRA GRPO/DAPO on verl + Megatron Lite (mLite) + vLLM with
# training/inference bitwise-identical log-probs, including 262,144-token context on 8 GPUs.
#
# Bitwise: the actor forward calls the rollout's own vLLM operators, so for every response
# token the FP32 log-prob bits of the vLLM rollout, the independent old-policy recompute and
# the grad-enabled actor are identical (not an epsilon threshold). Only the LoRA adapter is
# trained and published to vLLM each step; the FP4/FP8 release base stays frozen.
#
# Modes:
#   long262k  context 262,144 tokens (prompt 262,016 + response 128 by default), actor
#             EP8/CP8/PP1, rollout DEP4 x 2, actor and rollout strictly alternate on the GPUs
#             (exclusive rollout, native same-source reload of the rollout base); 2 prompts x n4.
#             The configuration is taken from the validated 262K run on the pinned commit. That run
#             used a validation driver that is not part of the public tree; this script goes through
#             verl.trainer.main_ppo, which has not been run at 262K. That run
#             was a capacity/on-policy gate: a synthetic repeated-text prefix, a seeded single-turn
#             agent, ignore_eos (exactly 128 response tokens) and a proxy reward, under a resident
#             driver whose third step only scored (two optimizer updates in three steps). Those
#             fixtures are not part of this example; reward, EOS and lengths come from your data,
#             and other response lengths or batch sizes are not covered by that run. The launcher
#             also selects vLLM's V2 model runner, which the validated run did not force.
#   rl        prompt 2048 / response 2048, 128 prompts x n8, actor EP8/CP1/PP1, rollout DEP4 x 2,
#             rollout base resident on GPU (only the KV cache is released between phases).
#             Its bitwise runs were measured on Megatron-LM a540b7db / edfc2db6 with the DAPO trainer
#             from verl-recipe; this mode on the pinned commit with verl.trainer.main_ppo is untested.
#             The pinned commit releases the CUDA cache after every micro-batch backward (later heads
#             make this optional), which lowers actor memory but roughly doubled the actor update
#             time where measured; the a540b7db step time does not apply here.
#
# Hardware: 2 nodes x 4 GB200 (aarch64, SM100), 8 GPUs in total. long262k additionally needs
#   about 850 GiB of host memory per node (allocate whole nodes, e.g. Slurm --mem=0).
#
# Image: docker/Dockerfile.ds41_lora_bitwise (vLLM, DeepGEMM and the runtime venv, all pinned).
#
# Sources (mounted at run time, not in the image):
#   VERL_ROOT      verl-project/verl at b9c513c44c0680822016b63e39249165f1fc6b94
#   MEGATRON_ROOT  ISEEKYAN/Megatron-LM at 915b94e097b51e3f7a11efe5a3279980c48dc1fb
#     git clone --branch ds41-lora-bitwise-256k-915b94e0 \
#       https://github.com/ISEEKYAN/Megatron-LM.git "${MEGATRON_ROOT}"
#
# Pinned versions:
#   Megatron-LM (mLite)  ISEEKYAN/Megatron-LM          915b94e097b51e3f7a11efe5a3279980c48dc1fb
#   vLLM                 ISEEKYAN/vllm v62365/lora-overlap  4d69bb528888017c7998b97972405c70076533f4
#                        (Python tree over the official fe8f2140e nightly wheel natives)
#   DeepGEMM             ISEEKYAN/DeepGEMM r1/mega-moe-lora 9f9f78ff7e638c6c365b82c0549d0dfd1e41ff28
#   verl                 verl-project/verl              b9c513c44c0680822016b63e39249165f1fc6b94
#   Base image           nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
#   Python packages      pinned in docker/Dockerfile.ds41_lora_bitwise (Torch 2.14.0+cu130,
#                        Transformer Engine 2.19.0 from the base image)
#
# Model: DeepSeek-V4.1-Flash native release (40 layers, 384 experts). LoRA rank 32 / alpha 32 on
#   layers 20-39, covering attention projections, shared experts and routed experts.
#
# Usage:
#   export VERL_ROOT=... MEGATRON_ROOT=... MODEL_PATH=... TRAIN_FILES=... VAL_FILES=...
#   export BASE_MANIFEST_SHA256=$(sha256sum /path/to/verified-base-manifest.json | awk '{print $1}')
#     The verified base manifest is a JSON file you write once for the release you load: the
#     release revision, the SHA-256 of config.json and the safetensors index, and for every loaded
#     weight file (including Engram tables and scale sidecars) its relative name, size and full
#     payload SHA-256 (download checksums may be reused), sorted by name and serialized
#     deterministically. Its digest binds the actor and the rollout to the same frozen base; the
#     runtime checks that both sides carry the same value, not that the files match it, so verify
#     the files before loading. Never reuse another release's digest.
#   bash examples/grpo_trainer/run_deepseek_v41_lora_bitwise_megatron_lite.sh --mode long262k [Hydra overrides...]
#   Start one container per node from the image, form a Ray cluster across both nodes
#   (ray start --head / ray start --address, --object-store-memory=8589934592 per node), then run
#   this script on the head node (RAY_ADDRESS points verl at the existing cluster). Mount VERL_ROOT
#   and MEGATRON_ROOT at the same paths on both nodes: the script forwards PYTHONPATH to the Ray
#   workers through runtime_env, since pre-started workers do not inherit the driver environment.
set -euo pipefail

MEGATRON_COMMIT=915b94e097b51e3f7a11efe5a3279980c48dc1fb

usage() {
  echo "usage: $0 --mode {long262k|rl} [Hydra overrides...]"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

MODE="${MODE:-}"
HYDRA_OVERRIDES=()
while (($# > 0)); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done
[[ "${MODE}" == long262k || "${MODE}" == rl ]] || die "set --mode long262k or --mode rl"

# --- Required sources and inputs ---
: "${VERL_ROOT:?mount verl and set VERL_ROOT}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to the pinned ISEEKYAN/Megatron-LM checkout}"
: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash native release directory}"
: "${TRAIN_FILES:?set TRAIN_FILES to a DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to a DAPO-format validation parquet}"
: "${BASE_MANIFEST_SHA256:?set BASE_MANIFEST_SHA256 to the frozen base identity}"
[[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"
head=$(git -C "${MEGATRON_ROOT}" rev-parse HEAD)
if [[ "${head}" != "${MEGATRON_COMMIT}" ]]; then
  [[ "${ALLOW_UNPINNED_MEGATRON:-0}" == 1 ]] || die "MEGATRON_ROOT is at ${head}, expected ${MEGATRON_COMMIT}"
  echo "warning: MEGATRON_ROOT is at ${head}, not the validated ${MEGATRON_COMMIT}" >&2
fi
LITE_ROOT="${MEGATRON_ROOT}/experimental/lite"
LAUNCHER="${LITE_ROOT}/examples/verl/scripts/run_deepseek_v41_dapo.sh"
[[ -f "${LAUNCHER}" ]] || die "missing ${LAUNCHER}"
export PYTHONPATH="${VERL_ROOT}:${LITE_ROOT}/examples/verl:${LITE_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"

# --- Common: 2 nodes x 4 GPUs, rollout DEP4 x 2, full recompute, chunked optimizer offload ---
export NNODES="${NNODES:-2}" NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export ACTOR_EP="${ACTOR_EP:-8}" ACTOR_PP="${ACTOR_PP:-1}" ROLLOUT_DP="${ROLLOUT_DP:-4}" ROLLOUT_EP="${ROLLOUT_EP:-4}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}" RECOMPUTE="${RECOMPUTE:-full}" ROLLOUT_SPEC=False
export PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=True GRAD_OFFLOAD=False OPTIMIZER_STATE_CHUNK_MB=2048
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}" ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export CAPTURE_SIZES='[1,2,4,8,16,24,32,48,64]' REPLAY_SIZES='[1,2,4,8,16]' MAX_CAPTURE_SIZE=64
# Fused MegaMoE LoRA and shrink overlap must reach every Ray worker (the launcher does not forward them).
export VLLM_MEGAMOE_LORA_FUSED=1 VLLM_LORA_OVERLAP_SHRINK=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
COMMON_ARGS=(
  '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MEGAMOE_LORA_FUSED="1"'
  '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LORA_OVERLAP_SHRINK="1"'
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"${PYTHONPATH}\""
)

MODE_ARGS=()
case "${MODE}" in
  long262k)
    export ACTOR_CP="${ACTOR_CP:-8}" ROLLOUT_SLEEP_LEVEL=2
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}" ROLLOUT_N="${ROLLOUT_N:-4}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-262016}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
    export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-0}"
    export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-262144}" LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-262144}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" SAVE_FREQ="${SAVE_FREQ:--1}" TEST_FREQ="${TEST_FREQ:--1}"
    MODE_ARGS=(
      "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
      "++actor_rollout_ref.actor.engine.exclusive_rollout=True"
      "++actor_rollout_ref.rollout.custom.ds41_lora_sleep_mode=weights_reload"
      "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=8589934592"
      "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256"
      "actor_rollout_ref.actor.use_dynamic_bsz=False"
      "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False"
      "++actor_rollout_ref.actor.engine.impl_cfg.attention_deterministic_backward=False"
      "++actor_rollout_ref.actor.engine.impl_cfg.head_chunk_size=1024"
      "data.dataloader_num_workers=0"
      # The validated 262K run scored with the naive manager; the overlong penalty is not applied.
      "reward.reward_manager.name=naive"
    )
    ;;
  rl)
    export ACTOR_CP="${ACTOR_CP:-1}" ROLLOUT_SLEEP_LEVEL=1
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}" ROLLOUT_N="${ROLLOUT_N:-8}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
    export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-8192}" LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-8192}"
    export ROLLOUT_FULL_DETERMINISM=False PROMPT_MODE="${PROMPT_MODE:-thinking}"
    export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"
    MODE_ARGS=(
      "++actor_rollout_ref.rollout.custom.ds41_lora_sleep_mode=kv_only"
      "++actor_rollout_ref.actor.engine.exclusive_rollout=False"
      "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648"
      "++actor_rollout_ref.actor.engine.impl_cfg.head_chunk_size=1024"
      # DAPO reward with the overlong penalty, as in the measured 25-step DAPO run.
      "reward.reward_manager.name=dapo"
      "++reward.reward_kwargs.max_resp_len=8192"
    )
    ;;
esac

# --- LoRA: rank 32 / alpha 32 on layers 20-39 with the targets the serving side exposes ---
mapfile -t LORA_ARGS < <(python3 - "${BASE_MANIFEST_SHA256}" <<'PY'
import json
import sys

from megatron.lite.model.deepseek_v41.vllm.lora import serving_targets
from megatron.lite.primitive.modules.lora import LoraConfig

base, layers, rank = sys.argv[1], 40, 32
int(base, 16)
assert len(base) == 64, "BASE_MANIFEST_SHA256 must be a SHA-256 hex digest"
targets = list(serving_targets(LoraConfig(rank=rank, alpha=rank, layers=(20, layers)), layers))
vllm = "actor_rollout_ref.rollout.engine_kwargs.vllm."
for key, value in {
    "++actor_rollout_ref.model.lora": dict(rank=rank, alpha=rank, merge=False, layers=[20, layers]),
    "+actor_rollout_ref.actor.engine.impl_cfg.fused_moe_lora": True,
    "+actor_rollout_ref.actor.engine.impl_cfg.lora": dict(rank=rank, alpha=rank, layers=[20, layers]),
    "++actor_rollout_ref.model.override_config.mlite_base_manifest_sha256": base,
    "++" + vllm + "hf_overrides.mlite_base_manifest_sha256": base,
    "+" + vllm + "lora_target_modules": targets,
    "+" + vllm + "lora_dtype": "bfloat16",
    "+" + vllm + "max_cpu_loras": 1,
}.items():
    if isinstance(value, dict):
        value = "{" + ",".join(k + ":" + json.dumps(v) for k, v in value.items()) + "}"
    else:
        value = json.dumps(value, separators=(",", ":"))
    print(key + "=" + value)
PY
)
[[ ${#LORA_ARGS[@]} == 8 ]] || die "LoRA override generation failed"

exec bash "${LAUNCHER}" "${COMMON_ARGS[@]}" "${LORA_ARGS[@]}" "${MODE_ARGS[@]}" "${HYDRA_OVERRIDES[@]}"
