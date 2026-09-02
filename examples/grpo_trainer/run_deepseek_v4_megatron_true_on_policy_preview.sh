#!/usr/bin/env bash
# DeepSeek-V4 true-on-policy preview for VERL.
# aligned/quick_alignment_test use batch-invariant vLLM for exact probabilities;
# baseline-r3 keeps the ordinary mLite actor and routing replay for comparison.
# Modes: quick_alignment_test (1x4, four layers, two steps), aligned,
#        baseline-r3.
# Hardware: h100 (8x8, PP4/CP2/EP16, rollout EP16);
#           gb200 (8x4, PP1/CP2/EP32, rollout EP8).
# Images build on `Dockerfile.dsv4_true_on_policy_preview`:
#   docker://iseekyan/verl:ds4_vllm_align.preview-arm64
#   docker://iseekyan/verl:ds4_vllm_align.preview-amd64
# Sources:
#   VERL_ROOT: verl-project/verl main checkout
#   MEGATRON_ROOT: pinned Megatron preview checkout created with:
#     git clone --branch ds4_vllm_align_preview \
#       https://github.com/ISEEKYAN/Megatron-LM.git "${MEGATRON_ROOT}"
#     git -C "${MEGATRON_ROOT}" checkout \
#       6566f0ad8c5f07d52b8960b13c77f9ae04172110
set -euo pipefail

# Required env: VERL_ROOT, MEGATRON_ROOT, MODEL_PATH, TRAIN_FILES, VAL_FILES.
# Optional env: training steps, batch size, lengths, output paths, WANDB_API_KEY,
# WANDB_ENTITY, WANDB_MODE, WANDB_BASE_URL, and Hydra overrides.
SEED="${SEED:-42}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
PROJECT_NAME="${PROJECT_NAME:-verl-ds4-v4-preview}"
ACTOR_OPTIMIZER="${ACTOR_OPTIMIZER:-dist_opt}"

VLLM_BATCH_INVARIANT_KERNEL_LIB="${VLLM_BATCH_INVARIANT_KERNEL_LIB:-/opt/ds4/kernels/_vllm_batch_invariant_C.so}"
DS4_BI_TOPK_LIB="${DS4_BI_TOPK_LIB:-/opt/ds4/kernels/ds4_bi_topk.so}"
export VLLM_BATCH_INVARIANT_KERNEL_LIB DS4_BI_TOPK_LIB

usage() {
  echo "usage: $0 --hardware {h100|gb200} --mode {quick_alignment_test|aligned|baseline-r3} [Hydra overrides...]"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

MODE="${MODE:-}"
HARDWARE="${HARDWARE:-}"
HYDRA_OVERRIDES=()
while (( $# > 0 )); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --hardware)
      [[ $# -ge 2 ]] || die "--hardware requires a value"
      HARDWARE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      HYDRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

[[ -n "${MODE}" ]] ||
  die "set --mode quick_alignment_test, aligned, or baseline-r3"
[[ -n "${HARDWARE}" ]] || die "set --hardware h100 or --hardware gb200"
[[ "${HARDWARE}" == h100 || "${HARDWARE}" == gb200 ]] || die "unknown hardware: ${HARDWARE}"

# --- Required repositories and inputs ---
: "${VERL_ROOT:?mount VERL and set VERL_ROOT}"
: "${MEGATRON_ROOT:?mount pinned ISEEKYAN/Megatron-LM ds4_vllm_align_preview and set MEGATRON_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH to a DS4 config/tokenizer directory or checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to DAPO-format validation parquet}"
MLITE_EXAMPLE_ROOT="${MEGATRON_ROOT}/experimental/lite/examples/verl"
[[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"

# --- Mode presets and alignment behavior ---
case "${MODE}" in
  quick_alignment_test)
    EXACT_ALIGNMENT=1
    : "${TOTAL_TRAINING_STEPS:=2}"
    : "${TRAIN_BATCH_SIZE:=4}"
    : "${PPO_MINI_BATCH_SIZE:=4}"
    : "${OVERLONG_BUFFER_LEN:=512}"
    : "${ROLLOUT_N:=1}"
    : "${MAX_RESPONSE_LENGTH:=2048}"
    : "${ROLLOUT_MAX_NUM_SEQS:=4}"
    : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.55}"
    ;;
  aligned|baseline-r3)
    [[ "${MODE}" == aligned ]] && EXACT_ALIGNMENT=1 || EXACT_ALIGNMENT=0
    : "${TOTAL_TRAINING_STEPS:=20}"
    : "${TRAIN_BATCH_SIZE:=32}"
    : "${PPO_MINI_BATCH_SIZE:=32}"
    : "${OVERLONG_BUFFER_LEN:=4096}"
    : "${ROLLOUT_N:=8}"
    : "${MAX_RESPONSE_LENGTH:=8192}"
    : "${ROLLOUT_MAX_NUM_SEQS:=32}"
    : "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.65}"
    ;;
  *)
    die "unknown mode '${MODE}'"
    ;;
esac

case "${ACTOR_OPTIMIZER}" in
  dist_opt|fsdp2) ;;
  *) die "ACTOR_OPTIMIZER must be dist_opt or fsdp2, got '${ACTOR_OPTIMIZER}'" ;;
esac

if [[ -n "${TRAINER_LOGGERS:-}" ]]; then
  :
elif [[ "${WANDB_MODE:-}" == disabled ]]; then
  TRAINER_LOGGERS='[console,file]'
elif [[ -v WANDB_API_KEY || -v WANDB_ENTITY || -v WANDB_MODE ]]; then
  TRAINER_LOGGERS='[console,file,wandb]'
else
  TRAINER_LOGGERS='[console,file]'
fi

# Alignment behavior belongs to the mode, not to the hardware profile.
MODE_ARGS=()
if [[ "${EXACT_ALIGNMENT}" == 1 ]]; then
  export VLLM_BATCH_INVARIANT=1
  export VLLM_DS4_DECODE_KERNEL=sparse
  export VERL_FULL_DETERMINISM=1
  MODE_ARGS=(
    actor_rollout_ref.actor.engine.impl=vllm
    +actor_rollout_ref.actor.engine.seed="${SEED}"
    +actor_rollout_ref.actor.engine.full_determinism=True
    actor_rollout_ref.actor.engine.attention_backend_override=null
    actor_rollout_ref.rollout.full_determinism=True
    actor_rollout_ref.rollout.seed="${SEED}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=deepep_low_latency
    +actor_rollout_ref.rollout.engine_kwargs.vllm.linear_backend=deep_gemm
  )
else
  export VLLM_BATCH_INVARIANT=0
  export VLLM_DS4_DECODE_KERNEL=paged
  export VERL_FULL_DETERMINISM=0
  MODE_ARGS=(
    actor_rollout_ref.actor.engine.attention_backend_override=fused
    +actor_rollout_ref.actor.engine.impl_cfg.use_deepep=True
    actor_rollout_ref.actor.engine.router_replay_mode=R3
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
  )
fi

OPTIMIZER_ARGS=(
  +actor_rollout_ref.actor.engine.impl_cfg.optimizer="${ACTOR_OPTIMIZER}"
  actor_rollout_ref.actor.engine.param_offload=True
  actor_rollout_ref.actor.engine.optimizer_offload=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True
)

VLLM_WORKER_EXTENSION="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
if [[ "${HARDWARE}" == h100 ]]; then
  VLLM_WORKER_EXTENSION="verl_mlite.rollout.vllm_worker.MLiteVLLMColocateWorkerExtension"
fi

# --- Validated hardware profiles ---
if [[ "${MODE}" == quick_alignment_test ]]; then
  : "${NNODES:=1}"
  : "${NGPUS_PER_NODE:=4}"
  : "${ACTOR_PP:=1}"
  : "${ACTOR_CP:=1}"
  : "${ACTOR_EP:=4}"
  : "${ROLLOUT_DP:=4}"
  : "${ROLLOUT_EP:=4}"
  : "${ROLLOUT_AGENT_WORKERS:=4}"
else
  case "${HARDWARE}" in
    h100)
      : "${NNODES:=8}"
      : "${NGPUS_PER_NODE:=8}"
      : "${ACTOR_PP:=4}"
      : "${ACTOR_CP:=2}"
      : "${ACTOR_EP:=16}"
      : "${ROLLOUT_DP:=16}"
      : "${ROLLOUT_EP:=16}"
      : "${ROLLOUT_AGENT_WORKERS:=64}"
      ;;
    gb200)
      : "${NNODES:=8}"
      : "${NGPUS_PER_NODE:=4}"
      : "${ACTOR_PP:=1}"
      : "${ACTOR_CP:=2}"
      : "${ACTOR_EP:=32}"
      : "${ROLLOUT_DP:=8}"
      : "${ROLLOUT_EP:=8}"
      : "${ROLLOUT_AGENT_WORKERS:=32}"
      ;;
  esac
fi

runtime_config_root="$(mktemp -d /tmp/ds4-v4-preview-config.XXXXXX)"
trap 'rm -rf "${runtime_config_root}"' EXIT
mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
printf '%s\n' \
  '# @package _global_' \
  'model_engine: mlite' \
  >"${runtime_config_root}/model_engine/mlite.yaml"
printf '%s\n' \
  '_target_: verl.workers.config.CriticConfig' \
  'enable: false' \
  'strategy: mlite' \
  >"${runtime_config_root}/critic/mlite_critic.yaml"

if [[ "${MODE}" == quick_alignment_test ]]; then
  source_model_path="$(readlink -m "${MODEL_PATH}")"
  quick_model_path="${runtime_config_root}/quick-model"
  mkdir -p "${quick_model_path}"
  for source in "${source_model_path}"/*; do
    [[ "$(basename "${source}")" == config.json ]] && continue
    ln -s "${source}" "${quick_model_path}/$(basename "${source}")"
  done
  cp "${source_model_path}/config.json" "${quick_model_path}/config.json"
  sed -E -i \
    's/("num_hidden_layers"[[:space:]]*:[[:space:]]*)[0-9]+/\1 4/' \
    "${quick_model_path}/config.json"
  grep -Eq \
    '"num_hidden_layers"[[:space:]]*:[[:space:]]*4[[:space:]]*[,}]' \
    "${quick_model_path}/config.json" ||
    die "failed to create the four-layer quick alignment config"
  MODEL_PATH="${quick_model_path}"
fi

ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/ds4_true_on_policy_preview/${HARDWARE}/${MODE}}"
RUN_NAME="${RUN_NAME:-ds4_v4_${HARDWARE}_${MODE//-/_}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
if [[ "${DRY_RUN:-0}" != 1 && "${NNODES}" -gt 1 ]]; then
  : "${RAY_ADDRESS:?multi-node modes require an existing Ray cluster}"
fi

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${CKPT_DIR}" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${JSONL_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

# --- Internal container/Ray environment; normally do not edit ---
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PATH="/opt/ds4-venv/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

export NVSHMEM_MAX_TEAMS="${NVSHMEM_MAX_TEAMS:-7}"
export NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}"
export VLLM_DEEPEP_BUFFER_SIZE_MB="${VLLM_DEEPEP_BUFFER_SIZE_MB:-1024}"
export DEEPEP_MAX_NVL_PEERS="${DEEPEP_MAX_NVL_PEERS:-${NGPUS_PER_NODE}}"
export ACTOR_MOE_DISPATCHER=deepep
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NGPUS_PER_NODE}"

export PYTHONHASHSEED="${SEED}"
export PYTHONPATH="/opt/ds4/overlays/deepep-nvl${NGPUS_PER_NODE}/site-packages:${VERL_ROOT}:${MEGATRON_ROOT}:${MEGATRON_ROOT}/experimental/lite:${MLITE_EXAMPLE_ROOT}:${PYTHONPATH:-}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES LD_PRELOAD

if [[ "${NNODES}" == 1 ]]; then
  export NVSHMEM_REMOTE_TRANSPORT=none
else
  unset NVSHMEM_REMOTE_TRANSPORT
  export NVSHMEM_ENABLE_NIC_PE_MAPPING="${NVSHMEM_ENABLE_NIC_PE_MAPPING:-1}"
  export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}"
  export NVSHMEM_IB_ADDR_FAMILY="${NVSHMEM_IB_ADDR_FAMILY:-AF_INET6}"
  export NVSHMEM_IB_ADDR_RANGE="${NVSHMEM_IB_ADDR_RANGE:-fe80::/10}"
  export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-0}"
fi

# Exporting in this launcher is not enough for an existing Ray cluster.
RAY_ENV_NAMES=(
  PATH PYTHONPATH LD_LIBRARY_PATH
  PYTHONNOUSERSITE CUDA_DEVICE_MAX_CONNECTIONS
  RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  PYTHONHASHSEED VLLM_BATCH_INVARIANT VERL_FULL_DETERMINISM
  VLLM_BATCH_INVARIANT_KERNEL_LIB DS4_BI_TOPK_LIB
  DEEPEP_MAX_NVL_PEERS NVSHMEM_MAX_TEAMS NVSHMEM_DISABLE_NCCL
  VLLM_DEEPEP_BUFFER_SIZE_MB
  ACTOR_MOE_DISPATCHER NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
  VLLM_DS4_DECODE_KERNEL VERL_FILE_LOGGER_PATH
)
RAY_RUNTIME_ENV=()
for name in "${RAY_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
  )
done

if (( NNODES > 1 )); then
  for name in NVSHMEM_ENABLE_NIC_PE_MAPPING NVSHMEM_HCA_LIST \
    NVSHMEM_IB_ADDR_FAMILY NVSHMEM_IB_ADDR_RANGE NVSHMEM_IB_GID_INDEX; do
    RAY_RUNTIME_ENV+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
    )
  done
fi

for name in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL; do
  if [[ -v "${name}" ]]; then
    RAY_RUNTIME_ENV+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\""
    )
  fi
done

if [[ -n "${NVSHMEM_REMOTE_TRANSPORT:-}" ]]; then
  RAY_RUNTIME_ENV+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.NVSHMEM_REMOTE_TRANSPORT=\"${NVSHMEM_REMOTE_TRANSPORT}\""
  )
fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ "${DEEPEP_MAX_NVL_PEERS}" == "${NGPUS_PER_NODE}" ]] ||
    die "DEEPEP_MAX_NVL_PEERS must equal NGPUS_PER_NODE"
  [[ -s "${VLLM_BATCH_INVARIANT_KERNEL_LIB}" ]] ||
    die "missing batch-invariant kernel"
  [[ -s "${DS4_BI_TOPK_LIB}" ]] || die "missing deterministic top-k kernel"
  IFS=, read -r -a train_files <<<"${TRAIN_FILES}"
  IFS=, read -r -a val_files <<<"${VAL_FILES}"
  for file in "${train_files[@]}"; do
    [[ -f "${file}" ]] || die "missing train file: ${file}"
  done
  for file in "${val_files[@]}"; do
    [[ -f "${file}" ]] || die "missing validation file: ${file}"
  done
fi

# --- Hydra overrides: only non-default preview behavior ---
HYDRA_ARGS=(
  # Algorithm and data.
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.norm_adv_by_std_in_grpo=False
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.prompt_key=prompt
  data.return_raw_chat=True
  data.filter_overlong_prompts=False
  data.truncation=error
  +data.apply_chat_template_kwargs.enable_thinking=True

  # Model and mLite actor.
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_fused_kernels=True
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
  actor_rollout_ref.actor.engine.cp="${ACTOR_CP}"
  actor_rollout_ref.actor.engine.ep="${ACTOR_EP}"
  '~actor_rollout_ref.actor.engine.grad_offload'
  '~actor_rollout_ref.ref.engine.grad_offload'
  actor_rollout_ref.actor.engine.load_hf_weights=True
  +actor_rollout_ref.actor.engine.cross_entropy_fusion=True
  actor_rollout_ref.actor.engine.resync_format=block_fp8
  +actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8
  +actor_rollout_ref.actor.engine.impl_cfg.recompute=full

  # vLLM rollout.
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_WORKERS}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="${VLLM_WORKER_EXTENSION}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8
  +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=deep_gemm
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.expert_dtype=fp8
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.activation_scheme=dynamic
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.fmt=e4m3
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.quant_method=fp8
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.scale_fmt=ue8m0
  +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.weight_block_size='[128,128]'
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config='{cudagraph_mode:FULL_DECODE_ONLY}'

  # Reward and trainer.
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=False
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${RUN_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
  trainer.use_v1=False
)

# --- Launch ---
COMMAND=(
  python3 -m verl.trainer.main_ppo
  "hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]"
  model_engine=mlite
  "${HYDRA_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${MODE_ARGS[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "${HYDRA_OVERRIDES[@]}"
)

print_command() {
  local arg
  for arg in "${COMMAND[@]}"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

printf 'MODE=%s HARDWARE=%s TOPOLOGY=%sx%s OPTIMIZER=%s\n' \
  "${MODE}" "${HARDWARE}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_OPTIMIZER}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command
  exit 0
fi

if [[ "${COMPOSE_ONLY:-0}" == 1 ]]; then
  "${COMMAND[@]}" --cfg job --resolve
  exit 0
fi

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
run_rc="${PIPESTATUS[0]}"
set -e
exit "${run_rc}"
