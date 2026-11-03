#!/usr/bin/env bash
# GRPO/DAPO | DeepSeek-V4-Flash FP8 | true on-policy | vLLM rollout + Megatron | AMD ROCm (MI355X, gfx950)
#
# One recipe and one image; VLLM_BATCH_INVARIANT alone selects the arm:
#   VLLM_BATCH_INVARIANT=1 (default)  consistent arm: batch-invariant vLLM rollout, Megatron mLite actor with the
#                                     vLLM-aligned implementation (actor_rollout_ref.actor.engine.impl=ALIGNED_ACTOR_IMPL),
#                                     rollout log-probs == trainer log-probs bitwise.
#   VLLM_BATCH_INVARIANT=0            native baseline: default vLLM kernels, Megatron-Core actor (model_engine=megatron)
#                                     with FP8 QAT and routing replay (R3), as the merged ROCm recipe
#                                     run_deepseek_v4_flash_megatron_rocm.sh trains it.
# Algorithm, data, lengths, batch, optimizer hyper-parameters and rollout layout are shared by both arms.
# Modelled on verl-project/verl#7688 (examples/grpo_trainer/run_deepseek_v4_true_on_policy_preview_megatron.sh).
#
# Image: docker/rocm/Dockerfile.rocm.deepseek-v4.true-on-policy (vLLM 0.26.0 + batch-invariance patches, Megatron-LM
# with mLite impl=vllm on ROCm). The full-profile defaults are the configuration of the validated 4-node run:
# rollout_probs_diff_max 0.0 on every step, including steps after exact resumes.
#
# Container requirement: raise the open-file limit (docker run --ulimit nofile=1048576:1048576). With the default
# limit, the multi-node full profile failed after ~26 steps in gloo ("Too many open files", tcp/device.cc).
#
# PROFILE
#   proxy  1 node x 4 GPUs, MODEL_PATH = a 4-layer cut of the checkpoint, short lengths, 3 steps: the daily cell.
#          A 4-layer cut never finishes an answer, so DAPO rewards are constant and every advantage is 0; pass a
#          varying reward for real updates, e.g. reward.custom_reward_function.path=<file> ...name=compute_score.
#   full   4 nodes x 8 GPUs, full checkpoint, the formal workload. Two-step acceptance = PROFILE=full
#          TOTAL_TRAINING_STEPS=2 with nothing else changed (constant LR, no warmup, so the first two steps of the
#          schedule are the formal ones).
#
# Required env: MODEL_PATH (HF directory with config.json and the tokenizer/chat template), TRAIN_FILES, VAL_FILES.
# Multi-node: an existing Ray cluster (RAY_ADDRESS) whose workers run this image.
# Modes: DRY_RUN=1 prints the arm environment and the full command; COMPOSE_ONLY=1 prints the resolved Hydra config.
# Extra arguments are passed to verl.trainer.main_ppo as Hydra overrides (last wins).
set -euo pipefail

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

# --- Arm --------------------------------------------------------------------------------------------------------
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
case "${VLLM_BATCH_INVARIANT}" in
  0|1) ;;
  *) die "VLLM_BATCH_INVARIANT must be 0 or 1, got '${VLLM_BATCH_INVARIANT}'" ;;
esac
PROFILE="${PROFILE:-full}"
SEED="${SEED:-42}"
# mLite engine implementation of the consistent arm (actor_rollout_ref.actor.engine.impl): `vllm` = the vLLM-aligned
# DeepSeek-V4 forward on the rollout's own ROCm kernels with deterministic VJPs (the validated default);
# `vllm_engine` = forward from an in-process vLLM engine (needs CP 1 and rollout DP = EP = actor EP).
ALIGNED_ACTOR_IMPL="${ALIGNED_ACTOR_IMPL:-vllm}"
HYDRA_OVERRIDES=("$@")

: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4-Flash checkpoint (or its 4-layer cut for PROFILE=proxy)}"
: "${TRAIN_FILES:?set TRAIN_FILES to the DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to the DAPO-format validation parquet}"
[[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"

# --- Workload profile (shared by both arms) ---------------------------------------------------------------------
case "${PROFILE}" in
  proxy)
    : "${NNODES:=1}" "${NGPUS_PER_NODE:=4}"
    : "${TRAIN_BATCH_SIZE:=4}" "${PPO_MINI_BATCH_SIZE:=4}" "${ROLLOUT_N:=4}"
    : "${MAX_PROMPT_LENGTH:=2048}" "${MAX_RESPONSE_LENGTH:=2048}" "${OVERLONG_BUFFER_LEN:=512}"
    : "${ROLLOUT_DP:=4}" "${ROLLOUT_EP:=4}" "${ROLLOUT_AGENT_WORKERS:=4}"
    : "${ROLLOUT_MAX_NUM_SEQS:=16}" "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.30}"
    : "${TOTAL_TRAINING_STEPS:=3}" "${SAVE_FREQ:=-1}" "${TEST_FREQ:=-1}"
    ;;
  full)
    : "${NNODES:=4}" "${NGPUS_PER_NODE:=8}"
    : "${TRAIN_BATCH_SIZE:=128}" "${PPO_MINI_BATCH_SIZE:=32}" "${ROLLOUT_N:=8}"
    : "${MAX_PROMPT_LENGTH:=2048}" "${MAX_RESPONSE_LENGTH:=14000}" "${OVERLONG_BUFFER_LEN:=4096}"
    # One rollout replica per node (TP1 x DP8 x EP8). impl=vllm_engine instead needs one replica across all GPUs
    # (ROLLOUT_DP = ROLLOUT_EP = NNODES x 8).
    : "${ROLLOUT_DP:=8}" "${ROLLOUT_EP:=8}" "${ROLLOUT_AGENT_WORKERS:=32}"
    : "${ROLLOUT_MAX_NUM_SEQS:=128}" "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.65}"
    : "${TOTAL_TRAINING_STEPS:=50}" "${SAVE_FREQ:=3}" "${TEST_FREQ:=10}"
    ;;
  *)
    die "PROFILE must be proxy or full, got '${PROFILE}'"
    ;;
esac
: "${ACTOR_LR:=1e-6}"
: "${ROLLOUT_TP:=1}"
: "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=8192}"
: "${ROLLOUT_CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"
: "${PARAM_OFFLOAD:=False}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PROJECT_NAME="${PROJECT_NAME:-verl-ds4flash-rocm-true-on-policy}"

# --- Environment shared by both arms ----------------------------------------------------------------------------
export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# MoE backend (rollout.moe_backend, set per arm below): the batch-invariant arm uses the Triton MoE, whose BI tiles the
# trainer mirrors; the native arm uses AITER, the fast ROCm MoE. The EP expert-mask bug that once required turning
# AITER MoE off is fixed in the vLLM branch (294014e28).

ARM_ENV_NAMES=(PYTHONHASHSEED VLLM_BATCH_INVARIANT VERL_FULL_DETERMINISM
  CUDA_DEVICE_MAX_CONNECTIONS VLLM_ALLOW_INSECURE_SERIALIZATION VLLM_NO_USAGE_STATS VERL_FILE_LOGGER_PATH)
[[ -v VLLM_ROCM_USE_AITER_MOE ]] && ARM_ENV_NAMES+=(VLLM_ROCM_USE_AITER_MOE)  # set by the image

# --- Arm switch: the only place the arms differ ------------------------------------------------------------------
runtime_config_root="$(mktemp -d "${TMPDIR:-/tmp}/ds4flash-rocm-config.XXXXXX")"
trap 'rm -rf "${runtime_config_root}"' EXIT

if [[ "${VLLM_BATCH_INVARIANT}" == 1 ]]; then
  ARM=consistent
  export VERL_FULL_DETERMINISM=1
  # Row-count-dependent tiles for vLLM's batch-invariant GEMMs (patches/vllm-bi-13): same output bits on gfx950,
  # faster. The trainer reads it through the same vLLM module, so both sides use the same setting.
  export VLLM_BI_GEMM_TILES="${VLLM_BI_GEMM_TILES:-1}"
  ARM_ENV_NAMES+=(VLLM_BI_GEMM_TILES)
  : "${ACTOR_PP:=1}" "${ACTOR_EP:=$((NNODES * NGPUS_PER_NODE))}"
  # Context parallelism of impl=vllm: request-local prefill chunks (impl=vllm_engine supports CP = 1 only).
  if [[ "${PROFILE}" == full ]]; then : "${ACTOR_CP:=4}"; else : "${ACTOR_CP:=1}"; fi
  : "${ACTOR_OPTIMIZER:=fsdp2}"
  : "${CROSS_ENTROPY_FUSION:=True}"
  : "${ROLLOUT_MOE_BACKEND:=triton}"
  # KV cache of the actor's private vLLM engine (impl=vllm_engine): it only ever holds one micro-batch, so a fixed size
  # instead of a gpu_memory_utilization share (its default 0.15 share leaves no KV room next to the model and the
  # profile run). Used by impl=vllm_engine only.
  : "${ACTOR_ENGINE_KV_CACHE_BYTES:=4000000000}"
  # Tokens per micro-batch and GPU (dynamic batching). The full profile packs 4096 per CP rank.
  [[ "${PROFILE}" == full ]] && : "${PPO_MAX_TOKEN_LEN_PER_GPU:=4096}"
  : "${PPO_MAX_TOKEN_LEN_PER_GPU:=$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"
  # verl's default config tree has no mlite model_engine; same construction as #7688.
  mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
  printf '%s\n' '# @package _global_' 'model_engine: mlite' >"${runtime_config_root}/model_engine/mlite.yaml"
  printf '%s\n' '_target_: verl.workers.config.CriticConfig' 'enable: false' 'strategy: mlite' \
    >"${runtime_config_root}/critic/mlite_critic.yaml"
  ENGINE_ARGS=("hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]" model_engine=mlite)
  ACTOR_ARGS=(
    actor_rollout_ref.actor.engine.impl="${ALIGNED_ACTOR_IMPL}"
    actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
    actor_rollout_ref.actor.engine.cp="${ACTOR_CP}"
    actor_rollout_ref.actor.engine.ep="${ACTOR_EP}"
    +actor_rollout_ref.actor.engine.rollout_ep="${ROLLOUT_EP}"
    +actor_rollout_ref.actor.engine.rollout_tp="${ROLLOUT_TP}"
    '~actor_rollout_ref.actor.engine.grad_offload'
    '~actor_rollout_ref.ref.engine.grad_offload'
    actor_rollout_ref.actor.engine.attention_backend_override=null
    +actor_rollout_ref.actor.engine.cross_entropy_fusion="${CROSS_ENTROPY_FUSION}"
    actor_rollout_ref.actor.engine.resync_format=block_fp8
    +actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8
    # block-FP8 export with the checkpoint's scale rule (ue8m0 -> power-of-two scales), as the rollout loads it
    +actor_rollout_ref.actor.engine.resync_config.fp8_scale_rule=checkpoint
    # The private engine reproduces the rollout: same DP = EP layout (one engine rank per actor rank; impl=vllm_engine
    # requires engine DP = actor EP), same max_model_len, and one prefill step per micro-batch (dynamic batching packs
    # up to PPO_MAX_TOKEN_LEN_PER_GPU tokens; log-prob micro-batches up to MAX_MODEL_LEN).
    +actor_rollout_ref.actor.engine.impl_cfg.engine_data_parallel_size="${ACTOR_EP}"
    +actor_rollout_ref.actor.engine.impl_cfg.engine_max_model_len="${MAX_MODEL_LEN}"
    +actor_rollout_ref.actor.engine.impl_cfg.engine_max_num_batched_tokens="$((PPO_MAX_TOKEN_LEN_PER_GPU > MAX_MODEL_LEN ? PPO_MAX_TOKEN_LEN_PER_GPU : MAX_MODEL_LEN))"
    +actor_rollout_ref.actor.engine.impl_cfg.engine_kv_cache_memory_bytes="${ACTOR_ENGINE_KV_CACHE_BYTES}"
    +actor_rollout_ref.actor.engine.impl_cfg.recompute=full
    +actor_rollout_ref.actor.engine.impl_cfg.mtp_enable=False
    +actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=False
    +actor_rollout_ref.actor.engine.impl_cfg.optimizer="${ACTOR_OPTIMIZER}"
    actor_rollout_ref.actor.engine.param_offload="${PARAM_OFFLOAD}"
    actor_rollout_ref.actor.engine.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=True
    +actor_rollout_ref.actor.engine.seed="${SEED}"
    +actor_rollout_ref.actor.engine.full_determinism=True
  )
  ROLLOUT_ARM_ARGS=(
    # Per-request sampling seeds derived from the request id (verl/workers/rollout/utils.py).
    actor_rollout_ref.rollout.full_determinism=True
    actor_rollout_ref.rollout.seed="${SEED}"
  )
else
  ARM=native
  # rollout.full_determinism=True makes verl's vLLM server export VLLM_BATCH_INVARIANT=1; it stays off here.
  export VERL_FULL_DETERMINISM=0
  if [[ "${PROFILE}" == full ]]; then
    : "${ACTOR_PP:=4}" "${ACTOR_EP:=8}" "${ACTOR_CP:=1}"
    : "${PIPELINE_MODEL_PARALLEL_LAYOUT:=Et*11|t*11|t*11|t*10L}"
  else
    : "${ACTOR_PP:=1}" "${ACTOR_EP:=$((NNODES * NGPUS_PER_NODE))}" "${ACTOR_CP:=1}"
  fi
  # AITER MoE: with batch invariance off, the untuned Triton MoE made generation ~23-26 % slower than the
  # batch-invariant arm; AITER is ~3 % faster than it (MI350X/MI355X rollout A/B).
  : "${ROLLOUT_MOE_BACKEND:=aiter}"
  # Transformer Engine grouped GEMM: the BF16 backward of the default kernel can hang on gfx950.
  export NVTE_USE_GROUPED_GEMM_TRITON=1
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  ARM_ENV_NAMES+=(NVTE_USE_GROUPED_GEMM_TRITON TORCH_NCCL_AVOID_RECORD_STREAMS)
  ENGINE_ARGS=(model_engine=megatron)
  ACTOR_ARGS=(
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
    actor_rollout_ref.actor.megatron.param_offload="${PARAM_OFFLOAD}"
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.router_replay.mode=R3
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
  )
  if [[ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]]; then
    ACTOR_ARGS+=("++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'")
  fi
  # FP8 QAT: weights 128x128 power-of-two scales, inputs 1x128 float scales, STE; the KV latent and the indexer
  # q/k are rounded as the rollout's FP8 KV cache and FP8 indexer round them. Actor and reference alike.
  for kv in gradient_accumulation_fusion=False apply_dsa_kernel_fusion=False fp8_fake_quant_kv_cache=True \
    fp8_fake_quant_indexer_qk=True fp8_fake_quant=True; do
    ACTOR_ARGS+=("++actor_rollout_ref.actor.megatron.override_transformer_config.${kv}"
      "++actor_rollout_ref.ref.megatron.override_transformer_config.${kv}")
  done
  if (( ACTOR_CP > 1 )); then
    ACTOR_ARGS+=(
      ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=contiguous
      ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_packing_scheduler=dp_balanced
      ++actor_rollout_ref.actor.megatron.override_transformer_config.max_seqlen_per_dp_cp_rank="$((MAX_MODEL_LEN / ACTOR_CP))"
    )
  fi
  ROLLOUT_ARM_ARGS=(
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
  )
fi
# Dynamic batching: a sequence must fit one CP group's token budget.
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=$(((MAX_MODEL_LEN + ACTOR_CP - 1) / ACTOR_CP))}"

# --- Outputs: only the native arm is suffixed, so resume_mode=auto never crosses arms ------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/ds4flash_rocm_true_on_policy/${PROFILE}}"
RUN_NAME="${RUN_NAME:-ds4flash_rocm_${PROFILE}}"
[[ "${ARM}" == native ]] && RUN_NAME="${RUN_NAME}_bi0"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
if [[ -n "${TRAINER_LOGGERS:-}" ]]; then
  :
elif [[ -v WANDB_API_KEY || -v WANDB_ENTITY || -v WANDB_MODE ]] && [[ "${WANDB_MODE:-}" != disabled ]]; then
  TRAINER_LOGGERS='[console,file,wandb]'
else
  TRAINER_LOGGERS='[console,file]'
fi

# Exporting here does not reach the workers of an existing Ray cluster: forward explicitly.
RAY_RUNTIME_ENV=()
for name in "${ARM_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\"")
done

# --- Hydra overrides shared by both arms -------------------------------------------------------------------------
VC=actor_rollout_ref.rollout.engine_kwargs.vllm
QC=${VC}.hf_overrides.quantization_config
HYDRA_ARGS=(
  # Algorithm and data.
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.norm_adv_by_std_in_grpo=False
  algorithm.rollout_correction.bypass_mode=False
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.seed="${SEED}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  +data.apply_chat_template_kwargs.enable_thinking=True

  # Model and actor.
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_fused_kernels=True
  actor_rollout_ref.nccl_timeout=3600
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0

  # vLLM rollout. Layout TP1 x DP x EP; FP8 W8A8 (E4M3, 128x128 blocks, UE8M0 scales), FP8 KV cache.
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
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  +${VC}.disable_custom_all_reduce=True
  +${VC}.kv_cache_dtype=fp8
  +${VC}.moe_backend="${ROLLOUT_MOE_BACKEND}"
  +${VC}.tokenizer_mode=deepseek_v4
  +${VC}.compilation_config.cudagraph_mode="${ROLLOUT_CUDAGRAPH_MODE}"
  +${VC}.hf_overrides.expert_dtype=fp8
  +${QC}.activation_scheme=dynamic
  +${QC}.fmt=e4m3
  +${QC}.quant_method=fp8
  +${QC}.scale_fmt=ue8m0
  "+${QC}.weight_block_size=[128,128]"

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
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.max_actor_ckpt_to_keep=2
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
  trainer.use_v1=False
)

COMMAND=(
  python3 -m verl.trainer.main_ppo
  "${ENGINE_ARGS[@]}"
  "${HYDRA_ARGS[@]}"
  "${ACTOR_ARGS[@]}"
  "${ROLLOUT_ARM_ARGS[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "${HYDRA_OVERRIDES[@]}"
)

printf 'ARM=%s VLLM_BATCH_INVARIANT=%s PROFILE=%s TOPOLOGY=%sx%s ACTOR=PP%s/CP%s/EP%s ROLLOUT=TP%s/DP%s/EP%s\n' \
  "${ARM}" "${VLLM_BATCH_INVARIANT}" "${PROFILE}" "${NNODES}" "${NGPUS_PER_NODE}" \
  "${ACTOR_PP}" "${ACTOR_CP}" "${ACTOR_EP}" "${ROLLOUT_TP}" "${ROLLOUT_DP}" "${ROLLOUT_EP}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for name in "${ARM_ENV_NAMES[@]}"; do
    printf 'ENV %s=%s\n' "${name}" "${!name}"
  done
  # The mktemp path is replaced so two dry runs of the same settings print identical bytes.
  for arg in "${COMMAND[@]}"; do
    printf '%q ' "${arg//${runtime_config_root}/<runtime_config_root>}"
  done
  printf '\n'
  exit 0
fi

if [[ "${COMPOSE_ONLY:-0}" == 1 ]]; then
  "${COMMAND[@]}" --cfg job --resolve
  exit 0
fi

(( NNODES == 1 )) || : "${RAY_ADDRESS:?multi-node profiles need an existing Ray cluster (RAY_ADDRESS)}"
IFS=, read -r -a files <<<"${TRAIN_FILES},${VAL_FILES}"
for file in "${files[@]}"; do
  [[ -f "${file}" ]] || die "missing data file: ${file}"
done
mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${VERL_FILE_LOGGER_PATH}")"

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
run_rc="${PIPESTATUS[0]}"
set -e
exit "${run_rc}"
