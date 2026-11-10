#!/usr/bin/env bash
# GRPO/DAPO | Kimi-K3.1 (MXFP4 routed experts) LoRA | true on-policy | vLLM rollout + Megatron mLite | AMD ROCm (MI355X)
#
# One recipe; VLLM_BATCH_INVARIANT alone selects the arm:
#   VLLM_BATCH_INVARIANT=1 (default)  consistent arm: batch-invariant vLLM rollout, Megatron mLite actor with the
#                                     vLLM-aligned Kimi-K3.1 implementation (actor_rollout_ref.actor.engine.impl=vllm,
#                                     model_name=k31): rollout log-probs == trainer log-probs bitwise
#                                     (training/rollout_probs_diff_max = 0.0).
#   VLLM_BATCH_INVARIANT=0            baseline: default vLLM kernels, the same actor without its batch-invariant paths;
#                                     the rollout/trainer mismatch is corrected with token-level truncated IS.
# Algorithm, data, lengths, batch, LoRA, optimizer and parallel layout are shared by both arms.
# Same structure as examples/grpo_trainer/run_deepseek_v4_flash_true_on_policy_rocm.sh (#8505).
#
# The defaults are the configuration of the validated 2 x 8 MI355X runs (seed 42, 16 prompts x 8 samples, prompt 3072 /
# response 4096 tokens): rollout_probs_diff_max 0.0 on every step.
#
# What trains: a LoRA adapter (r=32, alpha=64) on attention, dense / shared-expert MLP and the routed experts; the
# MXFP4 base stays frozen. LORA_ADAPTER_PATH is the initial adapter (a peft directory with these targets).
#
# Layout (2 nodes x 8 GPUs):
#   actor   mLite impl=vllm, PP4 x EP4, TP1, full recompute, parameters offloaded while the rollout generates.
#   rollout one vLLM engine per node, TP8 + EP8 (MI355X: 288 GB per GPU; ~190 GiB of weights per GPU at TP8).
#   sync    the rollout sleeps at level 2 and reloads its frozen base from MODEL_PATH on every wake
#           (VERL_K31_WAKE_RELOAD=1); the adapter is exported while vLLM still sleeps (VERL_LORA_EXPORT_BEFORE_WAKE=1)
#           and each rollout worker receives only the expert factors of its EP rank (VERL_LORA_EXPERT_SLICE=1).
#
# Image: docker/rocm/Dockerfile.rocm.kimi-k31.true-on-policy (the ROCm vLLM build with Kimi-K3.1 support + the DSA
# batch-invariance patch, Megatron-LM mLite with its Kimi-K3.1 ROCm port, and the vLLM plugins in K31_PLUGIN_ROOT:
# k31lora and k31kdabi for the consistent arm, k31lora and k31xlora for the baseline arm).
#
# Containers: --device /dev/kfd --device /dev/dri --device /dev/infiniband --ulimit memlock=-1:-1
#             --ulimit nofile=1048576:1048576. Export this arm's environment before `ray start` on every node as well:
#             some of it is read at import time in Ray-spawned processes.
#
# Required env: MODEL_PATH (Kimi-K3.1 HF directory, node-local on every node), LORA_ADAPTER_PATH, TRAIN_FILES,
# VAL_FILES; VERL_ROOT, MEGATRON_ROOT and K31_PLUGIN_ROOT (set by the image); RAY_ADDRESS of an existing Ray cluster
# for NNODES > 1. K31_PLUGIN_PATH overrides the arm's plugin directories.
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
SEED="${SEED:-42}"
HYDRA_OVERRIDES=("$@")

: "${MODEL_PATH:?set MODEL_PATH to the Kimi-K3.1 checkpoint (node-local on every node)}"
: "${LORA_ADAPTER_PATH:?set LORA_ADAPTER_PATH to the initial peft adapter directory}"
: "${TRAIN_FILES:?set TRAIN_FILES to the DAPO-format training parquet}"
: "${VAL_FILES:?set VAL_FILES to the DAPO-format validation parquet}"
: "${VERL_ROOT:?set VERL_ROOT to this verl checkout}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to the Megatron-LM checkout with experimental/lite}"
: "${K31_PLUGIN_ROOT:=/opt/k31-plugins/kimi-k31}"
if [[ "${VLLM_BATCH_INVARIANT}" == 1 ]]; then
  : "${K31_PLUGIN_PATH:=${K31_PLUGIN_ROOT}/k31lora:${K31_PLUGIN_ROOT}/k31kdabi}"
else
  : "${K31_PLUGIN_PATH:=${K31_PLUGIN_ROOT}/k31lora:${K31_PLUGIN_ROOT}/k31xlora}"
fi
[[ -s "${MODEL_PATH}/config.json" ]] || die "missing ${MODEL_PATH}/config.json"

# --- Workload (shared by both arms) -------------------------------------------------------------------------------
: "${NNODES:=2}" "${NGPUS_PER_NODE:=8}"
: "${TRAIN_BATCH_SIZE:=16}" "${PPO_MINI_BATCH_SIZE:=16}" "${ROLLOUT_N:=8}"
: "${MAX_PROMPT_LENGTH:=3072}" "${MAX_RESPONSE_LENGTH:=4096}"
: "${TOTAL_TRAINING_STEPS:=3}" "${SAVE_FREQ:=-1}" "${TEST_FREQ:=-1}"
: "${ACTOR_LR:=1e-6}" "${LORA_RANK:=32}" "${LORA_ALPHA:=64}"
: "${ACTOR_PP:=4}" "${ACTOR_EP:=4}"
: "${ROLLOUT_TP:=8}" "${ROLLOUT_EP:=8}"
: "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.85}" "${ROLLOUT_MAX_NUM_SEQS:=64}" "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=8192}"
# Shards in flight per worker for the safetensors reload on every wake (~6 GB each); 0 = serial.
: "${ROLLOUT_LOAD_THREADS:=4}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
# One max-length sequence (twice for the 4096-token tail) per micro-batch; the actor never splits a sequence (CP 1).
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=$((2 * MAX_MODEL_LEN))}"
WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))
PROJECT_NAME="${PROJECT_NAME:-verl-kimi-k31-rocm-true-on-policy}"

read -r NUM_LAYERS NUM_EXPERTS < <(python3 -I -c '
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("text_config") or c
print(t["num_hidden_layers"], t.get("n_routed_experts") or t.get("num_experts") or 0)' "${MODEL_PATH}/config.json")
(( WORLD_SIZE % ACTOR_PP == 0 )) || die "ACTOR_PP=${ACTOR_PP} must divide ${WORLD_SIZE} GPUs"
(( NUM_EXPERTS == 0 || NUM_EXPERTS % ACTOR_EP == 0 )) || die "ACTOR_EP=${ACTOR_EP} must divide ${NUM_EXPERTS} experts"
(( ROLLOUT_EP == 1 || ROLLOUT_EP == ROLLOUT_TP )) || die "ROLLOUT_EP must be 1 or ROLLOUT_TP"
SELECTED_LAYERS="[$(seq -s, 0 $((NUM_LAYERS - 1)))]"
LORA_TARGETS='[q_proj,k_proj,v_proj,g_proj,f_a_proj,b_proj,f_b_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,gate_proj,up_proj,down_proj,experts]'

# --- Environment shared by both arms ----------------------------------------------------------------------------
export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# tvm_ffi (imported through tilelang) otherwise rebuilds an optional torch DLPack addon on every import.
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ROCM_USE_AITER=1
# Routed experts with BF16 activations (W4A16), which the trainer reproduces; 1 would select the A4W4 MoE.
export VLLM_ROCM_USE_AITER_MOE_SITUV2=0
export MLITE_K31_BACKEND=rocm
# rollout.full_determinism=True would make verl's vLLM server set VLLM_BATCH_INVARIANT=1; both arms leave it off and
# set VLLM_BATCH_INVARIANT themselves.
export VERL_FULL_DETERMINISM=0
# Weight sync: level-2 sleep + reload of the frozen base on wake, adapter export before wake, per-rank expert slices.
export VERL_VLLM_SLEEP_LEVEL=2 VERL_K31_WAKE_RELOAD=1
export VERL_LORA_EXPORT_BEFORE_WAKE=1 VERL_FREE_GRADS_BEFORE_WAKE=1
export VERL_LORA_EXPERT_SLICE=1
export VERL_K31_SYNC_DEBUG="${VERL_K31_SYNC_DEBUG:-1}" VERL_K31_RELEASE_IF_AWAKE_GIB="${VERL_K31_RELEASE_IF_AWAKE_GIB:-120}"
# Per-token trainer and rollout log-probs in the rollout_data_dir dump (set trainer.rollout_data_dir=<dir>).
export VERL_K31_DUMP_LOGPROBS="${VERL_K31_DUMP_LOGPROBS:-1}"
# Keep the TaskRunner (dataloader state, checkpoint tracker) on this node, so a resume from a node-local CKPT_DIR
# finds them.
export VERL_TASK_RUNNER_ON_DRIVER_NODE=1
export PYTHONPATH="${K31_PLUGIN_PATH}:${VERL_ROOT}:${MEGATRON_ROOT}:${MEGATRON_ROOT}/experimental/lite/examples/verl"

ARM_ENV_NAMES=(PYTHONPATH PYTHONHASHSEED PYTHONUNBUFFERED CUDA_DEVICE_MAX_CONNECTIONS VLLM_NO_USAGE_STATS
  VLLM_ALLOW_INSECURE_SERIALIZATION TORCH_NCCL_AVOID_RECORD_STREAMS VLLM_USE_BREAKABLE_CUDAGRAPH
  VLLM_ENGINE_READY_TIMEOUT_S HF_HUB_OFFLINE TVM_FFI_DISABLE_TORCH_C_DLPACK VLLM_ROCM_USE_AITER
  VLLM_ROCM_USE_AITER_MOE_SITUV2 MLITE_K31_BACKEND VLLM_BATCH_INVARIANT VERL_FULL_DETERMINISM VERL_VLLM_SLEEP_LEVEL
  VERL_K31_WAKE_RELOAD VERL_LORA_EXPORT_BEFORE_WAKE VERL_FREE_GRADS_BEFORE_WAKE VERL_LORA_EXPERT_SLICE
  VERL_K31_SYNC_DEBUG VERL_K31_RELEASE_IF_AWAKE_GIB VERL_K31_DUMP_LOGPROBS VERL_TASK_RUNNER_ON_DRIVER_NODE
  VERL_FILE_LOGGER_PATH)
# RDMA: select the ibverbs userspace that matches the host's AMD ionic NIC driver (docker/rocm/ionic-rdma-env.sh of
# #8505; with the wrong one RCCL silently falls back to TCP sockets) and pass the choice on to the Ray workers.
if command -v ionic-rdma-env.sh >/dev/null; then
  eval "$(ionic-rdma-env.sh)"
  if [[ "${LD_LIBRARY_PATH:-}" == */opt/rdma-core/lib* ]]; then ARM_ENV_NAMES+=(LD_LIBRARY_PATH NCCL_DMABUF_ENABLE); fi
fi
# Site-specific RCCL / socket settings (NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_TOPO_FILE, NCCL_SOCKET_IFNAME,
# GLOO_SOCKET_IFNAME, ...) come from the environment and are forwarded as they are.
for name in $(compgen -e | grep -E '^(NCCL_|GLOO_|RCCL_)' || true); do ARM_ENV_NAMES+=("${name}"); done

VC=actor_rollout_ref.rollout.engine_kwargs.vllm
IMPL=actor_rollout_ref.actor.engine.impl_cfg

# --- Arm switch: the only place the arms differ ------------------------------------------------------------------
if [[ "${VLLM_BATCH_INVARIANT}" == 1 ]]; then
  ARM=consistent
  # Trainer: virtual TP = rollout TP (the trainer reproduces the TP8 rank partials and their batch-invariant
  # reduction), batch-invariant KDA and MoE.
  export MLITE_K31_BI_VTP="${ROLLOUT_TP}" MLITE_K31_KDA_BI=1 MLITE_K31_MOE_BI=1
  # Rollout plugins: batch-invariant MXFP4 MoE with expert LoRA, LoRA added in FP32, KDA FULL decode graphs.
  export K31_BI_MOE=1 K31_BI_LORA_F32ADD=1 K31LORA_EXPERTS=1 K31_BI_TMP="${K31_BI_TMP:-/tmp}" K31_KDA_BI_FULL=1
  # Bit-neutral speed switches of the batch-invariant MoE (same output bits; on in the faster validated run).
  export K31_BI_MOE_BM_LARGE="${K31_BI_MOE_BM_LARGE:-128}" K31_BI_MOE_FUSED_COMBINE_EP="${K31_BI_MOE_FUSED_COMBINE_EP:-1}"
  export K31_BI_MOE_FUSED_GLUE="${K31_BI_MOE_FUSED_GLUE:-1}" K31_BI_MOE_SCALED="${K31_BI_MOE_SCALED:-1}"
  ARM_ENV_NAMES+=(MLITE_K31_BI_VTP MLITE_K31_KDA_BI MLITE_K31_MOE_BI K31_BI_MOE K31_BI_LORA_F32ADD K31LORA_EXPERTS
    K31_BI_TMP K31_KDA_BI_FULL K31_BI_MOE_BM_LARGE K31_BI_MOE_FUSED_COMBINE_EP K31_BI_MOE_FUSED_GLUE K31_BI_MOE_SCALED)
  ARM_ARGS=(
    # FULL decode graphs only (batch-invariance gated; piecewise graphs are not). The KV cache and the sparse MLA
    # decode run the trainer's formulation (fp8_ds_mla, MQA).
    "+${VC}.compilation_config='{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"
    "+${VC}.kv_cache_dtype=fp8_ds_mla"
    "+${VC}.attention_config.sparse_mla_force_mqa=True"
    # The batch-invariant KDA refuses prefix caching; routing replay needs it and is not needed here.
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False
    actor_rollout_ref.actor.engine.router_replay_mode=disabled
    # old_log_prob is recomputed by the actor; with bitwise-equal log-probs no correction is applied.
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is=null
  )
else
  ARM=baseline
  # Expert LoRA through the W4A16 MoE of the LoRA plugin (XLORA_FAST) with AITER CK.
  export XLORA_FAST=1 VLLM_ROCM_USE_AITER_MOE=0 K31LORA_EXPERTS=1
  ARM_ENV_NAMES+=(XLORA_FAST VLLM_ROCM_USE_AITER_MOE K31LORA_EXPERTS)
  ARM_ARGS=(
    "+${VC}.moe_backend=emulation"
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False
    actor_rollout_ref.actor.engine.router_replay_mode=disabled
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=2.0
  )
fi

# --- Outputs: only the baseline arm is suffixed, so resume_mode=auto never crosses arms ----------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/kimi_k31_rocm_true_on_policy}"
RUN_NAME="${RUN_NAME:-kimi_k31_rocm}"
[[ "${ARM}" == baseline ]] && RUN_NAME="${RUN_NAME}_bi0"
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

# verl's default config tree has no mlite model_engine; same construction as #7688.
runtime_config_root="$(mktemp -d "${TMPDIR:-/tmp}/kimi-k31-rocm-config.XXXXXX")"
trap 'rm -rf "${runtime_config_root}"' EXIT
mkdir -p "${runtime_config_root}/critic" "${runtime_config_root}/model_engine"
printf '%s\n' '# @package _global_' 'model_engine: mlite' >"${runtime_config_root}/model_engine/mlite.yaml"
printf '%s\n' '_target_: verl.workers.config.CriticConfig' 'enable: false' 'strategy: mlite' \
  >"${runtime_config_root}/critic/mlite_critic.yaml"

# Exporting here does not reach the workers of an existing Ray cluster: forward explicitly.
RAY_RUNTIME_ENV=()
for name in "${ARM_ENV_NAMES[@]}"; do
  RAY_RUNTIME_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.${name}='${!name}'")
done

# --- Hydra overrides shared by both arms -------------------------------------------------------------------------
HYDRA_ARGS=(
  "hydra.searchpath=[file://${runtime_config_root},pkg://verl_mlite.config]"
  model_engine=mlite

  # Algorithm and data.
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.norm_adv_by_std_in_grpo=False
  "data.train_files=['${TRAIN_FILES//,/\',\'}']"
  "data.val_files=['${VAL_FILES//,/\',\'}']"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.seed="${SEED}"
  data.shuffle=False
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=64
  # In-process loading: forked loader workers get killed during Ray teardown.
  data.dataloader_num_workers=0

  # Model and LoRA.
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.lora.rank="${LORA_RANK}"
  actor_rollout_ref.model.lora.alpha="${LORA_ALPHA}"
  actor_rollout_ref.nccl_timeout=3600

  # Actor: Megatron mLite, vLLM-aligned Kimi-K3.1 implementation.
  actor_rollout_ref.actor.engine.impl=vllm
  actor_rollout_ref.actor.engine.model_name=k31
  actor_rollout_ref.actor.engine.tp=1
  actor_rollout_ref.actor.engine.pp="${ACTOR_PP}"
  actor_rollout_ref.actor.engine.cp=1
  actor_rollout_ref.actor.engine.ep="${ACTOR_EP}"
  actor_rollout_ref.actor.engine.etp=1
  actor_rollout_ref.actor.engine.attention_backend_override=null
  actor_rollout_ref.actor.engine.param_offload=True
  actor_rollout_ref.actor.engine.optimizer_offload=False
  # One host copy per tensor; checkpoints hold only the adapter, the frozen base is reloaded from MODEL_PATH.
  actor_rollout_ref.actor.engine.offload_persist=True
  actor_rollout_ref.actor.engine.ckpt_trainable_only=True
  '~actor_rollout_ref.actor.engine.grad_offload'
  "${IMPL}.use_thd=False"
  "${IMPL}.qat=null"
  # No DeepEP on ROCm: tokens move with the torch all-to-all over RCCL.
  "+${IMPL}.use_deepep=False"
  "+${IMPL}.optimizer=adamw"
  "+${IMPL}.rollout_max_model_len=${MAX_MODEL_LEN}"
  "+${IMPL}.max_tokens_per_rank=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "+${IMPL}.selector={global_layer_ids:${SELECTED_LAYERS},module_names:[attn,moe]}"
  "+${IMPL}.recompute=full"
  "+${IMPL}.runtime_plugins={}"
  "+${IMPL}.lora={rank:${LORA_RANK},alpha:${LORA_ALPHA},target_modules:${LORA_TARGETS}}"
  "+${IMPL}.max_lora_rank=${LORA_RANK}"
  "+${IMPL}.lora_adapter_path=${LORA_ADAPTER_PATH}"
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0

  # vLLM rollout: one TP8 + EP8 engine per node, text only, the adapter on every module the actor trains.
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=auto
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size=1
  actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}"
  actor_rollout_ref.rollout.agent.num_workers="${WORLD_SIZE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.full_determinism=False
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  # Also the actor's rollout_max_model_len: the DSA indexer's chunk limits depend on it.
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enforce_eager=False
  "+${VC}.language_model_only=True"
  "+${VC}.enable_moe_shared_loras=True"
  "+${VC}.lora_target_modules=${LORA_TARGETS}"
  "+${VC}.model_loader_extra_config='{\"enable_multithread_load\":true,\"num_threads\":${ROLLOUT_LOAD_THREADS}}'"

  # Reward and trainer.
  reward.reward_manager.name=dapo
  reward.custom_reward_function.path="${VERL_ROOT}/examples/grpo_trainer/k31_reward.py"
  reward.custom_reward_function.name=compute_score
  +reward.reward_kwargs.overlong_buffer_cfg.enable=False
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${RUN_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
  trainer.use_v1=False
  critic.enable=False
)

COMMAND=(
  python3 -m verl.trainer.main_ppo
  "${HYDRA_ARGS[@]}"
  "${ARM_ARGS[@]}"
  "${RAY_RUNTIME_ENV[@]}"
  "${HYDRA_OVERRIDES[@]}"
)

printf 'ARM=%s VLLM_BATCH_INVARIANT=%s TOPOLOGY=%sx%s ACTOR=PP%s/EP%s ROLLOUT=TP%s/EP%s LAYERS=%s EXPERTS=%s\n' \
  "${ARM}" "${VLLM_BATCH_INVARIANT}" "${NNODES}" "${NGPUS_PER_NODE}" "${ACTOR_PP}" "${ACTOR_EP}" \
  "${ROLLOUT_TP}" "${ROLLOUT_EP}" "${NUM_LAYERS}" "${NUM_EXPERTS}"

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

(( NNODES == 1 )) || : "${RAY_ADDRESS:?multi-node runs need an existing Ray cluster (RAY_ADDRESS)}"
[[ -s "${LORA_ADAPTER_PATH}/adapter_config.json" ]] || die "missing ${LORA_ADAPTER_PATH}/adapter_config.json"
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
