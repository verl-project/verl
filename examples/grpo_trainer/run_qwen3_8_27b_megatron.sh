#!/usr/bin/env bash
# Qwen3.8-27B GRPO with Megatron/MindSpeed-Bridge and vLLM rollout.
#
# The default Ascend setup uses one 16-NPU node and the Geo3K dataset:
#   TP=2 PP=4 CP=1 EP=1 ETP=1, rollout TP=8.
#
# Requirements on Ascend:
#   - 8 A3 cards (2*64GB each, exposed as 16 NPU devices)
#   - Base environment: Python 3.12.13, CANN 9.0.0, ATB 9.0.0.B160,
#     PyTorch 2.10.0 and torch-npu 2.10.0.post4
#   - Main Python packages:
#       transformers==5.5.4 vllm==0.23.0 vllm-ascend==0.23.0 ray==2.56.1
#       flash-linear-attention==0.5.2 fla-npu==1.0.0 nvidia-modelopt==0.45.0
#   - Megatron-LM / Megatron-Core==0.18.0 (ba7b5ebce12a)
#   - MindSpeed-Bridge==0.3.1 with Qwen3.8 provider registration
#   - Megatron-Bridge==0.5.0 (e1ef727af058)
#   - TransformerEngineNPU==2.13.0 (87b4ded237b1)
#   - Qwen3.8 registration under `mindspeed_bridge/models/qwen/` and
#     `mindspeed_bridge/recipes/qwen/` must be present in MindSpeed-Bridge.
#
# Activate an environment satisfying the versions above, then source the CANN
# and ATB `set_env.sh` files from their installation locations.

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
set -xeuo pipefail

########################### Quick Config ###########################

# DEVICE is auto-detected by probing torch_npu. Override it only when needed.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
case "${DEVICE}" in
    gpu)
        TP=${TP:-2}
        PP=${PP:-1}
        CP=${CP:-1}
        EP=${EP:-1}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        ;;
    npu)
        TP=${TP:-2}
        PP=${PP:-4}
        CP=${CP:-1}
        EP=${EP:-1}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

rollout_name=${ROLLOUT_NAME:-vllm}
project_name=${PROJECT_NAME:-verl_grpo_qwen3_8_27b_geo3k}
exp_name=${EXPERIMENT_NAME:-qwen3_8_27b_megatron}
adv_estimator=${ADV_ESTIMATOR:-grpo}

# MODEL_PATH/TRAIN_FILE/VAL_FILE are accepted as compatibility aliases.
HF_MODEL_PATH=${HF_MODEL_PATH:-${MODEL_PATH:-Qwen3.8-27B}}
RUN_ROOT=${RUN_ROOT:-$HOME/verl_runs/qwen3_8_27b_megatron}
train_path=${train_path:-${TRAIN_FILE:-$HOME/data/geo3k/train.parquet}}
test_path=${test_path:-${VAL_FILE:-$HOME/data/geo3k/test.parquet}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
# CaMem sleep mode requires a vllm-ascend C extension built for the installed
# torch-npu. Keep it opt-in so a missing allocator does not prevent startup.
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}

SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_STEPS=${TOTAL_STEPS:-200}

if ((n_devices_per_node % (TP * PP * CP) != 0)); then
    echo "NDEVICES_PER_NODE=${n_devices_per_node} must be divisible by TP*PP*CP=$((TP * PP * CP))." >&2
    exit 1
fi
if ((n_devices_per_node % GEN_TP != 0)); then
    echo "NDEVICES_PER_NODE=${n_devices_per_node} must be divisible by GEN_TP=${GEN_TP}." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/rollouts"

########################### Parameter Arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="${train_path}"
    data.val_files="${test_path}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.truncation=error
    data.filter_overlong_prompts=True
    data.shuffle=False
    data.validation_shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.val_before_train=False
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_STEPS}
    trainer.resume_mode=disable
    trainer.default_local_dir="${RUN_ROOT}/checkpoints"
    trainer.rollout_data_dir="${RUN_ROOT}/rollouts"
)

EXTRA=(
    model_engine=megatron
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
        export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
        export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
        export MINDSPEED_BRIDGE_AUTOREG_MODE="${MINDSPEED_BRIDGE_AUTOREG_MODE:-strict}"
        unset PYTORCH_NPU_ALLOC_CONF

        ACTOR+=(
            actor_rollout_ref.actor.megatron.vanilla_mbridge=False
            actor_rollout_ref.actor.checkpoint.strict=False
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=False
            +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0
        )
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
