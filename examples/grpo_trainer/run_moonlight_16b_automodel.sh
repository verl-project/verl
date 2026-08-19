#!/usr/bin/env bash
# GRPO | Moonlight-16B-A3B-Instruct | vLLM rollout | Automodel (nemo_automodel) training | 8 GPU
# Set USE_LORA=1 to enable Automodel's built-in PEFT support.

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-moonshotai/Moonlight-16B-A3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/dataset/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/vePFS-Mindverse/user/songlin/aime-2024-first30.parquet}
USE_LORA=${USE_LORA:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGETS=${LORA_TARGETS:-all-linear}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

# GRPO, no critic, no KL (so disable_adapter is never needed even for LoRA)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}

ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_N=${ROLLOUT_N:-5}

SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

lora_args=()
case "${USE_LORA}" in
    0)
        ACTOR_LR=${ACTOR_LR:-1e-6}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-true}
        PROJECT_NAME=${PROJECT_NAME:-automodel-moonlight16b-grpo}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-moonlight16b_full}
        ;;
    1)
        ACTOR_LR=${ACTOR_LR:-1e-4}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
        FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-false}
        PROJECT_NAME=${PROJECT_NAME:-automodel-moonlight16b-grpo-lora}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-moonlight16b_lora}
        lora_args+=(
            "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
            "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
            "actor_rollout_ref.model.lora.merge=false"
            "actor_rollout_ref.model.target_modules=${LORA_TARGETS}"
        )
        ;;
    *)
        echo "USE_LORA must be 0 or 1, got '${USE_LORA}'" >&2
        exit 2
        ;;
esac

# TE 2.17 prefers cuDNN FusedAttention over FlashAttention on Hopper "for performance",
# but cuDNN 9.20's fused_attn_f16_arbitrary_seqlen thd backward hits CUDNN_STATUS_BAD_PARAM
# in the reshape op. Force FlashAttention so the thd (remove-padding) backward works.
export NVTE_FUSED_ATTN=0
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

python3 -m verl.trainer.main_ppo \
    model_engine=automodel \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.trust_remote_code=true \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    "${lora_args[@]}" \
    \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    \
    actor_rollout_ref.actor.automodel_config.distributed_strategy=fsdp2 \
    actor_rollout_ref.actor.automodel_config.tp_size=1 \
    actor_rollout_ref.actor.automodel_config.pp_size=1 \
    actor_rollout_ref.actor.automodel_config.cp_size=1 \
    actor_rollout_ref.actor.automodel_config.ep_size=8 \
    actor_rollout_ref.actor.automodel_config.backend_config.dispatcher=deepep \
    actor_rollout_ref.actor.automodel_config.backend_config.attn=te \
    actor_rollout_ref.actor.automodel_config.backend_config.linear=te \
    actor_rollout_ref.actor.automodel_config.backend_config.rms_norm=torch_fp32 \
    actor_rollout_ref.actor.automodel_config.backend_config.enable_fsdp_optimizations=true \
    actor_rollout_ref.actor.automodel_config.backend_config.experts=torch_mm \
    actor_rollout_ref.actor.automodel_config.activation_checkpointing=true \
    actor_rollout_ref.actor.automodel_config.model_dtype=bf16 \
    actor_rollout_ref.actor.automodel_config.attn_implementation=te \
    actor_rollout_ref.actor.automodel_config.use_torch_compile=false \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.free_cache_engine=true \
    \
    critic.enable=false \
    \
    trainer.logger='[console]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=false \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
