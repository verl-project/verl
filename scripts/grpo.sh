#!/bin/bash
# set -x

# ============================ Configurations ===========================
PROJECT_DIR="$HOME/CoT-Data-verl"   # !!!!! Change this to where you want to save logs!!!!!
STORE_DIR="/data/hjw"               # !!!!! Change this to where you want to save checkpoints!!!!!

MODEL_NAME="Qwen2.5-0.5B"
DATA_NAME="gsm8k"
DATA_DIR="/data/open_datasets/GSM8K"
SFT_CHECKPOINT="${STORE_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--sft/checkpoint-last"    # SFT后的模型路径

GRPO_OUTPUT_DIR="${STORE_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--grpo"
LOG_PATH="${PROJECT_DIR}/wandb_logs"
LOG_NAME="${MODEL_NAME}--${DATA_NAME}--grpo--$(date +%Y%m%d-%H%M%S)"
# =======================================================================

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <gpu_ids> [other_configs...]"
    echo "Example: bash $0 0,1,2,3,4,5,6,7"
    echo "         bash $0 0,1,2,3      # 使用4卡"
    echo "         bash $0 4,5,6,7      # 使用后4张卡"
    exit 1
fi

gpu_ids=$1
shift 1

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=$gpu_ids

# 自动计算GPU数量
n_gpus_per_node=$(echo $gpu_ids | tr ',' ' ' | wc -w)

echo "=========================================="
echo "Using GPUs: $gpu_ids"
echo "Auto-calculated n_gpus_per_node: $n_gpus_per_node"
echo "=========================================="

# 检查 tensor_model_parallel_size 兼容性
tp_size=2
if [ $((n_gpus_per_node % tp_size)) -ne 0 ]; then
    echo "Warning: n_gpus_per_node($n_gpus_per_node) is not divisible by tensor_model_parallel_size($tp_size)"
    echo "建议修改 tensor_model_parallel_size 为 $n_gpus_per_node 的约数 (如1, 2, $n_gpus_per_node)"
    tp_size=1  # 自动回退到 1 避免错误
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

export WANDB_DIR=${LOG_PATH}
export WANDB_RUN_ID="exp-${MODEL_NAME}-${DATA_NAME}-grpo"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${SFT_CHECKPOINT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${DATA_NAME}_grpo \
    trainer.experiment_name=${LOG_NAME} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=${GRPO_OUTPUT_DIR} $@
