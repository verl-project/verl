#!/bin/bash
# set -x

# ============================ Configurations ===========================
PROJECT_DIR="$HOME/CoT-Data-verl"   # !!!!! Change this to where you want to save checkpoints & logs!!!!!

MODEL_NAME="Qwen2.5-0.5B"
MODEL_ID="Qwen/Qwen2.5-0.5B"
DATA_NAME="gsm8k"
DATA_DIR="/data/open_datasets/GSM8K"

SAVE_PATH="${PROJECT_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--sft"
LOG_PATH="${PROJECT_DIR}/wandb_logs"
LOG_NAME="${MODEL_NAME}--${DATA_NAME}--sft--$(date +%Y%m%d-%H%M%S)"
# =======================================================================

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <gpu_ids> [other_configs...]"
    echo "Example: bash $0 0,1,2,3,4,5,6,7"
    echo "         bash $0 0,1,2,3      # 使用4卡"
    echo "         bash $0 4,5,6,7      # 使用后4张卡"
    exit 1
fi

gpu_ids=$1

# Shift the arguments so $@ refers to the rest
shift 1

# Limit the visible GPUs
export CUDA_VISIBLE_DEVICES=$gpu_ids

# 自动计算GPU数量（将逗号转为空格后计数）
nproc_per_node=$(echo $gpu_ids | tr ',' ' ' | wc -w)

echo "=========================================="
echo "Using GPUs: $gpu_ids"
echo "Auto-calculated nproc_per_node: $nproc_per_node"
echo "=========================================="

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

export WANDB_DIR=${LOG_PATH}
export WANDB_RUN_ID="exp-${MODEL_NAME}-${DATA_NAME}-sft"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/train_messages.parquet \
    data.val_files=${DATA_DIR}/test_messages.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    optim.lr=1e-4 \
    model.partial_pretrain=${MODEL_ID} \
    model.trust_remote_code=true \
    model.lora_rank=8 \
    model.lora_alpha=16 \
    model.enable_gradient_checkpointing=true \
    model.use_liger=true \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=${DATA_NAME}-sft \
    trainer.experiment_name=${LOG_NAME} \
    trainer.total_epochs=4 \
    trainer.logger='["console","wandb"]' \
    trainer.default_hdfs_dir=null $@