#!/usr/bin/env bash
# Qwen3.5-397B-A17B SFT with Megatron backend
# Using verlai/verl:uv.cu130.dev3 docker image
# Requirements:
#   - 128+ GPUs (80GB each, e.g. 16x8 H100/H200)
#   - Image dependency cache: Megatron-Core 0.18.0 / Megatron-Bridge 0.5.2.
#   - Install the current uv.lock's megatron extra, including flash-linear-attention:
#       uv sync --frozen --extra megatron
#       source .venv/bin/activate
#
# CUDA dependencies from the current uv.lock (Python 3.12):
#   Megatron-Core 0.19.2 / Megatron-Bridge 0.6.2; flash-linear-attention 0.5.2.
#
# Qwen3.5 architecture notes:
#   This example uses BSHD compute format:
#     - model.use_remove_padding=False
#     - engine.use_remove_padding=False
#     - data.use_dynamic_bsz=False
#   Megatron-Core 0.18.0 and 0.19.2 also support THD for GDN.
#   The settings above retain BSHD for this example.
#
# MTP (Multi-Token Prediction) notes:
#   - model.mtp.enable=True               enables MTP module
#   - model.mtp.enable_train=True         enables MTP training loss
#   - model.mtp.detach_encoder=True       detaches encoder gradients for MTP
#   - model.mtp.mtp_loss_scaling_factor   weight of MTP auxiliary loss (e.g. 0.1)
#
# Tested parallelism config (128 GPUs / 16 nodes):
#   TP=2 PP=4 EP=32 CP=1

set -xeuo pipefail

# ============================================================
# Distributed
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-16}
NODE_RANK=${NODE_RANK:-0}

# ============================================================
# Data
# ============================================================
DATASET_DIR=${DATASET_DIR:-~/dataset}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}

# ============================================================
# Model
# ============================================================
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-397B-A17B}

# ============================================================
# Parallelism
# ============================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-4}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-32}
ETP_SIZE=${ETP_SIZE:-1}

# ============================================================
# Training
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-2048}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-2e-6}
DTYPE=${DTYPE:-bfloat16}

BACKEND=megatron
RESUME_MODE=${RESUME_MODE:-disable}

project_name=verl_sft_qwen3_5
exp_name=qwen3_5-${BACKEND}-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}
ckpts_home=${ckpts_home:-~/verl/checkpoints/${project_name}/${exp_name}}
mkdir -p "${ckpts_home}"

# ============================================================
# MTP hyper-parameters
# ============================================================
MTP_ENABLE=${MTP_ENABLE:-True}
MTP_ENABLE_TRAIN=${MTP_ENABLE_TRAIN:-True}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-True}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# ============================================================
# Engine config
# ============================================================
# Key Qwen3.5 settings:
#   engine.use_remove_padding=False   - use BSHD for this recipe
ENGINE_CONFIG="\
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps=10 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    +optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +optim.override_optimizer_config.optimizer_cpu_offload=True \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_mbridge=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=False \
    engine.override_transformer_config.attention_backend=auto \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1"

# ============================================================
# Launch
# ============================================================
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    data.messages_key=messages \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.mtp.enable=${MTP_ENABLE} \
    model.mtp.enable_train=${MTP_ENABLE_TRAIN} \
    model.mtp.detach_encoder=${MTP_DETACH_ENCODER} \
    model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} 
