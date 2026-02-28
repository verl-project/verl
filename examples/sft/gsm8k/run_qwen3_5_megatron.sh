#!/usr/bin/env bash
# Qwen3.5-397B-A17B SFT with Megatron backend + mbridge
#
# Requirements:
#   - 128+ GPUs (80GB each, e.g. 16x8 H100/H200)
#   - Docker: verlai/verl:vllm015 (or equivalent)
#   - Additional packages on top of the base image:
#       pip install --upgrade transformers
#       pip install flash-linear-attention
#       pip install -U git+https://github.com/ISEEKYAN/mbridge.git
#   - Megatron-LM dev branch with Qwen3.5 GDN support
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - engine.use_remove_padding=False  (forces bshd compute format)
#     - model.use_remove_padding=True    (keeps NestedTensor in data pipeline)
#     - data.use_dynamic_bsz=False       (required for bshd mode)
#
#   Once https://github.com/NVIDIA/Megatron-LM/pull/2644 is merged, THD
#   format will be supported and engine.use_remove_padding can be set to True
#   for better performance.
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
# Engine config
# ============================================================
# Key Qwen3.5 settings:
#   engine.use_remove_padding=False   - GDN requires bshd format (no THD)
#   engine.vanilla_mbridge=True       - use mbridge (not megatron-bridge)
ENGINE_CONFIG="\
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps=10 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
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
    engine.vanilla_mbridge=True \
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
    model.use_remove_padding=True \
    model.trust_remote_code=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger="['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE}
