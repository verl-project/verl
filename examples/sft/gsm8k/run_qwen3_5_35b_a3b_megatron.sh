#!/usr/bin/env bash
# Qwen3.5-35B-A3B SFT with Megatron backend, packed sequences (THD format)
#
# This is the counterpart to run_qwen3_5_397b_a17b_megatron.sh, which runs the same
# architecture family in bshd (padded) format. Qwen3.5 uses Gated Delta Net (GDN) linear
# attention, and GDN did not support packed sequences until
# https://github.com/NVIDIA/Megatron-LM/pull/2644 -- merged 2026-04-07 and first shipped
# in a megatron-core release in 0.18.0. On 0.18.0+ THD works, so the padding no longer has
# to be computed.
#
# Requirements:
#   - 16+ GPUs (80GB each, e.g. 2x8 H100/H200/H20) for the parallelism below
#   - megatron-core >= 0.18.0    <-- GDN packed-sequence support; earlier releases
#                                    (0.16.x, 0.17.x) silently lack the THD branch in
#                                    megatron/core/ssm/gated_delta_net.py
#   - Additional packages on top of the base image:
#       pip install --upgrade transformers
#       pip install flash-linear-attention
#
# Data:
#   python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir ~/dataset
#
# THD (packed sequence) settings -- all three are needed:
#   model.use_remove_padding=True    unpad on the model side
#   engine.use_remove_padding=True   thd compute format in the Megatron engine
#   data.use_dynamic_bsz=True        batch by token budget instead of sample count;
#                                    this is what turns unpadding into a throughput win
#   data.max_token_len_per_gpu       per-GPU token budget. The effective micro-batch
#                                    budget is max_token_len_per_gpu * CP_SIZE, so it must
#                                    be >= data.max_length / CP_SIZE for a longest-sample
#                                    micro-batch to fit.
#
# attention_backend is deliberately not set: the default (Megatron-Bridge) path already
# pins AttnBackend.flash. Do not switch this example to the deprecated
# engine.vanilla_mbridge=True path without also setting
#   +engine.override_transformer_config.attention_backend=flash
# There, attention_backend falls back to megatron's `auto`, which prefers cuDNN fused
# attention on Hopper+; cuDNN's SDPA backward has an uninitialized-workspace defect on
# Hopper (fixed in cuDNN 9.18) that makes THD backward produce huge dK/dV and then NaN
# gradients. See https://github.com/NVIDIA/TransformerEngine/issues/2186.
#
# Tested parallelism config (16 GPUs / 2 nodes):
#   TP=2 PP=2 CP=2 EP=8

set -xeuo pipefail

# ============================================================
# Distributed
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}

# ============================================================
# Data
# ============================================================
DATASET_DIR=${DATASET_DIR:-~/dataset}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}

# ============================================================
# Model
# ============================================================
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}

# ============================================================
# Parallelism
# ============================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-2}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

# ============================================================
# Training
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
# Token budget per GPU per micro-batch. Effective budget is this * CP_SIZE.
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-2048}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-2e-6}
DTYPE=${DTYPE:-bfloat16}

BACKEND=megatron
RESUME_MODE=${RESUME_MODE:-disable}

project_name=verl_sft_qwen3_5
exp_name=qwen3_5-35b-a3b-${BACKEND}-thd-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}
ckpts_home=${ckpts_home:-~/verl/checkpoints/${project_name}/${exp_name}}
mkdir -p "${ckpts_home}"

# ============================================================
# Engine config
# ============================================================
# Key Qwen3.5 THD settings:
#   engine.use_remove_padding=True   - thd compute format (needs megatron-core >= 0.18.0)
#   attention_backend unset          - default bridge path already pins flash (see header)
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
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=True \
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
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    data.messages_key=messages \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=True \
    model.trust_remote_code=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE}
