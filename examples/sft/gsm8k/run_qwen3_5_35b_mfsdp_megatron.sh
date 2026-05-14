#!/usr/bin/env bash
# Qwen3.5-35B-A3B SFT with Megatron-FSDP on 32 GPUs.
# It defaults to a 32-GPU EP8/TP2/CP2/GBS64 configuration.
#
# Requirements:
#   - Docker: verlai/verl:vllm018.dev1 (or equivalent)
#   - Runtime Python packages, if the container does not include them:
#       python3 -m pip install -q "nvidia-modelopt[torch]>=0.37.0"
#       python3 -m pip install -q "flash-linear-attention==0.4.1"
#   - Megatron-Bridge with https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3746
#   - Megatron-LM dev branch with https://github.com/NVIDIA/Megatron-LM/pull/4799
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR=${VERL_DIR:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
ROOT=${ROOT:-"$(cd "${VERL_DIR}/.." && pwd)"}

# ============================================================
# Distributed
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-${SLURM_NNODES:-4}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_PORT=${MASTER_PORT:-29500}

if [ -z "${MASTER_ADDR:-}" ] || { [ "${NNODES}" -gt 1 ] && { [ "${MASTER_ADDR}" = "localhost" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ]; }; }; then
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
        MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        NODELIST="${SLURM_JOB_NODELIST}"
        if [[ "${NODELIST}" == *"["* ]]; then
            NODE_PREFIX="${NODELIST%%[*}"
            NODE_RANGE="${NODELIST#*[}"
            NODE_RANGE="${NODE_RANGE%%]*}"
            FIRST_NODE="${NODE_RANGE%%,*}"
            FIRST_NODE="${FIRST_NODE%%-*}"
            MASTER_ADDR="${NODE_PREFIX}${FIRST_NODE}"
        else
            MASTER_ADDR="${NODELIST%%,*}"
        fi
    else
        MASTER_ADDR=localhost
    fi
fi

# ============================================================
# Data
# ============================================================
DATASET_DIR=${DATASET_DIR:-~/data/gsm8k_sft}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/test.parquet}

# ============================================================
# Model
# ============================================================
MODEL_PATH=${MODEL_PATH:-/root/models/hf/Qwen3.5-35B-A3B}

# ============================================================
# Parallelism
# ============================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-2}
CP_COMM_TYPE=${CP_COMM_TYPE:-p2p}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

# ============================================================
# Training
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-8}
LR=${LR:-1.2e-4}
MIN_LR=${MIN_LR:-1.2e-5}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-100}
LR_DECAY_STEPS=${LR_DECAY_STEPS:-2000}
DTYPE=${DTYPE:-bfloat16}
RESUME_MODE=${RESUME_MODE:-disable}

project_name=${PROJECT_NAME:-verl_sft_qwen35_mfsdp_example}
exp_name=${EXP_NAME:-qwen3_5_35b_a3b-mfsdp-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}-gbs${TRAIN_BATCH_SIZE}-seq${MAX_LENGTH}}
ckpts_home=${ckpts_home:-${ROOT}/checkpoints/${project_name}/${exp_name}}
mkdir -p "${ckpts_home}"

# ============================================================
# MTP hyper-parameters
# ============================================================
MTP_ENABLE=${MTP_ENABLE:-False}
MTP_ENABLE_TRAIN=${MTP_ENABLE_TRAIN:-False}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-null}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-True}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# ============================================================
# Environment
# ============================================================
export HYDRA_FULL_ERROR=1
export HF_HOME=${HF_HOME:-/root/models}
export PYTHONPATH="${VERL_DIR}:${ROOT}/Megatron-LM:${ROOT}/Megatron-Bridge/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_IB_SL=${NCCL_IB_SL:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
unset ROCR_VISIBLE_DEVICES

cd "${VERL_DIR}"

# ============================================================
# Engine config
# ============================================================
ENGINE_CONFIG="\
    engine=megatron \
    optim=megatron \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    optim.lr_decay_steps=${LR_DECAY_STEPS} \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    +optim.override_optimizer_config.optimizer_offload_fraction=0 \
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +optim.override_optimizer_config.optimizer_cpu_offload=False \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=False \
    engine.use_megatron_fsdp=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=False \
    +engine.override_transformer_config.cp_comm_type=${CP_COMM_TYPE} \
    engine.override_transformer_config.attention_backend=auto \
    engine.override_transformer_config.recompute_method=uniform \
    engine.override_transformer_config.recompute_granularity=full \
    engine.override_transformer_config.recompute_num_layers=1 \
    +engine.override_transformer_config.mtp_num_layers=${MTP_NUM_LAYERS} \
    +engine.override_transformer_config.calculate_per_token_loss=True \
    +engine.override_transformer_config.gradient_accumulation_fusion=False"

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
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=0 \
    model=hf_model \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.mtp.enable=${MTP_ENABLE} \
    model.mtp.enable_train=${MTP_ENABLE_TRAIN} \
    model.mtp.detach_encoder=${MTP_DETACH_ENCODER} \
    model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "trainer.logger=['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.default_local_dir="${ckpts_home}" \
    'checkpoint.save_contents=["model","optimizer"]' \
    'checkpoint.load_contents=["model","optimizer"]' \
    trainer.resume_mode=${RESUME_MODE}
