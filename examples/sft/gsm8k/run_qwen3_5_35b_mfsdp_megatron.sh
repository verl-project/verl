#!/usr/bin/env bash
# Qwen3.5-35B-A3B SFT with Megatron-FSDP on 32 GPUs.
# It defaults to a 32-GPU EP8/TP2/CP2/GBS64 configuration.
#
# Requirements:
#   - Docker: verlai/verl:vllm018.dev1 (or equivalent)
#   - Runtime Python packages, if the container does not include them:
#       python3 -m pip install -q "nvidia-modelopt[torch]>=0.37.0"
#       python3 -m pip install -q "flash-linear-attention==0.4.1"
#       python3 -m pip install -U "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
#       python3 -m pip install -U "git+https://github.com/NVIDIA/Megatron-LM.git@dev"
#
# Note: the required Megatron-FSDP changes are available in Megatron-LM dev
# (https://github.com/NVIDIA/Megatron-LM/pull/4799) and Megatron-Bridge main
# (https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3746).
set -xeuo pipefail

# ============================================================
# Distributed
# ============================================================
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}

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
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

# ============================================================
# Training
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-8}

project_name=${PROJECT_NAME:-verl_sft_qwen35_mfsdp_example}
exp_name=${EXP_NAME:-qwen3_5_35b_a3b-mfsdp-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}-gbs${TRAIN_BATCH_SIZE}-seq${MAX_LENGTH}}

# ============================================================
# Parameter arrays
# ============================================================
DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}
    data.max_length=${MAX_LENGTH}
    data.pad_mode=no_padding
    data.truncation=error
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu=${MAX_LENGTH}
    data.messages_key=messages
    data.ignore_input_ids_mismatch=True
    data.num_workers=0
)

MODEL=(
    model=hf_model
    model.path="${MODEL_PATH}"
    model.use_remove_padding=True
    model.trust_remote_code=True
    model.mtp.enable=False
)

ENGINE=(
    engine=megatron
    optim=megatron
    +optim.override_optimizer_config.use_precision_aware_optimizer=True
    engine.tensor_model_parallel_size=${TP_SIZE}
    engine.pipeline_model_parallel_size=${PP_SIZE}
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE}
    engine.context_parallel_size=${CP_SIZE}
    engine.expert_model_parallel_size=${EP_SIZE}
    engine.expert_tensor_parallel_size=${ETP_SIZE}
    engine.use_mbridge=True
    engine.vanilla_mbridge=False
    engine.use_megatron_fsdp=True
    engine.use_remove_padding=True
    +engine.override_ddp_config.megatron_fsdp_use_decoupled_grad=True
    +engine.override_transformer_config.moe_router_dtype=fp32
    +engine.override_transformer_config.moe_token_dispatcher_type=flex
    +engine.override_transformer_config.moe_flex_dispatcher_backend=deepep
    +engine.override_transformer_config.moe_grouped_gemm=True
    +engine.override_transformer_config.moe_permute_fusion=True
    +engine.override_transformer_config.moe_router_fusion=True
    engine.override_transformer_config.attention_backend=flash
    engine.override_transformer_config.recompute_method=uniform
    engine.override_transformer_config.recompute_granularity=full
    engine.override_transformer_config.recompute_num_layers=1
    +engine.override_transformer_config.mtp_num_layers=null
    +engine.override_transformer_config.calculate_per_token_loss=True
    +engine.override_transformer_config.gradient_accumulation_fusion=False
)

TRAINER=(
    trainer.test_freq=-1
    trainer.save_freq=-1
    "trainer.logger=['console']"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS}
    trainer.resume_mode=disable
)

# ============================================================
# Launch
# ============================================================
CMD=(
    torchrun
    --nproc_per_node=${NUM_GPUS}
    --nnodes=${NNODES}
    --node_rank=${NODE_RANK}
    --master_addr=${MASTER_ADDR}
    --master_port=${MASTER_PORT}
    -m
    verl.trainer.sft_trainer
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ENGINE[@]}"
    "${TRAINER[@]}"
    'checkpoint.save_contents=["model","optimizer"]'
    'checkpoint.load_contents=["model","optimizer"]'
)

"${CMD[@]}"
