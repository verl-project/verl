#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
MODEL_ID=${MODEL_ID:-/llm-align/open_models/Qwen3/Qwen3-8B-Base}
TRAIN_FILES=${TRAIN_FILES:-/llm-align/liuchonghan/verl_parquet_merged/all_merged_to_eng.parquet}

PROJECT_NAME=${PROJECT_NAME:-nips2026_qwen3_8b_base_sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-all_merged_to_eng}
CKPT_HOME=${CKPT_HOME:-/llm-align/liuchonghan/ckpt_verl/sft/${PROJECT_NAME}/${EXPERIMENT_NAME}}
RESUME_MODE=${RESUME_MODE:-disable}

BACKEND=${BACKEND:-megatron}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
IGNORE_INPUT_IDS_MISMATCH=${IGNORE_INPUT_IDS_MISMATCH:-True}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23457}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
FSDP_SIZE=${FSDP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}

FSDP_ENGINE_CONFIG="
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

MEGATRON_ENGINE_CONFIG="
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True"

if [ "$BACKEND" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
fi

mkdir -p "${CKPT_HOME}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${CKPT_HOME}/wandb}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONPATH=${PYTHONPATH:-}:/llm-align/liuchonghan/verl_lao
mkdir -p "${WANDB_DIR}"

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc-per-node=${NPROC_PER_NODE} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=${PAD_MODE} \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=${IGNORE_INPUT_IDS_MISMATCH} \
    model.path=${MODEL_ID} \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    model.enable_gradient_checkpointing=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=3 \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]'
