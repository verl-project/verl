#!/usr/bin/env bash
# MiMo-7B MTP SFT with the CUDA dependencies pinned in uv.lock:
#   Python 3.12, Megatron-Core 0.19.2, Megatron-Bridge 0.6.2.
# Bridge 0.6.2 includes MiMoForCausalLM registration and MTP weight mappings.
#   uv sync --frozen --extra megatron
#   source .venv/bin/activate
# Prepare data with examples/data_preprocess/gsm8k_multiturn_sft.py.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
SP_SIZE=${SP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
LR="1e-5"
MINLR="1e-6"

export VERL_SFT_LOGGING_LEVEL=INFO

backend=${BACKEND:-megatron}

TENSORBOARD_DIR=~/tensorboard

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-1}
RANK=${RANK:-0}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
read -r -a ENTRYPOINT_ARGS <<< "$ENTRYPOINT"

# The preprocessing script writes complete conversations to the messages column.
DATASET_DIR=${DATASET_DIR:-~/data/gsm8k_sft}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/test.parquet}
MESSAGES_KEY=${MESSAGES_KEY:-messages}

project_name=verl_sft_test

RESUME_MODE=disable

MODEL_PATH=${MODEL_PATH:-XiaomiMiMo/MiMo-7B-RL}
ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-${backend}}

MEGATRON_ENGINE_CONFIG=(
    engine=${backend}
    optim=${backend}
    optim.lr=${LR}
    optim.min_lr=${MINLR}
    optim.lr_warmup_steps=10
    optim.weight_decay=0.1
    optim.betas='[0.9,0.95]'
    optim.clip_grad=1.0
    optim.lr_warmup_init=0
    optim.lr_decay_style=cosine
    engine.override_transformer_config.recompute_method=uniform
    engine.override_transformer_config.recompute_granularity=full
    engine.override_transformer_config.recompute_num_layers=1
    engine.use_dist_checkpointing=False
    engine.tensor_model_parallel_size=${TP_SIZE}
    engine.pipeline_model_parallel_size=${PP_SIZE}
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE}
    engine.context_parallel_size=${CP_SIZE}
    engine.use_mbridge=True
    engine.use_remove_padding=${USE_REMOVE_PADDING}
)

echo "Using megatron engine"
exp_name=gsm8k-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-lr-${MINLR}-${LR}

mkdir -p "${ckpts_home}"

torchrun --nnodes="${NNODES}" --nproc-per-node="${NUM_GPUS}" \
    --node-rank="${RANK}" --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${ENTRYPOINT_ARGS[@]}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.max_length=1024 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key="${MESSAGES_KEY}" \
    data.num_workers=0 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    model.trust_remote_code=True \
    model.mtp.enable=True \
    model.mtp.enable_train=True \
    "${MEGATRON_ENGINE_CONFIG[@]}" \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    "$@"
