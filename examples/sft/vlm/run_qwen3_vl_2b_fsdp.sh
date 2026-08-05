#!/usr/bin/env bash
# python examples/data_preprocess/pokemon.py
set -xeuo pipefail

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

TRAIN_FILES=${HOME}/data/pokemon-gpt4o-captions/train.parquet

backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=auto
MODEL_ID=${HDFS_ROOT}/model/Qwen3-VL-2B-Instruct
# MODEL_ID=${HDFS_ROOT}/model/Qwen3-VL-30B-A3B-Instruct

SP_SIZE=${SP_SIZE:-2}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=2e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=pokemon-qwen3-2b-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp-1202a1
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=pokemon-qwen3-2b-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-megatron-1202a1
fi

CKPT_HOME=${CKPT_HOME:-$HOME/open_verl/sft/${project_name}/${exp_name}}
mkdir -p "${CKPT_HOME}"

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS:-8} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=96 \
    data.max_length=2048 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=65536 \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=4000 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents=[model,optimizer,extra]