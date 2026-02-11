#!/usr/bin/env bash
# python examples/data_preprocess/pokemon.py
set -xeuo pipefail

NUM_GPUS=8
NNODES=1
MODE=spmd
HOME=$pwd

if [ "$MODE" = "spmd" ]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer"
    COMMAND="torchrun --standalone --nnodes=${NNODES} --nproc-per-node=${NUM_GPUS} ${ENTRYPOINT}"
else
    ENTRYPOINT="-m verl.trainer.sft_trainer_ray"
    COMMAND="python ${ENTRYPOINT} trainer.nnodes=${NNODES} trainer.n_gpus_per_node=${NUM_GPUS}"
fi

MODEL_ID=Qwen3_VL-30B-MOE
MODEL_PATH=${HOME}/Qwen3-VL-30B-A3B-Instruct
TRAIN_FILES=${HOME}/data/pokemon-gpt4o-captions/train.parquet
VAL_FILES=${HOME}/data/pokemon-gpt4o-captions/test.parquet

SP_SIZE=${SP_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-8}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-2}
DATA_PARALLEL_MODE=${DATA_PARALLEL_MODE:-fsdp2}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=no_padding
USE_REMOVE_PADDING=True

PROJECT_NAME=qwen3vl-30b-moe-sft
EXPERIMENT_NAME=veomni-sp${SP_SIZE}-fsdp${DATA_PARALLEL_SIZE}
CKPT_HOME=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

VEOMNI_CONFIG="\
    engine=veomni \
    optim=veomni \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_min=1e-6 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_parallel_size=${SP_SIZE} \
    engine.data_parallel_mode=${DATA_PARALLEL_MODE} \
    engine.data_parallel_size=${DATA_PARALLEL_SIZE} \
    engine.expert_parallel_size=${EXPERT_PARALLEL_SIZE} \
    engine.enable_full_shard=True \
    engine.moe_implementation=fused \
    engine.attn_implementation=flash_attention_2"

mkdir -p "${CKPT_HOME}"

$COMMAND \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=4096 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    data.messages_key=messages \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=False \
    ${VEOMNI_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=1000 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=disable \
    trainer.max_ckpt_to_keep=3 $@ 
