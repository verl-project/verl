#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k_sft/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k_sft/test.parquet}

SP_SIZE=${SP_SIZE:-1}
LIGER=${LIGER:-False}
MULTITURN=${MULTITURN:-False}
LORA_RANK=${LORA_RANK:-0}
RM_PAD=${RM_PAD:-True}

TOTAL_TRAIN_STEP=${TOTAL_TRAIN_STEP:-1}
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-1}

micro_bsz=2
NUM_GPUS=8

project_name="verl-test"
exp_name="$(basename "${MODEL_ID,,}")-sft-minimal"
ckpts_home=${ckpts_home:-$HOME/${project_name}/${exp_name}}

mkdir -p "${ckpts_home}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${micro_bsz} \
    optim.lr=1e-4 \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    model.path="${MODEL_PATH}" \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.use_liger="${LIGER}" \
    model.use_remove_padding="${RM_PAD}" \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    trainer.save_freq=${SAVE_FREQ} \
    checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.logger=['console'] $@

rm -rf "${ckpts_home:?}/*"