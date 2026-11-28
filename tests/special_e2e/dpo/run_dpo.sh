#!/usr/bin/env bash
# One short DPO run on the GSM8K preference pairs from examples/data_preprocess/gsm8k_dpo.py.
# Knobs: NUM_GPUS, SP_SIZE, FSDP_SIZE, FSDP_STRATEGY, LORA_RANK, REFERENCE_FREE, USE_DYNAMIC_BSZ,
#        LOSS_TYPE, TOTAL_TRAIN_STEP, RESUME_MODE, MODEL_ID, MODEL_PATH, DATASET_DIR.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.dpo_trainer"}
COMMAND="python -m torch.distributed.run --standalone --nnodes=${NNODES:-1} --nproc-per-node=${NUM_GPUS} ${ENTRYPOINT}"

DATASET_DIR=${DATASET_DIR:-~/data/gsm8k_dpo}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
LORA_RANK=${LORA_RANK:-0}
REFERENCE_FREE=${REFERENCE_FREE:-False}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
LOSS_TYPE=${LOSS_TYPE:-sigmoid}
TOTAL_TRAIN_STEP=${TOTAL_TRAIN_STEP:-2}
RESUME_MODE=${RESUME_MODE:-disable}

project_name=verl_dpo_test
exp_name=gsm8k-dpo-${FSDP_STRATEGY}-gpu${NUM_GPUS}-sp${SP_SIZE}-lora${LORA_RANK}-reffree-${REFERENCE_FREE}-dynbsz-${USE_DYNAMIC_BSZ}-${LOSS_TYPE}
ckpts_home=${ckpts_home:-~/verl/test/gsm8k-dpo/${exp_name}}

mkdir -p "${ckpts_home}"
if [ "${RESUME_MODE}" = "disable" ]; then
    rm -rf -- "${ckpts_home:?}/"*
fi

$COMMAND \
    data.train_files="${DATASET_DIR}/train.parquet" \
    data.val_files="${DATASET_DIR}/test.parquet" \
    data.train_max_samples=512 \
    data.val_max_samples=128 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    data.max_token_len_per_gpu=4096 \
    data.max_length=1024 \
    data.truncation=right \
    data.shuffle=False \
    data.chat_template_key=default \
    data.dataloader_num_workers=2 \
    model.path="${MODEL_PATH}" \
    model.lora_rank=${LORA_RANK} \
    model.lora_alpha=32 \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE} \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    optim.lr=1e-6 \
    trainer.loss_type=${LOSS_TYPE} \
    trainer.beta=0.1 \
    trainer.reference_free=${REFERENCE_FREE} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    trainer.test_freq=${TOTAL_TRAIN_STEP} \
    trainer.save_freq=${TOTAL_TRAIN_STEP} \
    trainer.logger=['console','file'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.n_gpus_per_node=${NUM_GPUS}
