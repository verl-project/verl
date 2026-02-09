#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}

mode=${mode:-spmd}

if [ "$mode" = "spmd" ]; then
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
  COMMAND="torchrun --standalone --nnodes=${NNODES:-1} --nproc-per-node=${NUM_GPUS:-1} ${ENTRYPOINT}"
else
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer_ray"}
  COMMAND="python ${ENTRYPOINT} trainer.nnodes=${NNODES:-1} trainer.n_gpus_per_node=${NUM_GPUS:-1}"
fi

DATASET_DIR=${DATASET_DIR:-~/data/gsm8k_sft}
TRAIN_FILES=${DATASET_DIR}/train.parquet
VAL_FILES=${DATASET_DIR}/test.parquet

backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=disable

ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-${backend}}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NUM_GPUS}}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

VEOMNI_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_min=1e-6 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_parallel_size=${SP_SIZE} \
    engine.data_parallel_mode=${FSDP_STRATEGY} \
    engine.data_parallel_size=${FSDP_SIZE}"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    +engine.override_transformer_config.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=gsm8k-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}-mode-${mode}
elif [ "$backend" = "veomni" ]; then
    ENGINE_CONFIG="$VEOMNI_ENGINE_CONFIG"
    echo "Using veomni engine"
    exp_name=gsm8k-${backend}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}-mode-${mode}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=gsm8k-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}-mode-${mode}
fi

mkdir -p "${ckpts_home}"

$COMMAND \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=128 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=messages \
    model.path=$MODEL_PATH \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger=['console','file'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps=2 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \

    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    # trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    # trainer.max_ckpt_to_keep=1 \
    
rm -rf "${ckpts_home:?}/*"