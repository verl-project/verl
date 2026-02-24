#!/usr/bin/env bash
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

# Note the default MultiturnSFT Dataset requires all the sys/user/assistant in 'data.message_key'
DATASET_DIR=${DATASET_DIR:-~/dataset/rl/gsm8k}
TRAIN_FILES=${DATASET_DIR}/train.parquet
VAL_FILES=${DATASET_DIR}/eval.parquet

project_name=verl_sft_test

RESUME_MODE=disable

MODEL_PATH="XiaomiMiMo/MiMo-7B-RL"
ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-${backend}}

# currently relies on these two commits that is not on master
PYPATH=$HOME/pythonpath
mkdir -p $PYPATH && cd $PYPATH
[ -d Megatron-LM ] || git clone https://github.com/NVIDIA/Megatron-LM -b dev && (cd Megatron-LM; git checkout 23e092f41ec8bc659020e401ddac9576c1cfed7e)
[ -d mbridge ] || git clone https://github.com/ArronHZG/mbridge -b feature/verl_mtp && (cd mbridge; git checkout 6bf2d45a15dc4fb52d2f0c38ff546bee33447d10)
cd -
export PYTHONPATH=$PYTHONPATH:$PYPATH/mbridge:$PYPATH/Megatron-LM


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=${LR} \
    optim.min_lr=${MINLR} \
    optim.lr_warmup_steps=10 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    engine.override_transformer_config.recompute_method=uniform \
    engine.override_transformer_config.recompute_granularity=full \
    engine.override_transformer_config.recompute_num_layers=1 \
    engine.use_dist_checkpointing=False \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    "

ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
echo "Using megatron engine"
exp_name=gsm8k-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-lr-${MINLR}-${LR}

mkdir -p "${ckpts_home}"

$COMMAND \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TRAIN_FILES}" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.max_length=1024 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=prompt \
    data.num_workers=0 \
    model.path=$MODEL_PATH \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    model.trust_remote_code=True \
    model.mtp.enable=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger="['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE}
    