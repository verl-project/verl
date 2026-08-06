#!/usr/bin/env bash
set -xeuo pipefail

########################### Quick Config ###########################

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
SAVE_PATH=${SAVE_PATH:-/root/checkpoints/Qwen2.5-Math-7B}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k_sft/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k_sft/test.parquet}

NPROC=${NPROC:-8}
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-1}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES

########################### Parameter Arrays ###########################

DATA=(
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    data.messages_key=messages
    data.train_batch_size=8
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu=1024
    data.pad_mode=no_padding
    data.truncation=error
)

MODEL=(
    model=hf_model
    "model.path=${MODEL_PATH}"
    model.trust_remote_code=True
    model.use_remove_padding=true
)

OPTIM=(
    optim=megatron
    optim.lr=1e-5
    optim.lr_warmup_steps_ratio=0.2
    optim.weight_decay=0.1
    "optim.betas=[0.9,0.95]"
    optim.clip_grad=1.0
    optim.lr_warmup_init=0
    optim.lr_decay_style=cosine
    optim.min_lr=1e-6
)

ENGINE=(
    engine=megatron
    engine.tensor_model_parallel_size=${TP}
    engine.pipeline_model_parallel_size=${PP}
    engine.expert_model_parallel_size=${EP}
    engine.use_mbridge=True
    engine.vanilla_mbridge=False
    engine.use_megatron_fsdp=True
    +engine.override_transformer_config.gradient_accumulation_fusion=False
)

TRAINER=(
    "trainer.default_local_dir=${SAVE_PATH}"
    trainer.project_name=gsm8k-sft
    trainer.experiment_name=SFT-qwen2.5-7b-mfsdp
    trainer.logger='["console","wandb","file"]'
    trainer.total_epochs=4
)

########################### Launch ###########################

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    -m verl.trainer.sft_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${OPTIM[@]}" \
    "${ENGINE[@]}" \
    "${TRAINER[@]}" \
    "$@"
