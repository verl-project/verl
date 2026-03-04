#!/usr/bin/env bash
set -xeuo pipefail
################################################### environment ###################################################
### # 1. use docker image `verlai/verl:vllm015.dev`` and install correct dependencies:
# pip install nvidia-modelopt
# MAX_JOBS=32 pip install git+https://github.com/Dao-AILab/causal-conv1d.git --no-build-isolation --no-cache-dir
# MAX_JOBS=32 pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-cache-dir
# pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge 
# pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_dev_r0.16.0
# unset ROCR_VISIBLE_DEVICES
# unset PYTORCH_CUDA_ALLOC_CONF

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

TRAIN_FILES=$HOME/data/gsm8k/train.parquet
VAL_FILES=$HOME/data/gsm8k/eval.parquet

backend=${BACKEND:-megatron}

project_name=verl_sft_gsm8k

RESUME_MODE=auto
MODEL_NAME=${MODEL_NAME:-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}

TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

DTYPE=${DTYPE:-"bfloat16"}


MEGATRON_ENGINE_CONFIG=(
    engine=${backend}
    optim=${backend}
    optim.lr=2e-5
    optim.lr_warmup_steps=5
    optim.weight_decay=0.1
    optim.betas="[0.9,0.95]"
    optim.clip_grad=1.0
    optim.lr_warmup_init=0
    optim.lr_decay_style=cosine
    optim.min_lr=2e-6
    +optim.override_optimizer_config.optimizer_offload_fraction=1
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +optim.override_optimizer_config.use_precision_aware_optimizer=True
    +optim.override_optimizer_config.optimizer_cpu_offload=True
    engine.tensor_model_parallel_size=${TP_SIZE}
    engine.pipeline_model_parallel_size=${PP_SIZE}
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE}
    engine.context_parallel_size=${CP_SIZE}
    engine.use_mbridge=True
    engine.dtype=${DTYPE}
    engine.vanilla_mbridge=False
    engine.expert_model_parallel_size=${EP_SIZE}
    engine.expert_tensor_parallel_size=${ETP_SIZE}
    engine.override_transformer_config.attention_backend=auto
    +engine.override_transformer_config.recompute_method=uniform
    +engine.override_transformer_config.recompute_granularity=full
    +engine.override_transformer_config.recompute_num_layers=1
)

ENGINE_CONFIG="${MEGATRON_ENGINE_CONFIG[@]}"
echo "Using megatron engine"
exp_name=${MODEL_NAME}-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-megatron-20260210

torchrun --nnodes=1 --nproc_per_node=8 ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=96 \
    data.max_length=2048 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    model.path=$MODEL_PATH \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    model.trust_remote_code=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=10 \
    checkpoint.save_contents=[model,optimizer,extra]