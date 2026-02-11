#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
TRAIN_FILES=${TRAIN_FILES:-/mnt/data/liuchonghan/translate_parquet/train_data.parquet}
backend=${BACKEND:-megatron}
project_name=verl_sft_translate_0109
RESUME_MODE=disable
MODEL_ID=${MODEL_ID:-/mnt/data/liuchonghan/Qwen3-30B-A3B-Instruct-2507}

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-64}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="
    engine=${backend} \
    optim=${backend} \
    optim.lr=5e-6 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

MEGATRON_ENGINE_CONFIG="
    engine=${backend} \
    optim=${backend} \
    optim.lr=6e-6 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=6e-7 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=nvidia-qwen3-30b-moe-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=nvidia-qwen3-30b-moe-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}
fi

CKPT_HOME=${CKPT_HOME:-/mnt/data/liuchonghan/ckpt_verl/sft/${project_name}/${exp_name}}
NNODES=${WORLD_SIZE:-8}           
NODE_RANK=${RANK:-0}              
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} 
MASTER_PORT=${MASTER_PORT:-23457} 

echo ">>> 节点信息: RANK $NODE_RANK / WORLD_SIZE $NNODES"
echo ">>> 通信信息: MASTER $MASTER_ADDR : $MASTER_PORT"

if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "${CKPT_HOME}"
fi

export WANDB_MODE=offline
export NCCL_DEBUG=WARN
export PYTHONPATH=${PYTHONPATH:-}:/mnt/data/liuchonghan/verl

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc-per-node=8 \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=512 \
    data.max_length=8192 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=49152 \
    data.messages_key=messages \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    +model.override_config.output_router_logits=True \
    +model.override_config.router_dtype="float32" \
    model.enable_gradient_checkpointing=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=5000 \
    'trainer.logger=[console]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=3 \
    'checkpoint.save_contents=[model,optimizer,extra]'