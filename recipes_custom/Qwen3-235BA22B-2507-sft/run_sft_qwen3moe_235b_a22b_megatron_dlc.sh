#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
TRAIN_FILES=${TRAIN_FILES:-/mnt/data/liuchonghan/235b_dataset/merged_sft_with_messages.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
backend=${BACKEND:-megatron}
project_name=verl_sft_235ba22b_2507
RESUME_MODE=disable
MODEL_ID=${MODEL_ID:-/mnt/data/liuchonghan/Qwen3-235B-A22B-Instruct-2507}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}

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
    exp_name=nvidia-qwen3-235b-a22b-moe-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=nvidia-qwen3-235b-a22b-moe-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}
fi

CKPT_HOME=${CKPT_HOME:-/mnt/data/liuchonghan/ckpt_verl/sft/${project_name}/${exp_name}}
NNODES=${WORLD_SIZE:-16}           
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${PYTHONPATH:-}:/mnt/data/liuchonghan/verl_lao

if [[ -z "${TOTAL_TRAINING_STEPS}" ]]; then
    # Megatron's OptimizerParamScheduler asserts `lr_decay_steps > 0`.
    # VERL SFT derives total steps from `len(train_dataloader)`, which can be 0/unknown with some samplers
    # (e.g. dynamic-bsz). Provide a safe positive estimate based on parquet row count.
    TOTAL_TRAINING_STEPS=$(python3 - <<'PY'
import math
import os

train_files = os.environ.get("TRAIN_FILES", "")
batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "256"))
epochs = int(os.environ.get("TOTAL_EPOCHS", "1"))

rows = None
try:
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(train_files).metadata.num_rows
except Exception:
    rows = None

if rows is None:
    steps = 1000 * max(1, epochs)
else:
    steps_per_epoch = max(1, math.ceil(rows / max(1, batch_size)))
    steps = steps_per_epoch * max(1, epochs)

print(steps)
PY
)
fi

echo ">>> SFT steps: total_epochs=${TOTAL_EPOCHS}, train_batch_size=${TRAIN_BATCH_SIZE}, total_training_steps=${TOTAL_TRAINING_STEPS}"

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc-per-node=8 \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_length=1024 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=10240 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    +model.override_config.router_dtype="float16" \
    model.enable_gradient_checkpointing=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=2000 \
    'trainer.logger=[console]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=2 \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]'
