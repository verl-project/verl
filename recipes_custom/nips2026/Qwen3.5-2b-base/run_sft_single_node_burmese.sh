#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${WORLD_SIZE:-${NNODES:-1}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-8888}

RAW_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_ADDR=$(python3 -c "import socket; print(socket.getaddrinfo('${RAW_MASTER_ADDR}', None, socket.AF_INET)[0][4][0])" 2>/dev/null || echo "${RAW_MASTER_ADDR}")

MODEL_ID=${MODEL_ID:-/llm-align/open_models/Qwen3.5/Qwen3.5-2b-Base}
PARQUET_DIR=${PARQUET_DIR:-/llm-align/liuchonghan/ins_dataset/ins_dataset/parquet_by_language}
TRAIN_FILES=${TRAIN_FILES:-"[${PARQUET_DIR}/burmese__to__chinese.parquet,${PARQUET_DIR}/burmese__to__english.parquet,${PARQUET_DIR}/burmese__to__filipino.parquet,${PARQUET_DIR}/burmese__to__indonesian.parquet,${PARQUET_DIR}/burmese__to__khmer.parquet,${PARQUET_DIR}/burmese__to__lao.parquet,${PARQUET_DIR}/burmese__to__malay.parquet,${PARQUET_DIR}/burmese__to__tamil.parquet,${PARQUET_DIR}/burmese__to__thai.parquet,${PARQUET_DIR}/burmese__to__vietnamese.parquet]"}

PROJECT_NAME=${PROJECT_NAME:-nips2026_qwen3_5_2b_base}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-burmese_to_all_parquet_by_language}
CKPT_HOME=${CKPT_HOME:-/llm-align/liuchonghan/ckpt_verl/sft/${PROJECT_NAME}/${EXPERIMENT_NAME}}
RESUME_MODE=${RESUME_MODE:-disable}

BACKEND=${BACKEND:-megatron}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
PAD_MODE=${PAD_MODE:-no_padding}
TRUNCATION=${TRUNCATION:-right}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
DTYPE=${DTYPE:-bfloat16}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
ETP_SIZE=${ETP_SIZE:-1}

MEGATRON_ENGINE_CONFIG="
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps=20 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    +optim.override_optimizer_config.optimizer_offload_fraction=0 \
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +optim.override_optimizer_config.optimizer_cpu_offload=False \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=False \
    engine.override_transformer_config.attention_backend=auto \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1"

mkdir -p "${CKPT_HOME}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${CKPT_HOME}/wandb}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export PYTHONPATH=${PYTHONPATH:-}:/llm-align/liuchonghan/verl_lao
mkdir -p "${WANDB_DIR}"

echo ">>> 数据文件: ${TRAIN_FILES}, total_epochs=${TOTAL_EPOCHS}"
echo ">>> 节点信息: RANK ${NODE_RANK} / WORLD_SIZE ${NNODES}"
echo ">>> 通信信息: MASTER ${MASTER_ADDR} : ${MASTER_PORT}"

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${CKPT_HOME}"
fi

if [ "${PAD_MODE}" != "no_padding" ]; then
    echo "ERROR: PAD_MODE must be no_padding for Qwen3.5 megatron bshd path."
    exit 1
fi

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=${PAD_MODE} \
    data.truncation=${TRUNCATION} \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    data.num_workers=${NUM_WORKERS} \
    data.messages_key=messages \
    model.path=${MODEL_ID} \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    ${MEGATRON_ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.max_ckpt_to_keep=3 \
    trainer.logger="['console']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    'checkpoint.save_contents=[model,optimizer,extra,hf_model]'
