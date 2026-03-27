#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${WORLD_SIZE:-${NNODES:-2}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-8888}

RAW_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_ADDR=$(python3 -c "import socket; print(socket.getaddrinfo('${RAW_MASTER_ADDR}', None, socket.AF_INET)[0][4][0])" 2>/dev/null || echo "${RAW_MASTER_ADDR}")

TRAIN_FILES=${TRAIN_FILES:-"[/llm-align/liuchonghan/ins_dataset/ins_dataset/dayuzhong/lowest_metricx_train_selected.cleaned.messages.parquet]"}

MODEL_PATH=${MODEL_PATH:-/llm-align/open_models/Qwen3.5/Qwen3.5-35B-A3B}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
PAD_MODE=${PAD_MODE:-no_padding}
TRUNCATION=${TRUNCATION:-right}
NUM_WORKERS=${NUM_WORKERS:-1}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
DTYPE=${DTYPE:-bfloat16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
OUTPUT_ROUTER_LOGITS=${OUTPUT_ROUTER_LOGITS:-True}
ROUTER_DTYPE=${ROUTER_DTYPE:-float32}
MOE_ROUTER_LOAD_BALANCING_TYPE=${MOE_ROUTER_LOAD_BALANCING_TYPE:-}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-}
MOE_Z_LOSS_COEFF=${MOE_Z_LOSS_COEFF:-}

echo ">>> 数据文件: ${TRAIN_FILES}, total_epochs=${TOTAL_EPOCHS}"

BACKEND=megatron
RESUME_MODE=${RESUME_MODE:-disable}

project_name=${PROJECT_NAME:-verl_sft_qwen3_5_35b_a3b}
exp_name=${EXP_NAME:-qwen3_5_35b_a3b_megatron_translate_0326-${BACKEND}-tp${TP_SIZE}-pp${PP_SIZE}-ep${EP_SIZE}-etp${ETP_SIZE}-cp${CP_SIZE}}
ckpts_home=${ckpts_home:-/llm-align/liuchonghan/ckpt_verl/sft/${project_name}/${exp_name}}

echo ">>> 节点信息: RANK ${NODE_RANK} / WORLD_SIZE ${NNODES}"
echo ">>> 通信信息: MASTER ${MASTER_ADDR} : ${MASTER_PORT}"

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${ckpts_home}"
fi

# Qwen3.5 GDN + megatron bshd path currently requires no_padding + static bsz.
if [ "${PAD_MODE}" != "no_padding" ]; then
    echo "ERROR: PAD_MODE must be no_padding for Qwen3.5 megatron bshd path."
    exit 1
fi

if [ "${EP_SIZE}" -lt 1 ] || [ "${ETP_SIZE}" -lt 1 ]; then
    echo "ERROR: EP_SIZE and ETP_SIZE must be >= 1."
    exit 1
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=${PYTHONPATH:-}:/llm-align/liuchonghan/verl_lao

# Key settings:
#   engine.use_remove_padding=False   - keep bshd path instead of THD
#   engine.vanilla_mbridge=True       - use mbridge (not megatron-bridge)
MOE_TRANSFORMER_CONFIG=""
if [ -n "${MOE_ROUTER_LOAD_BALANCING_TYPE}" ]; then
    MOE_TRANSFORMER_CONFIG="${MOE_TRANSFORMER_CONFIG} +engine.override_transformer_config.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE}"
fi
if [ -n "${MOE_AUX_LOSS_COEFF}" ]; then
    MOE_TRANSFORMER_CONFIG="${MOE_TRANSFORMER_CONFIG} +engine.override_transformer_config.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF}"
fi
if [ -n "${MOE_Z_LOSS_COEFF}" ]; then
    MOE_TRANSFORMER_CONFIG="${MOE_TRANSFORMER_CONFIG} +engine.override_transformer_config.moe_z_loss_coeff=${MOE_Z_LOSS_COEFF}"
fi

ENGINE_CONFIG="\
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
    +optim.override_optimizer_config.use_precision_aware_optimizer=False \
    +optim.override_optimizer_config.optimizer_cpu_offload=False \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_distributed_optimizer=False \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=False \
    engine.override_transformer_config.attention_backend=auto \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1 \
    ${MOE_TRANSFORMER_CONFIG}"

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
    model.path=${MODEL_PATH} \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    +model.override_config.output_router_logits=${OUTPUT_ROUTER_LOGITS} \
    +model.override_config.router_dtype="${ROUTER_DTYPE}" \
    model.enable_gradient_checkpointing=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=1000 \
    trainer.max_ckpt_to_keep=10 \
    trainer.logger="['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    'checkpoint.save_contents=[model,hf_model]'
