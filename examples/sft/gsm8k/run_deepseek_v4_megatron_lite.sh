#!/usr/bin/env bash
# GSM8K SFT scale demo | DeepSeek-V4 | Megatron Lite training | GPU
#
# Megatron Lite is Megatron's agentic experimental path. Its upstream home is
# Megatron-LM's dev branch:
# https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite
#
# This launcher currently tracks the submitter's active branch until the latest
# mlite changes merge upstream. That checkout provides both megatron.lite and
# the verl_mlite backend glue:
#
#   git clone -b lite https://github.com/verl-project/Megatron-LM mlite
#   pip install -e mlite/experimental/lite/examples/verl
#
# DeepSeek-V4 uses fused DSA kernels on Hopper and Blackwell GPUs. The critical
# DSA-only dependencies are nvidia-cutlass-dsl==4.5.2 and nvidia-cudnn-frontend.
# cudnn-frontend release 1.24.1 is sufficient for Blackwell, while Hopper still
# needs a develop-branch build with IndexerForwardSm90 support.
#
# MODEL_VARIANT selects the DeepSeek-V4 target and its default mlite mesh:
#   - flash: 16 nodes, PP4 EP8  CP4
#   - pro:   64 nodes, PP8 EP16 CP4
#
# DS4 is fixed to TP1/ETP1. The architecture does not support TP/ETP
# sharding, and there is no plan to support it.
#
# OPTIMIZER selects the Megatron Lite optimizer path:
#   - fsdp2:    Megatron Lite FSDP2 wrapper, lower memory pressure, default
#   - dist_opt: original Megatron distributed optimizer
# When using dist_opt, prefer a larger PP*EP mesh to reduce per-rank model and
# optimizer memory pressure and avoid OOM.

set -xeuo pipefail

########################### mlite backend knobs ###########################
MODEL_VARIANT=${MODEL_VARIANT:-flash}
MLITE_ROOT=${MLITE_ROOT:-$HOME/mlite}
MLITE_VERL_ROOT=${MLITE_VERL_ROOT:-${MLITE_ROOT}/experimental/lite/examples/verl}
MLITE_LITE_ROOT=${MLITE_LITE_ROOT:-${MLITE_ROOT}/experimental/lite}

OPTIMIZER=${OPTIMIZER:-fsdp2} # dist_opt
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

case "${MODEL_VARIANT}" in
    flash)
        MODEL_PATH=${MODEL_PATH:-${FLASH_MODEL_PATH:-}}
        NNODES=${NNODES:-16}
        PP=${PP:-4}
        EP=${EP:-8}
        CP=${CP:-4}
        ;;
    pro)
        MODEL_PATH=${MODEL_PATH:-${PRO_MODEL_PATH:-}}
        NNODES=${NNODES:-64}
        PP=${PP:-8}
        EP=${EP:-16}
        CP=${CP:-4}
        ;;
    *)
        echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}. Expected flash or pro." >&2
        exit 1
        ;;
esac

: "${MODEL_PATH:?set MODEL_PATH, or set FLASH_MODEL_PATH/PRO_MODEL_PATH for MODEL_VARIANT=${MODEL_VARIANT}}"
########################### end mlite backend knobs ###########################

########################### user-adjustable ###########################
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}

LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-verl-mlite-deepseek_v4_${MODEL_VARIANT}-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_${MODEL_VARIANT}_gsm8k_sft_${OPTIMIZER}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
export PYTHONPATH="${MLITE_VERL_ROOT}:${MLITE_LITE_ROOT}:${MLITE_ROOT}:${VERL_ROOT:-}:${MEGATRON_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
fi

########################### parameter arrays ###########################
DATA=(
    data.train_files="${TRAIN_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu=${MAX_LENGTH}
    data.max_length=${MAX_LENGTH}
    data.pad_mode=no_padding
    data.truncation=error
    data.messages_key=messages
)

MODEL=(
    model=hf_model
    model.path="${MODEL_PATH}"
    model.trust_remote_code=True
)

OPTIM=(
    optim=megatron
    optim.lr=${LR}
    optim.min_lr=${MIN_LR}
    optim.weight_decay=${WEIGHT_DECAY}
    optim.clip_grad=${CLIP_GRAD}
    optim.lr_warmup_steps=0
    optim.lr_decay_style=constant
    +optim.override_optimizer_config.offload_fraction=1.0
    +optim.override_optimizer_config.use_precision_aware_optimizer=True
    +optim.override_optimizer_config.decoupled_weight_decay=True
)

ENGINE=(
    hydra.searchpath=[pkg://verl_mlite.config]
    engine=mlite
    engine.tp=1
    engine.pp=${PP}
    engine.vpp=1
    engine.ep=${EP}
    engine.cp=${CP}
    engine.etp=1
    engine.param_offload=${ALL_OFFLOAD}
    engine.optimizer_offload=${ALL_OFFLOAD}
    engine.grad_offload=${ALL_OFFLOAD}
    engine.attention_backend_override=flash
    engine.impl_cfg.use_thd=True
    +engine.impl_cfg.optimizer=${OPTIMIZER}
    +engine.impl_cfg.recompute=[full]
)

TRAINER=(
    trainer.logger=[console]
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
)

########################### launch ###########################
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NDEVICES_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl_mlite.launch verl.trainer.sft_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${OPTIM[@]}" \
    "${ENGINE[@]}" \
    "${TRAINER[@]}" \
    "$@"
