#!/usr/bin/env bash
# SFT | GSM8K | FSDP engine | Ascend NPU
# Toggle Ulysses sequence parallel and LoRA/PEFT via env vars.
#
# Examples:
#   # plain SFT
#   bash run_qwen3_8b_fsdp.sh 8 /tmp/sft-ckpt
#
#   # sequence-parallel (Ulysses) = 2 + LoRA (the default demo)
#   SP_SIZE=2 USE_PEFT=1 bash run_qwen3_8b_fsdp.sh 8 /tmp/sft-ckpt

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_8b_fsdp.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "  Env: SP_SIZE (default 2), USE_PEFT (0|1, default 1)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
SP_SIZE=${SP_SIZE:-2}
USE_PEFT=${USE_PEFT:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-64}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
PROJECT_NAME=${PROJECT_NAME:-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gsm8k-sft-qwen3-8b-instruct}
# ---- end user-adjustable ----

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    optim.lr=${LR} \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "${extra_args[@]}" "$@"
