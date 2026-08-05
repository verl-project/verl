#!/usr/bin/env bash
# SFT | multiturn | FSDP engine | NVIDIA GPUs
# Toggle Ulysses sequence parallel via SP_SIZE env var.

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_0_5b_fsdp.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "  Env: SP_SIZE (default 1)"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
SP_SIZE=${SP_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
PROJECT_NAME=${PROJECT_NAME:-multiturn-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multiturn-sft-qwen2_5_0_5b}
# ---- end user-adjustable ----

torchrun --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/multiturn/train.parquet \
    data.val_files=$HOME/data/multiturn/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} "$@"
