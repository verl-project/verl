# Gemma4 31B single-node HF/FSDP SFT entrypoint.
#
# Usage:
#   bash examples/sft/gsm8k/run_gemma4_31b.sh <nproc_per_node> <save_path> [other hydra overrides...]
#
# Optional environment overrides:
#   MODEL_PATH=/path/to/model
#   TRAIN_FILES=/path/to/train.parquet
#   VAL_FILES=/path/to/val.parquet
#   TRAIN_BATCH_SIZE=8
#   MICRO_BATCH_SIZE_PER_GPU=1
#   ATTN_IMPLEMENTATION=sdpa
#
# Notes:
# - This is the conservative HF/FSDP path for verl.
# - Keep `model.use_remove_padding=False` for now. Gemma4 does not yet have the
#   specialized VLM remove-padding forward patch that Qwen/GLM models use.
# - Activation checkpointing should remain enabled for 31B.
# - Default to `sdpa` attention because Gemma4 31B exceeds FlashAttention 2's
#   supported head dimension on current training environments.

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma4_31b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

shift 2

model_path=${MODEL_PATH:-google/gemma-4-31B-it}
train_files=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
val_files=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
train_batch_size=${TRAIN_BATCH_SIZE:-8}
micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
project_name=${PROJECT_NAME:-gsm8k-sft}
experiment_name=${EXPERIMENT_NAME:-gsm8k-sft-gemma4-31b-it}

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_batch_size=$train_batch_size \
    data.max_length=2048 \
    data.truncation=left \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    data.ignore_input_ids_mismatch=True \
    model.path=$model_path \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    +model.override_config.attn_implementation=$attn_implementation \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.betas='[0.9,0.95]' \
    engine=fsdp \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@
