# Gemma4 31B SFT entrypoint aligned with the Automodel branch's validated settings.
#
# Usage:
#   bash examples/sft/gsm8k/run_gemma4_31b.sh <nproc_per_node> <save_path> [other hydra overrides...]
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

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=8 \
    data.max_length=2048 \
    data.truncation=left \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    data.ignore_input_ids_mismatch=True \
    model.path=google/gemma-4-31B-it \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    +model.override_config.attn_implementation=sdpa \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.betas='[0.9,0.95]' \
    engine=fsdp \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma4-31b-it \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@
