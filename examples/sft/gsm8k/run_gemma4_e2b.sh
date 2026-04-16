# Tested for Gemma4 HF-style SFT with multi-turn `messages` parquet data.
#
# Notes:
# - Gemma4 support in verl currently uses the native HF forward path.
# - Keep `model.use_remove_padding=False` until a Gemma4-specific VLM
#   remove-padding patch is added, similar to the Qwen/GLM paths.
# - Gemma4 emits a thinking-channel prefix before assistant responses; the
#   dataset code masks that prefix out automatically.

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma4_e2b.sh <nproc_per_node> <save_path> [other_configs...]"
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
    data.train_batch_size=32 \
    data.max_length=2048 \
    data.truncation=left \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    data.ignore_input_ids_mismatch=True \
    model.path=google/gemma-4-E2B-it \
    model.use_remove_padding=False \
    model.trust_remote_code=True \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    engine=fsdp \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma4-e2b-it \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@
