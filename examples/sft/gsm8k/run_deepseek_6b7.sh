set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_deepseek_6b7.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-4 \
    engine=fsdp \
    model.path=deepseek-ai/deepseek-coder-6.7b-instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-deepseek-coder-6.7b-instruct \
    trainer.total_epochs=4 \
    trainer.logger='["console","wandb"]' $@