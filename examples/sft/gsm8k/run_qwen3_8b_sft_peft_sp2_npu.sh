set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_8b_sft_peft_sp2_npu.sh <nproc_per_node> <save_path> [other_configs...]"
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
    data.micro_batch_size_per_gpu=64 \
    optim.lr=1e-4 \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=2 \
    model.path=Qwen/Qwen3-8B \
    model.use_remove_padding=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3-8b-instruct \
    trainer.logger=console \
    trainer.total_epochs=2 $@ \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear
