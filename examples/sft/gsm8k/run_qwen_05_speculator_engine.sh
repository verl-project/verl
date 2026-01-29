set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_speculator_engine.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=/model/ljl/arctic-traing-datasets/data/train.parquet \
    data.val_files=/model/ljl/arctic-traing-datasets/data/test.parquet \
    data.micro_batch_size_per_gpu=1 \
    ++data.ignore_input_ids_mismatch=true \
    ++data.enable_thinking_key=null \
    model.path=/model/ljl/Qwen3MoeCustom3 \
    model.use_remove_padding=false \
    ++model.speculator_adapter.fqn=verl.trainer.speculators.lstm_adapter.LSTMSpeculatorAdapter \
    ++model.speculator.n_predict=3 \
    ++model.speculator.method=sum_lstm \
    ++model.speculator.tie_lstm_embs=true \
    ++model.speculator.tie_weights=true \
    engine.strategy=fsdp2 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct-speculator-engine \
    trainer.logger=console \
    trainer.total_epochs=1 $@
