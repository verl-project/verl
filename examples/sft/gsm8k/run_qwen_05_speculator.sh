set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2
 
 
 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.fsdp_config.model_dtype=bf16 \
    model.partial_pretrain=/root/autodl-tmp/qwen3_moe_small \
    +model.speculator.n_predict=3 \
    +model.speculator.method=sum_lstm \
    +model.speculator.tie_lstm_embs=true \
    +model.speculator.tie_weights=true \
    +model.speculator_adapter.fqn=verl.trainer.speculators.lstm_adapter.LSTMSpeculatorAdapter \
    +model.use_remove_padding=false \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct-speculator \
    trainer.logger=console \
    trainer.total_epochs=1 $@

 
 
 
 
 
