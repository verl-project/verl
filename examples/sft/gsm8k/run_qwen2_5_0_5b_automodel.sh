# Requires: Automodel, transformers>=5.3.0, torchao
# MoE also requires: grouped_gemm (github.com/fanshiqing/grouped_gemm v1.1.4)

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_0_5b_automodel.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k_sft/train.parquet \
    data.val_files=$HOME/data/gsm8k_sft/test.parquet \
    data.train_batch_size=128 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    model=hf_model \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    model.use_remove_padding=True \
    engine=automodel \
    engine.distributed_strategy=fsdp2 \
    engine.tp_size=1 \
    engine.pp_size=1 \
    engine.cp_size=1 \
    engine.ep_size=1 \
    engine.use_torch_compile=False \
    optim=automodel \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.init_lr_ratio=0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-automodel \
    trainer.total_epochs=2 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger=console \
    trainer.seed=1111 \
    trainer.resume_mode=disable $@
