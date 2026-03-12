# Requires: Automodel, transformers>=5.3.0, torchao
# MoE also requires: grouped_gemm (github.com/fanshiqing/grouped_gemm v1.1.4)

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_30b_automodel.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/hellaswag_sft/hellaswag_sft.parquet \
    data.val_files=$HOME/data/hellaswag_sft/hellaswag_sft.parquet \
    data.train_batch_size=512 \
    data.max_length=2048 \
    data.truncation=left \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=8192 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.train_max_samples=-1 \
    data.val_max_samples=1024 \
    model=hf_model \
    model.path=Qwen/Qwen3-30B-A3B-Base \
    model.trust_remote_code=True \
    model.use_remove_padding=True \
    engine=automodel \
    engine.distributed_strategy=fsdp2 \
    engine.tp_size=1 \
    engine.pp_size=1 \
    engine.cp_size=1 \
    engine.ep_size=8 \
    engine.backend_config.dispatcher=deepep \
    engine.backend_config.attn=te \
    engine.backend_config.linear=te \
    engine.backend_config.rms_norm=torch_fp32 \
    engine.backend_config.enable_fsdp_optimizations=True \
    engine.backend_config.experts=torch_mm \
    engine.activation_checkpointing=True \
    engine.model_dtype=bf16 \
    engine.attn_implementation=te \
    engine.use_torch_compile=False \
    optim=automodel \
    optim.optimizer=FusedAdam \
    optim.optimizer_impl=transformer_engine.pytorch.optimizers.fused_adam \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.init_lr_ratio=0.1 \
    optim.min_lr_ratio=0.01 \
    optim.lr_scheduler_type=cosine \
    optim.master_weights=true \
    optim.store_param_remainders=true \
    optim.exp_avg_dtype=bf16 \
    optim.exp_avg_sq_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=hellaswag-sft \
    trainer.experiment_name=hellaswag-sft-qwen3-30b-automodel \
    trainer.total_epochs=2 \
    trainer.total_training_steps=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.logger=console \
    trainer.seed=1111 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable $@
