set -x

WORKSPACE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_PATH="/root/models/Qwen2.5-Math-7B"
SAVE_PATH="/root/checkpoints/Qwen2.5-Math-7B"
NPROC=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
     -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.train_batch_size=8 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=1024 \
    data.pad_mode=no_padding \
    data.truncation=error \
    model=hf_model \
    model.trust_remote_code=True \
    engine=megatron \
    optim=megatron \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=4 \
    engine.pipeline_model_parallel_size=1 \
    engine.expert_model_parallel_size=1 \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=False \
    engine.use_megatron_fsdp=True \
    model.path=$MODEL_PATH \
    model.use_remove_padding=true \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=SFT-qwen2.5-7b-mfsdp \
    trainer.logger='["console","wandb","file"]' \
    trainer.total_epochs=4 "$@"
