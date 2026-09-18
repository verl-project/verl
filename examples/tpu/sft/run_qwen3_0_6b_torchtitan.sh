#!/usr/bin/env bash
# TorchTitan | SFT Training | Qwen3-0.6B | TPU v6e-4 Local VM
#
# Hardware Setup:
#   1 Slice of TPU v6e-4 (1 physical host VM, 4 TPU chips total).
#   
# Parallelism Config:
#   TP (Tensor Parallel) = 2
#   DP (Data Parallel / FSDP) = 2
#   PP (Pipeline Parallel) = 1
#   Total Chips = TP * DP * PP = 2 * 2 * 1 = 4 chips.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1

# Disable XLA HLO fusion passes for stable TPU compiler execution
export XLA_FLAGS="--xla_disable_hlo_passes=instruction-fusion,fusion-merger,multi-output-fusion,horizontal-fusion"

# Project and Experiment details
project_name='GRPO_TPU_SFT'
exp_name='SFT-Qwen3-0.6B-tpu-torchtitan-v6e4'

# Paths (overridable via env vars; supports HuggingFace model ID or local path)
MODEL_PATH="${MODEL_PATH:-assets/hf/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-gsm8k_sft/train.parquet}"
TEST_FILE="${TEST_FILE:-gsm8k_sft/test.parquet}"

# JAX/XLA Memory Preallocation and Launch Barrier Configuration
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536"

# TPU Node topology configs
export NNODES_TRAINER=1       # 1 physical VM host
export N_CHIPS_TRAINER=4      # 4 TPU chips per VM host

# Launch Ray SFT Trainer
# We use the SFT Trainer Ray entrypoint to orchestrate across TPU VMs
python3 -m verl.trainer.sft_trainer_ray \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.val_max_samples=32 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=tpu_binned_pack \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_length=2048 \
    data.max_token_len_per_gpu=2048 \
    data.ignore_input_ids_mismatch=True \
    model.use_remove_padding=False \
    engine=torchtitan \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    trainer.total_training_steps=8 \
    engine.tensor_parallel_size=2 \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size=1 \
    engine.data_parallel_shard_size=2 \
    engine.use_torch_compile=False \
    engine.attn_type=varlen \
    engine.max_seq_len=2048 \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger="['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "$@"
