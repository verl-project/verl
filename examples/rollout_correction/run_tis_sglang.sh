#!/usr/bin/env bash
set -xeuo pipefail

ID=${1:-"qwen3_8b_base_sglang_fp8_8gpu_tis_on"}
CAP=${2:-"5"}

project_name='DAPO-1'
exp_name=$ID

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"

# Paths
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${project_name}/${exp_name}"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B-Base"}

export SGLANG_PATCH=1
export SGLANG_PORT=30300
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_TEMP_DIR=/workdir/ray_temp
mkdir -p ${RAY_TEMP_DIR}


# Start a new Ray cluster using GPUs 2,3,4,5 on a different port
# This allows running multiple training tasks simultaneously
RAY_PORT=6811
RAY_DASHBOARD_PORT=8897

# Check if Ray server is already running on the specified port
# Use localhost instead of IP address to avoid IP mismatch issues
echo "Checking if Ray server is already running on localhost:${RAY_PORT}..."
if ray status --address="localhost:${RAY_PORT}" >/dev/null 2>&1; then
    echo "Ray server is already running on localhost:${RAY_PORT}, using existing cluster."
    RAY_RUNNING=true
else
    echo "Ray server is not running, starting new Ray cluster..."
    # Start new Ray cluster with 8 GPUs
    ray start --head \
        --port=${RAY_PORT} \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=8 \
        --dashboard-host=0.0.0.0 \
        --temp-dir=${RAY_TEMP_DIR} \
        --disable-usage-stats || true
    echo "Ray cluster started successfully on localhost:${RAY_PORT}"
    # Wait a moment for Ray to fully initialize
    echo "Waiting for Ray cluster to be ready..."
    sleep 5
    # Verify Ray is actually running
    if ray status --address="localhost:${RAY_PORT}" >/dev/null 2>&1; then
        echo "Ray cluster is ready and accessible."
        RAY_RUNNING=true
    else
        echo "Warning: Ray cluster may not be fully ready yet, but continuing..."
        RAY_RUNNING=false
    fi
fi

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files=$HOME/data/BytedTsinghua-SIA/DAPO-Math-17k/train.parquet \
    data.val_files=$HOME/data/AIME-2024/train.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.val_max_samples=-1 \
    data.max_response_length=$((1024 * 20)) \
    data.gen_batch_size=96 \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=22528 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.load_format=flash_rl \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.use_policy_gradient=False \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    algorithm.rollout_correction.rollout_rs_threshold_lower=null \
    algorithm.rollout_correction.rollout_token_veto_threshold=null \
    actor_rollout_ref.rollout.calculate_log_probs=True