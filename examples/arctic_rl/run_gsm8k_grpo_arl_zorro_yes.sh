#!/bin/bash
# Canonical GSM8K example for the Arctic RemoteBackend.
#
# Demonstrates the generic `verl.remote_backend` abstraction with the
# Arctic adapter (`trainer.remote_backend=arctic`). Single-GPU, GRPO,
# Qwen3-0.6B; intended as a quick convergence sanity check.

set -x
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_HOME=/checkpoint/huggingface
export USE_ARCTIC_TRAINING_CLIENT=1
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0)
if   [[ $gpu_name == *"H200"* ]]; then flash_attention_v=flash_attention_3
elif [[ $gpu_name == *"B200"* || $gpu_name == *"B300"* ]]; then flash_attention_v=flash_attention_2
else flash_attention_v=flash_attention_2
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/code/shared/gsm8k/train.parquet \
    data.val_files=/code/shared/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    reward.num_workers=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$flash_attention_v \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=arctic \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    remote_backend.arctic.colocate=False \
    remote_backend.arctic.training_gpus=1 \
    remote_backend.arctic.sampling_gpus=1 \
    remote_backend.arctic.log_prob_gpus=0 \
    remote_backend.arctic.zero_optimization.stage=2 \
    remote_backend.arctic.zero_optimization.offload_optimizer.device=none \
    remote_backend.arctic.zero_optimization.offload_param.device=none \
    remote_backend.arctic.use_zorro=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.experiment_name=gsm8k_grpo_qwen3_0p6b_ngpu1_gbs16_rolln5_zorroTrue \
    trainer.project_name=arctic_rl_gsm8k_public \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=80 \
    trainer.total_epochs=15 \
    "$@" 2>&1
