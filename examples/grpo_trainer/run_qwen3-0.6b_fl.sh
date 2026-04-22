#!/bin/bash
# FL Multi-Chip Support Version of run_qwen3-0.6b.sh
# This script demonstrates training with FL (FlagOS) multi-chip support
# including FlagGems operators, Transformer-Engine-FL, and FlagCX communication.
#
# Reference: docs/design/fl_multi_chip_support.md

set -x

# ============ Device Configuration ============
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HYDRA_FULL_ERROR=1

# ============ FlagCX Communication Library ============
# export FLAGCX_PATH=/path/to/FlagCX
# export PYTHONPATH=/path/to/FlagCX/plugin/torch:${PYTHONPATH}

# ============ FL Configuration via verl fl_config ============
# Note: Environment variables below are for reference only.
# In verl FL architecture, these are set dynamically by FLEnvManager
# based on fl_config YAML configuration.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_ENGINE_DEVICE=flagos
# Training phase environment variables:
export TE_FL_PREFER=flagos	#flagos / vendor / reference	flagos
export TE_FL_PREFER_VENDOR=0	# Prefer vendor (legacy)	1 / 0	0
export TE_FL_STRICT=0	# Strict mode (no fallback)	1 / 0	0
# TE_FL_ALLOW_VENDORS=nvidia,amd	# Allowed vendors (whitelist)	nvidia,amd
# TE_FL_DENY_VENDORS=vendor_a	# Denied vendors (blacklist)	vendor_a
# TE_FL_PER_OP=rmsnorm_fwd=vendor:cuda|default
export VLLM_FL_FLAGOS_BLACKLIST="where_scalar_other,where_scalar_self,where_self,where_self_out,pad"
# Logging
# Variable	Description	Values	Default
export TEFL_LOG_LEVEL=DEBUG # / INFO / WARNING / ERROR	INF

# Rollout phase environment variables:
# export VLLM_PLUGINS=""
# export VLLM_FL_PREFER_ENABLED=true
# export VLLM_FL_PLATFORM=cuda # will cause error
# export VLLM_FL_PREFER=flagos
export USE_FLAGGEMS=true
export VLLM_FL_OOT_ENABLED=1
export USE_FLAGCX=1
# unset FLAGCX_PATH
export FLAGCX_PATH=/path/to/FlagCX

export FLAGCX_LOG_LEVEL=DEBUG

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=~/data/rl/gsm8k/train.parquet \
    data.val_files=~/data/rl/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=~/models/Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.custom_engine_module='pkg://verl_plugin_fl.engine' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.custom_engine_module='pkg://verl_plugin_fl.engine' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_fl' \
    trainer.experiment_name='qwen3_0.6b_fl' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.use_legacy_worker_impl='disable' \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.total_epochs=15 \
    $@
