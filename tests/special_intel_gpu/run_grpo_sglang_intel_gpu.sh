#!/usr/bin/env bash
# E2E GRPO test on Intel GPU with SGLang rollout; the vLLM variant is run_grpo_intel_gpu.sh.
#
# Validates the full RL training loop on XPU in hybrid mode:
#   FSDP2 training -> sleep (release sglang KV + weights) -> resume weights -> refit
#   -> resume KV -> SGLang rollout -> reward -> train
#
# Prerequisites:
#   - PyTorch with XPU support (torch.xpu.is_available() == True)
#   - SGLang with XPU support, plus torch_memory_saver for sleep/wake
#   - oneCCL for xccl distributed backend
#
# Usage:
#   NUM_GPUS=2 bash tests/special_intel_gpu/run_grpo_sglang_intel_gpu.sh

set -x

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
DATA_DIR=${DATA_DIR:-$HOME/data}

# One absolute ZE_AFFINITY_MASK for every process; Ray must not rewrite it per actor,
# so each sglang replica is placed at its rank offset inside the mask.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK=1
# The inductor static launcher segfaults on XPU in torch 2.13.
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$DATA_DIR/gsm8k/train.parquet \
    data.val_files=$DATA_DIR/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_sglang' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@
