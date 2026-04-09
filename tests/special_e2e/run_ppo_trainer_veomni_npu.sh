#!/usr/bin/env bash
pkill -9 ray
pkill -9 VLLM
set -xeuo pipefail

export ASCEND_LAUNCH_BLOCKING=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export MULTI_STREAM_MEMORY_REUSE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ASCEND_ENABLE_NZ=0
export RAY_DEDUP_LOGS=0
export CPU_AFFINITY_CONF=1
export HCCL_EXEC_TIMEOUT=17340
export HCCL_CONNECT_TIMEOUT=7200
export VLLM_USE_V1=1
export HCCL_IF_BASE_PORT=50000
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=20
export HYDRA_FULL_ERROR=1
export HCCL_BUFFSIZE=1024


# Download model if not exists
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/Qwen/Qwen3-30B-A3B-16layer-Merge}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
NUM_GPUS=${NUM_GPUS:-8}
FSDP_SIZE=${FSDP_SIZE:-2}
SP_SIZE=${SP_SIZE:-2}
EP_SIZE=${EP_SIZE:-2}
VERL_EXP_NAME=${VERL_EXP_NAME:function-reward-minimal-fsdp-size${FSDP_SIZE}}

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.expert_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_veomni_test' \
    trainer.experiment_name="${VERL_EXP_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.device=npu \
