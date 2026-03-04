#!/usr/bin/env bash
set -xeuo pipefail

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
NUM_GPUS=${NUM_GPUS:-1}
FSDP_SIZE=${FSDP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
VERL_EXP_NAME=${VERL_EXP_NAME:-Titan_Qwen3_30B_A3B_DP8_EP8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

python3 -m verl.trainer.main_ppo \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=8  \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}"  \
    data.seed=42 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.min_lr_factor=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1  \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.attn_type=flex \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.param_offload=True \
    actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_SEQ_LEN}" \
    actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_SEQ_LEN}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len="${MAX_SEQ_LEN}" \
    critic.optim.lr=1e-5 \
    critic.model.path="${MODEL_PATH}" \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger=['console','file','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k_0302' \
    trainer.experiment_name="${VERL_EXP_NAME}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations=1 \
    trainer.test_freq=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps=100 $@
