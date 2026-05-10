# Tentative for testing, will polish later
#!/usr/bin/env bash

set -xeuo pipefail

project_name='DAPO_fully_async'
exp_name='DAPO-Qwen3-8b-Base-MATH-megatron-fully-async-trtllm-gb200-4-4-bypassFalse'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

loss_agg_mode="token-mean"


NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3/Qwen3-8B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
# To run nsys profiling, set TLLM_NSYS_OUTPUT_DIR to a directory.

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False
gen_tp=2
train_tp=2
train_pp=1
train_cp=1

train_prompt_bsz=128
n_resp_per_prompt=16
train_prompt_mini_bsz=32

fully_async=(
  data.train_batch_size=0
  data.gen_batch_size=1
  trainer.test_freq=10
  actor_rollout_ref.hybrid_engine=False
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.actor.optim.lr_decay_steps=51200
  rollout.total_rollout_steps=$(((512*100)))
  trainer.nnodes=1
  trainer.n_gpus_per_node=4
  rollout.nnodes=1
  rollout.n_gpus_per_node=4
  async_training.staleness_threshold=0.5
  async_training.trigger_parameter_sync_step=4
  async_training.require_batches=1
  async_training.partial_rollout=True
)

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.name=trtllm \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    reward.reward_manager.name=naive \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.model.trust_remote_code=True \
    data.trust_remote_code=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.total_epochs=10 \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_is=token \
    "${fully_async[@]}" \
    "$@"
