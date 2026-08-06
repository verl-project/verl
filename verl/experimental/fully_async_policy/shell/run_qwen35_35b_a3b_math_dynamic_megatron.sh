#!/usr/bin/env bash
# Qwen3.5-35B-A3B GRPO with Megatron backend + MTP + Fully Async Policy + Dynamic Resource Scheduling
#
# MTP (Multi-Token Prediction) notes:
#   - actor_rollout_ref.model.mtp.enable=True        enables MTP module
#   - actor_rollout_ref.model.mtp.enable_train=True  enables MTP training loss
#   - actor_rollout_ref.model.mtp.enable_rollout=True enables speculative decoding in SGLang
#
# Example parallelism configs for Qwen3.5-35B-A3B:
#   24 GPUs (3 hybrid nodes + 1 rollout node): train_tp=4 train_pp=3 EP=8 gen_tp=8
#
# Run:
#     MODEL_PATH=/path/to/Qwen3.5-35B-A3B TRAIN_FILE=/path/to/train.parquet TEST_FILE=/path/to/test.parquet \
#       NNODES_TRAIN=3 NNODES_ROLLOUT=1 bash run_qwen35_35b_a3b_math_dynamic_megatron.sh

set -xeuo pipefail

# ================= data / model =================
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

project_name=${PROJECT_NAME:-'GRPO-Qwen35-35b-MATH'}
exp_name=${EXP_NAME:-'GRPO-Qwen35-35b-MATH-dynamic-megatron'}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

mtp_params=(
  actor_rollout_ref.model.mtp.enable=True
  actor_rollout_ref.model.mtp.enable_train=False
  actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1
  actor_rollout_ref.model.mtp.detach_encoder=True
  actor_rollout_ref.model.mtp.enable_rollout=True
  actor_rollout_ref.model.mtp.speculative_algorithm="NEXTN"
  actor_rollout_ref.model.mtp.speculative_num_steps=4
  actor_rollout_ref.model.mtp.speculative_eagle_topk=1
  actor_rollout_ref.model.mtp.speculative_num_draft_tokens=5
)

# Fully async 分离架构专用参数
rollout_mode="async"
return_raw_chat="True"

NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

gen_prompt_bsz=1
total_rollout_steps=30000
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

python3 -X faulthandler -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=0 \
    data.max_prompt_length=4096 \
    data.val_batch_size=32 \
    data.max_response_length=64000 \
    data.truncation='left' \
    data.return_raw_chat=${return_raw_chat} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=70000 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=${True} \
    actor_rollout_ref.actor.megatron.grad_offload=${True} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${True} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
   +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
   +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
   +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=70000 \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.val_before_train=False \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.use_dynamic_resource_scheduling=True \
    async_training.dynamic_schedule_policy="default" \
    async_training.dynamic_schedule_deactivate_ratio=0.6 \
    async_training.dynamic_schedule_enable_rebalance=True \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=44398 \
    actor_rollout_ref.nccl_timeout=9600 \
    "${mtp_params[@]}" \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=True \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=128 \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=128 \
   +trainer.worker_env.PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False \
    trainer.total_epochs=1 $@