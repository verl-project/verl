#!/usr/bin/env bash
set -xeuo pipefail

# Test script for fully_async_policy + Online Policy Distillation (OPD)
# This script runs fully async training with a standalone teacher model
# to verify that distillation works correctly in async mode.
#
# GPU allocation (3 GPUs minimum):
#   - 1 GPU: Rollout (student vLLM async)
#   - 1 GPU: Training (student FSDP2, offload enabled)
#   - 1 GPU: Teacher (teacher vLLM standalone)
#
# Usage:
#   bash tests/special_e2e/run_fully_async_policy_opd.sh
#
#   # Run longer:
#   bash tests/special_e2e/run_fully_async_policy_opd.sh rollout.total_rollout_steps=10000
#
#   # Enable wandb:
#   bash tests/special_e2e/run_fully_async_policy_opd.sh \
#       trainer.logger='["console","wandb"]' \
#       trainer.project_name='async-opd-test'

NUM_GPUS=${NUM_GPUS:-3}

# Model paths
# Follow PR #5834 testing setup: Qwen2.5-0.5B student + Qwen2.5-3B-Instruct teacher
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-3B-Instruct}

rollout_mode="async"
rollout_name="vllm"
export VLLM_USE_V1=1

# Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False

# Response length parameters
max_prompt_length=256
max_response_length=512
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# Distillation parameters (following PR #5834 setup)
distillation_loss_mode="k1"
distillation_topk=64
use_policy_gradient=True
use_task_rewards=False
distillation_loss_max_clamp=10.0
distillation_log_prob_min_clamp=-10.0

# Fully async specific parameters
n_gpus_rollout=1
n_gpus_training=1
n_gpus_teacher=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=4
total_rollout_steps=16  # Small number to just verify the flow works
test_freq=-1
staleness_threshold=0.5
trigger_parameter_sync_step=4
partial_rollout=True
use_trainer_do_validate=False

exp_name="$(basename "${STUDENT_MODEL,,}")-fully-async-opd-fsdp2-minimal"

echo "Running fully_async_policy + OPD with FSDP2 strategy"
echo "Total GPUs: ${NUM_GPUS}, Rollout: ${n_gpus_rollout}, Training: ${n_gpus_training}, Teacher: ${n_gpus_teacher}"
echo "Student: ${STUDENT_MODEL}, Teacher: ${TEACHER_MODEL}"

# Detect device
device_name=$(python3 - <<'EOF'
from verl.utils.device import get_device_name
print(get_device_name())
EOF
)

gen_tp=1
sp_size=1
fsdp_size=1
ref_offload=True
actor_offload=False

if [ -n "$device_name" ] && [ "$device_name" == "npu" ]; then
    actor_offload=True
fi

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_tokens} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    distillation.enabled=True \
    distillation.num_workers=8 \
    distillation.teacher_model.n_gpus_per_node=${n_gpus_teacher} \
    distillation.teacher_model.nnodes=1 \
    distillation.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_model.inference.name=${rollout_name} \
    distillation.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_model.inference.gpu_memory_utilization=0.3 \
    distillation.teacher_model.inference.max_model_len=${max_num_tokens} \
    distillation.teacher_model.inference.max_num_batched_tokens=${max_num_tokens} \
    distillation.teacher_model.inference.max_num_seqs=${max_num_tokens} \
    distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
    distillation.distillation_loss.topk=${distillation_topk} \
    distillation.distillation_loss.use_task_rewards=${use_task_rewards} \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    distillation.distillation_loss.loss_max_clamp=${distillation_loss_max_clamp} \
    distillation.distillation_loss.log_prob_min_clamp=${distillation_log_prob_min_clamp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test-fully-async-opd' \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.log_val_generations=10 \
    trainer.use_legacy_worker_impl=disable \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    trainer.total_epochs=2 \
    trainer.test_freq=${test_freq} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.use_trainer_do_validate=${use_trainer_do_validate} \
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    "$@"

echo "Fully async policy + OPD E2E test completed successfully"
