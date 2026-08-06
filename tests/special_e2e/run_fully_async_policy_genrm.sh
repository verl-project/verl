#!/usr/bin/env bash
set -xeuo pipefail

# Test script for fully_async_policy + GenRM E2E regression testing
# This script runs fully async PPO training with a standalone GenRM reward model
# to verify that GenRM works correctly in async mode.
#
# GPU allocation (3 GPUs minimum):
#   - 1 GPU: Rollout (vLLM async)
#   - 1 GPU: Training (FSDP2, offload enabled)
#   - 1 GPU: GenRM (vLLM standalone)

NUM_GPUS=${NUM_GPUS:-3}

# Model paths (use HF repo ID by default, auto-downloaded by transformers/vLLM)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
# Use the same small model as GenRM judge for testing (production would use a larger model)
GRM_PATH=${GRM_PATH:-Qwen/Qwen2.5-3B-Instruct}

rollout_mode="async"
rollout_name="vllm"
export VLLM_USE_V1=1
export GENRM_MODEL_NAME="${GRM_PATH}"
return_raw_chat="True"

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=256
max_response_length=512
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Temperature parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Fully async specific parameters
n_gpus_rollout=1
n_gpus_training=1
n_gpus_genrm=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=4
total_rollout_steps=3200  # ~200 wandb data points on 3x H100
test_freq=-1
staleness_threshold=0.5
trigger_parameter_sync_step=4
partial_rollout=True
use_trainer_do_validate=False

exp_name="$(basename "${MODEL_PATH,,}")-fully-async-policy-genrm-fsdp2-minimal"

echo "Running fully_async_policy + GenRM with FSDP2 strategy"
echo "Total GPUs: ${NUM_GPUS}, Rollout GPUs: ${n_gpus_rollout}, Training GPUs: ${n_gpus_training}, GenRM GPUs: ${n_gpus_genrm}"

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
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
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
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.n_gpus_per_node=${n_gpus_genrm} \
    reward.reward_model.nnodes=1 \
    reward.reward_model.model_path="${GRM_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.5 \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.custom_reward_function.path=tests/experimental/reward_loop/reward_fn.py \
    reward.custom_reward_function.name=compute_score_gsm8k \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-test-fully-async-genrm' \
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

echo "Fully async policy + GenRM E2E test completed successfully"
