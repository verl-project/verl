#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

RM_PAD=${RM_PAD:-True}
FUSED_KERNELS=${FUSED_KERNELS:-False}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-torch} # or 'triton' for triton backend
SP_SIZE=${SP_SIZE:-1}
SEQ_BALANCE=${SEQ_BALANCE:-False}
LIGER=${LIGER:-False}
# Validation
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

train_traj_micro_bsz_per_gpu=2 # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

train_max_token_num_per_gpu=32768
infer_max_token_num_per_gpu=32768

exp_name="$(basename "${MODEL_ID,,}")-model-reward-minimal"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_liger="${LIGER}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.model.use_fused_kernels=${FUSED_KERNELS} \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz="${SEQ_BALANCE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_max_token_num_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_max_token_num_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_max_token_num_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.optim.lr=1e-5 \
    critic.ulysses_sequence_parallel_size="${SP_SIZE}" \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.use_dynamic_bsz="${SEQ_BALANCE}" \
    critic.ppo_max_token_len_per_gpu=${train_max_token_num_per_gpu} \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model_path="${MODEL_PATH}" \
    reward_model.use_reward_loop=True \
    reward_model.rollout.gpu_memory_utilization=0.8 \
    reward_model.rollout.tensor_model_parallel_size=1 \
    reward_model.rollout.prompt_length=1024 \
    reward_model.rollout.response_length=512 \
    reward_model.num_workers=8 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${VAL_BEFORE_TRAIN}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@
