#!/usr/bin/env bash
# This script is a demo for GRPO training of DeepSeek-V4-Flash using VeOmniEngine.
#
# Environment:
#   - transformers==5.9.0
#   - vllm==0.24.0
#   - veomni==0.1.12

set -xeuo pipefail

# ---- user-adjustable ----
n_gpus_per_node=8
trainer_nnodes=8

project_name='wuxibin_dapo'
exp_name='deepseek_v4_flash_base_ep8_sp2_qat_0824a'

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=5e-4
clip_ratio_high=5.3e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================
model_path=$HDFS_ROOT/model/deepseek-ai/DeepSeek-V4-Flash-Base
train_files=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet
test_files=$DATA_ROOT/dataset/aime25_test.parquet
checkpoint_dir=$DATA_ROOT/checkpoint/$project_name/$exp_name

train_batch_size=128
ppo_mini_batch_size=64
n_resp_per_prompt=8
n_resp_per_prompt_val=16
enable_thinking=${enable_thinking:-True}

# Training config
usp_size=${usp_size:-4}
expert_size=${expert_size:-8}
use_remove_padding=True
use_dynamic_bsz=True
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 30))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / usp_size))

# Inference config
rollout_name=vllm
infer_tp=8
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.6
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
# VeOmni config
ENGINE_CONFIG=(
    model_engine=veomni
    actor_rollout_ref.actor.veomni.enable_fsdp_offload=True
    # actor_rollout_ref.actor.veomni.param_offload=True
    # actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.veomni.enable_full_shard=True
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=$usp_size
    actor_rollout_ref.actor.veomni.expert_parallel_size=$expert_size
    actor_rollout_ref.actor.veomni.moe_implementation=fused_triton
    actor_rollout_ref.actor.veomni.attn_implementation=eager
    actor_rollout_ref.actor.veomni.dsa_indexer_implementation=tilelang
    actor_rollout_ref.actor.veomni.dsa_attention_implementation=tilelang
    actor_rollout_ref.actor.veomni.mhc_implementation=tilelang
    actor_rollout_ref.actor.veomni.qat_implementation=fp8_blockwise
    actor_rollout_ref.actor.veomni.router_replay.mode=R3
)

# Actor model config
ACTOR_CONFIG=(
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding
    actor_rollout_ref.actor.optim.lr=$actor_lr
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.actor.pad_to_length=True
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization
    actor_rollout_ref.rollout.n=$n_resp_per_prompt
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8
)

DATA=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.use_kl_in_reward=$use_kl_in_reward
    algorithm.kl_ctrl.kl_coef=$kl_coef
    algorithm.gamma=$gae_gamma
    algorithm.lam=$gae_lam
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.return_raw_chat=True
    data.train_batch_size=$train_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
    +data.apply_chat_template_kwargs.enable_thinking=${enable_thinking}
)

TRAINER=(
    trainer.critic_warmup=$critic_warmup
    trainer.logger=['console','wandb']
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes=$trainer_nnodes
    trainer.val_before_train=False
    trainer.log_val_generations=100
    trainer.save_freq=20
    trainer.test_freq=20
    trainer.max_actor_ckpt_to_keep=2
    trainer.default_local_dir=$checkpoint_dir
    trainer.total_epochs=10
    trainer.total_training_steps=500
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${TRAINER[@]}" \
    "${ENGINE_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "$@"
