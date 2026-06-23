#!/usr/bin/env bash
#
# GRPO | Qwen3-VL-30B-A3B | VeOmni training | NVIDIA GPUs or Ascend NPUs
#
# INFER_BACKEND controls rollout backend: vllm

set -xeuo pipefail

# ---- user-adjustable ----
data_path=${data_path:-$HOME/geo3k}
model_path=${model_path:-$HOME/Qwen3-VL-30B-A3B-Instruct}

nnodes=${nnodes:-2}
num_gpus_per_node=8

# Parallelism settings
dp_size=${dp_size:-16}
usp_size=${usp_size:-1}
ep_size=${ep_size:-1}

# Model and project settings
model_id=Qwen3_VL-30B-MOE
project_name=${model_id}-veomni
exp_name=grpo-${num_gpus_per_node}gpu

backend=fsdp2
model_engine=veomni

# ===================================== Algorithm =====================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.001

# Actor settings
use_kl_loss=True
kl_loss_coef=0.01
kl_loss_type=low_var_kl
entropy_coeff=0
actor_lr=1e-6
lr_scheduler_type=constant

ppo_mini_batch_size=32
critic_warmup=0

# ===================================== Data/Model =====================================
train_files=$data_path/train.parquet
test_files=$data_path/test.parquet

actor_model_path=$model_path

max_prompt_length=1024
max_response_length=2048
train_batch_size=64

use_remove_padding=True
enable_gradient_checkpointing=True
max_position_embeddings=32768

# ===================================== Training =====================================
ppo_micro_batch_size_per_gpu=1

# VeOmni config
ACTOR_VEOMNI_CONFIG="
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.fsdp_size=$dp_size \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=$usp_size \
    actor_rollout_ref.actor.veomni.expert_parallel_size=$ep_size \
    actor_rollout_ref.actor.veomni.attn_implementation=veomni_flash_attention_2_with_sp \
    actor_rollout_ref.actor.veomni.moe_implementation=fused"

# Ref model config
REF_VEOMNI_CONFIG="
    actor_rollout_ref.ref.strategy=veomni \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.veomni.param_offload=False \
    actor_rollout_ref.ref.veomni.expert_parallel_size=1"
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$lr_scheduler_type \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    +actor_rollout_ref.model.override_config.max_position_embeddings=$max_position_embeddings \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"

CONFIG_NAME=ppo_trainer
ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_VEOMNI_CONFIG $REF_VEOMNI_CONFIG"

# ===================================== Inference =====================================
rollout_name=vllm
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.6
n_resp_per_prompt=8
max_model_len=4096
max_num_batched_tokens=5120

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$max_model_len"

########################### parameter arrays ###########################

CONFIG=(
    --config-path=./config
    --config-name=$CONFIG_NAME
)

DATA=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.use_kl_in_reward=$use_kl_in_reward
    algorithm.kl_ctrl.kl_coef=$kl_coef
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.train_batch_size=$train_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=True
    data.shuffle=False
    data.truncation='error'
    data.image_key=images
)

TRAINER=(
    trainer.resume_mode=auto
    trainer.critic_warmup=$critic_warmup
    trainer.logger=['console','wandb']
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=$num_gpus_per_node
    trainer.nnodes=$nnodes
    trainer.save_freq=40
    trainer.test_freq=5
    trainer.total_epochs=20
    trainer.total_training_steps=200
)

EXTRA=(
    model_engine=$model_engine
    $ACTOR_CONFIG
    $ROLLOUT_CONFIG
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${CONFIG[@]}" \
    "${DATA[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"