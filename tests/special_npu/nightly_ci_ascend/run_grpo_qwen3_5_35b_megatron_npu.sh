#!/usr/bin/env bash

set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/geo3k/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3.5-35B-Megatron}

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}" .sh)"
LOG_DIR=/root/.cache/nightly_log/$SCRIPT_NAME
mkdir -p ${LOG_DIR}
rm -f ${LOG_DIR}/${SCRIPT_NAME}.log

export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CPU_AFFINITY_CONF=1

export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

###########################
# Parallel Config
###########################

TP=${TP:-2}
PP=${PP:-2}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}

n_devices_per_node=16

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}

###########################
# DATA
###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=32
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.shuffle=False
    data.validation_shuffle=False
    data.truncation=error
)

###########################
# MODEL
###########################

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

###########################
# ACTOR
###########################

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.checkpoint.strict=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

###########################
# REF
###########################

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=True
)

###########################
# ROLLOUT
###########################

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

###########################
# TRAINER
###########################

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=15
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

EXTRA=(
    model_engine=megatron
)

###########################
# Launch
###########################

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" | tee ${LOG_DIR}/${SCRIPT_NAME}.log