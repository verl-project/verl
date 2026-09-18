#!/bin/bash
set -xeuo pipefail

# Moonlight-16B-A3B-Instruct GRPO on one node with 16 Ascend NPUs.
# Megatron-Bridge is used for training and vLLM is used for rollout.

# Project Configuration
project_name='GRPO-Moonlight-16B-A3B-INSTRUCT-MATH'
exp_name='GRPO-Moonlight-16B-A3B-INSTRUCT-Megatron-vLLM'

# Necessary env
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export DISABLE_L2_CACHE=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=0

# Node Info
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# Model Weights Paths
MODEL_PATH=${MODEL_PATH:-moonshotai/Moonlight-16B-A3B-Instruct}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/gsm8k/test.parquet"}
# Data Length Configuration
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))

# Training Batch Configuration
train_prompt_bsz=16
train_prompt_mini_bsz=16
n_resp_per_prompt=8

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Performance and Memory Management Configuration
all_offload=True
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

# Megatron Parallelism Configuration
train_tp=4
train_ep=4
train_etp=4
train_pp=1
train_cp=1

# vLLM Generation Configuration
gen_tp=4
gen_dp=1
gpu_memory_utilization=0.5
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

# Data Configuration
DATA_CONFIG=(
    # File Paths
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # Data Structure
    data.prompt_key=prompt
    # Batch and Length Configuration
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # Preprocessing
    data.filter_overlong_prompts=False
    data.truncation='left'
)

# Model Configuration
MODEL_CONFIG=(
    # Model Path
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # Model Processing
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.trust_remote_code=True
)

# Reinforcement Learning Algorithm Configuration
ALGORITHM_CONFIG=(
    # Advantage Estimation
    algorithm.adv_estimator=${adv_estimator}
    # KL Divergence Control
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    # Megatron Parallelism Strategy
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # Transformer Architecture Optimizations
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

REF_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.ref.use_torch_compile=False
    # Log Probability Inference
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Megatron Parallelism Strategy
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

ROLLOUT_CONFIG=(
    # Rollout Engine
    actor_rollout_ref.rollout.name=vllm
    # Generation Parameters
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    # Log Probability Inference
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Memory Management
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    # Parallelism Strategy
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    # Performance Optimization
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    # Validation Generation
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
)

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RAY_DATA_HOME}/rollout_debug/${TIMESTAMP}"}

TRAINER_CONFIG=(
    # Logger Configuration
    trainer.logger='["console"]'
    # Project Settings
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    # Hardware Configuration
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    # Training Schedule
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    # Checkpoint Directory
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@"
