#!/bin/bash
set -xeuo pipefail
mkdir -p logs

# Project Configuration
project_name='GRPO-Qwen2.5-32B-BASE-SGLang'
exp_name='GRPO-Qwen2.5-32B-BASE-FSDP-SGLang'

# Necessary env
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# If the number of nodes is 16, ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DISABLE_L2_CACHE=1
export TASK_QUEUE_ENABLE=1

# Node Info
NNODES=${NNODES:-2}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# Model Weights Paths
MODEL_PATH=Qwen/Qwen2.5-32B
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/datasets/deepscaler/test.parquet"}

# Data Configuration
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# Training Batch Configuration
train_prompt_bsz=32
train_prompt_mini_bsz=32
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

# SGLang Configuration
gen_tp=4
gen_sp=1
gen_dp=1
gen_ep=1
gpu_memory_utilization=0.5

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
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# Reinforcement Learning Algorithm Configuration
ALGORITHM_CONFIG=(
    # Advantage Estimation
    algorithm.adv_estimator=${adv_estimator}
    # KL Divergence Control
    algorithm.use_kl_in_reward=${use_kl_in_reward}
)

# Actor Model Configuration
ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.fsdp_config.param_offload=${all_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${all_offload}
    )

# Reference Model Configuration
REF_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.ref.use_torch_compile=False
    # Log Probability Inference
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    # Memory Optimization
    actor_rollout_ref.ref.fsdp_config.param_offload=${all_offload}
)

# Rollout Configuration
ROLLOUT_CONFIG=(
    # Rollout Engine
    actor_rollout_ref.rollout.name=sglang
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
    # Generation Parameters
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    # Log Probability Inference
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    # Memory Management
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    # Validation Generation
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.nccl_timeout=1800
)

# Trainer Configuration
TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.total_epochs=5
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=100
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.critic_warmup=0
)

# Main GRPO Training Command
# Add the reward function processing for the DeepScaler dataset here
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    custom_reward_function.path=recipe/r1_ascend/deepscaler.py \
    custom_reward_function.name=compute_score \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@" | tee logs/run_qwen2_5-32b_grpo_fsdp_sglang_npu.log