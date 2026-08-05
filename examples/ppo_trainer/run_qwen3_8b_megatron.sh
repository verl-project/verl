#!/usr/bin/env bash
# PPO | text | vLLM rollout | Megatron training | NVIDIA GPUs

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
critic_lr=${CRITIC_LR:-1e-5}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-2}
critic_tp=${CRITIC_TP:-2}
critic_pp=${CRITIC_PP:-2}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-1}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_ppo_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_vllm_megatron}

gsm8k_train=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
gsm8k_test=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
math_train=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
math_test=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=gae
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
)

CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.optim.lr=${critic_lr}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    critic.megatron.tensor_model_parallel_size=${critic_tp}
    critic.megatron.pipeline_model_parallel_size=${critic_pp}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
