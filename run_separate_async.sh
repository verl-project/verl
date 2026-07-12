#!/usr/bin/env bash
# GRPO | Qwen3-8B | Megatron training | NVIDIA GPUs
#
# INFER_BACKEND controls rollout backend: vllm | sglang | trtllm.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/went/model/Qwen3-0.6B}

# trainer (actor/ref/critic) GPU group
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# standalone rollout GPU group (physically separate from trainer group)
ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-8}

# separate async requires train_batch_size == parameter_sync_step * ppo_mini_batch_size
# (one trainer step runs parameter_sync_step inner updates of ppo_mini_batch_size each)
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((1024 * 10))}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}
offload=${OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}
rollout_n=${ROLLOUT_N:-8}

# weight-sync backend between trainer and standalone rollout (must not be naive)
checkpoint_engine_backend=${CHECKPOINT_ENGINE_BACKEND:-nccl}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
parameter_sync_step=${PARAMETER_SYNC_STEP:-4}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-2}
test_freq=${TEST_FREQ:-10}
val_before_train=${VAL_BEFORE_TRAIN:-False}

project_name=${PROJECT_NAME:-went_sep_async}
experiment_name=${EXPERIMENT_NAME:-test_tq_ckpt}
default_local_dir=${DEFAULT_LOCAL_DIR:-/mnt/hdfs/went/checkpoint/${project_name}/${experiment_name}}

########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['/mnt/hdfs/went/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet']"
    data.val_files="['/mnt/hdfs/went/dataset/BytedTsinghua-SIA/AIME-2024/data/aime-2024.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
    actor_rollout_ref.rollout.checkpoint_engine.backend=${checkpoint_engine_backend}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.default_local_dir=${default_local_dir}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=${val_before_train}
    trainer.use_v1=True
    trainer.total_training_steps=5
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=${num_warmup_batches}
    trainer.v1.separate_async.parameter_sync_step=${parameter_sync_step}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
ray job submit \
    --working-dir . \
    -- python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"

