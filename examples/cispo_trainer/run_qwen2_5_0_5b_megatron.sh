#!/usr/bin/env bash
# CISPO | Qwen2.5-0.5B-Instruct | Megatron training | vLLM-Ascend rollout | Ascend NPUs

set -xeuo pipefail

########################### environment ###########################

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}


########################### user-adjustable ###########################

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_ROOT=${DATA_ROOT:-${HOME}/data}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
micro_batch_size=${MICRO_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
max_token_length=${MAX_TOKEN_LENGTH:-$((max_prompt_length + max_response_length))}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}
seed=${SEED:-42}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
clip_ratio_low=${CLIP_RATIO_LOW:-10.0}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
rollout_tp=${ROLLOUT_TP:-2}
rollout_n=${ROLLOUT_N:-4}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
weight_bucket_mb=${WEIGHT_BUCKET_MB:-512}

rollout_world_size=$((NNODES * NPUS_PER_NODE))
if (( rollout_tp <= 0 || rollout_world_size % rollout_tp != 0 )); then
    echo "ROLLOUT_TP must be a positive divisor of NNODES * NPUS_PER_NODE." >&2
    exit 2
fi
if (( rollout_n < 2 )); then
    echo "CISPO requires ROLLOUT_N >= 2." >&2
    exit 2
fi
rollout_replicas=$((rollout_world_size / rollout_tp))
default_rollout_max_num_seqs=$(((train_batch_size * rollout_n + rollout_replicas - 1) / rollout_replicas))
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-${default_rollout_max_num_seqs}}

offload=${OFFLOAD:-False}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}
resume_mode=${RESUME_MODE:-auto}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}

project_name=${PROJECT_NAME:-verl_cispo_gsm8k}
experiment_name=${EXPERIMENT_NAME:-qwen2_5_0_5b_cispo_megatron_vllm_ascend}
output_dir=${OUTPUT_DIR:-${PWD}/checkpoints/${experiment_name}}
log_dir=${LOG_DIR:-${PWD}/logs}
mkdir -p "${output_dir}" "${log_dir}"

train_files="['${DATA_ROOT}/gsm8k/train.parquet']"
val_files="['${DATA_ROOT}/gsm8k/test.parquet']"

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

DATA=(
    "data.train_files=${train_files}"
    "data.val_files=${val_files}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.dataloader_num_workers=${dataloader_num_workers}
    data.seed=${seed}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_length}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.max_model_len=${max_token_length}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${weight_bucket_mb}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_length}
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_length}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.resume_mode=${resume_mode}
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    "trainer.default_local_dir=${output_dir}"
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################

log_file="${log_dir}/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${log_file}"
