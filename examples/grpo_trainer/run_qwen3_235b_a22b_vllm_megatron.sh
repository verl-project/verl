#!/usr/bin/env bash
# GRPO scale demo | Qwen3-235B-A22B | vLLM rollout | Megatron training | NVIDIA GPUs
# Requires multi-node clusters (defaults: 8 nodes x 8 GPUs). Tune MEGATRON_* per your fabric.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-235B-A22B}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}   # path to Megatron dist checkpoint
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

actor_tp=${ACTOR_TP:-4}
actor_pp=${ACTOR_PP:-8}
actor_ep=${ACTOR_EP:-4}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-8}
rollout_ep=${ROLLOUT_EP:-64}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.75}
rollout_n=${ROLLOUT_N:-16}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_grpo_scale_demo}
experiment_name=${EXPERIMENT_NAME:-qwen3_235b_a22b_vllm_megatron}
CKPTS_DIR=${CKPTS_DIR:-.ckpt}
# ---- end user-adjustable ----

train_files=$HOME/data/gsm8k/train.parquet
val_files=$HOME/data/gsm8k/test.parquet

dist_ckpt_args=""
if [ -n "$MCORE_MODEL_PATH" ]; then
    dist_ckpt_args="actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True"
fi

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.actor.megatron.param_offload=${all_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.ref.megatron.param_offload=${all_offload} \
    actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    ${dist_ckpt_args} \
    actor_rollout_ref.nccl_timeout=7200 \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" "$@"
