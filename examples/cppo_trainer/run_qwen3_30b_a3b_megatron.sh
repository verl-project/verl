#!/usr/bin/env bash
# CPPO | MoE | vLLM rollout | Megatron training | NVIDIA GPUs
# CPPO replaces DPPO's uniform per-token divergence threshold with a position-weighted
# threshold and a cumulative prefix budget (paper: cppo.pdf, "Beyond Uniform Token-Level
# Trust Region in LLM Reinforcement Learning"). Binary-TV variant.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# CPPO hyperparameters (paper Table 3, Qwen3-30B-A3B-Base / MoE).
# clip_ratio is the token-level divergence threshold scale delta (same field DPPO uses):
#   delta = 0.20 for the 30B-A3B MoE model (0.15 for dense models).
clip_ratio=${CLIP_RATIO:-0.20}
# Position weight floor w_t in [w_min, 1].
cppo_w_min=${CPPO_W_MIN:-0.8}
# Floor delta_b_min of the per-sequence dynamic prefix budget
# delta_b = clamp(cppo_delta_b_k * quantile(D_t, cppo_delta_b_q), delta_b_min, 2*delta_b_min).
cppo_delta_b=${CPPO_DELTA_B:-0.02}
# Quantile and scale of the budget calibration; (0.9, 1.0) = paper P90,
# e.g. CPPO_DELTA_B_Q=0.95 CPPO_DELTA_B_K=0.5 uses half the 95th percentile.
cppo_delta_b_q=${CPPO_DELTA_B_Q:-0.9}
cppo_delta_b_k=${CPPO_DELTA_B_K:-1.0}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-4}
actor_pp=${ACTOR_PP:-1}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
rollout_n=${ROLLOUT_N:-16}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_cppo_qwen3_moe}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_cppo_vllm_megatron}
# ---- end user-adjustable ----

train_file=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
val_file=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=False
    # CPPO measures the divergence D_t between the rollout policy mu and the current
    # policy pi, so old_log_probs must be the rollout log-probs (no recomputed pi_old).
    # bypass_mode=True sets old_log_probs = rollout_log_probs; for divergence losses
    # (cppo / dppo_tv / dppo_kl) the trainer keeps loss_mode unchanged and runs their mask.
    algorithm.rollout_correction.bypass_mode=True
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
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
    actor_rollout_ref.actor.policy_loss.loss_mode=cppo
    actor_rollout_ref.actor.policy_loss.cppo.cppo_w_min=${cppo_w_min}
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b=${cppo_delta_b}
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b_q=${cppo_delta_b_q}
    actor_rollout_ref.actor.policy_loss.cppo.cppo_delta_b_k=${cppo_delta_b_k}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.clip_ratio=${clip_ratio}
    actor_rollout_ref.actor.clip_ratio_c=20.0
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
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
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=megatron
    # Synchronous TransferQueue trainer. In this trainer bypass_mode=True simply sets
    # old_log_probs = rollout_log_probs and leaves loss_mode untouched, so CPPO reads the
    # rollout policy mu directly and runs its own mask (no core-trainer patch needed).
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=1
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo_sync \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
