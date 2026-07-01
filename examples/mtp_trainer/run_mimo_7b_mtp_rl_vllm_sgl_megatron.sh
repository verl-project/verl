#!/usr/bin/env bash
# Reference: https://github.com/THUDM/slime/blob/main/scripts/run-mimo-7B-rl-eagle.sh
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_PATH=${MODEL_PATH:-XiaomiMiMo/MiMo-7B-RL}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

rollout_batch_size=${ROLLOUT_BATCH_SIZE:-32}
n_samples_per_prompt=${N_SAMPLES_PER_PROMPT:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR:-0.2}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
actor_cp=${ACTOR_CP:-1}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# Rollout backend: "sglang" (default) or "vllm".
rollout_backend=${ROLLOUT_BACKEND:-sglang}

# Set MTP_ROLLOUT_SPEC=0 to disable rollout-time speculative decoding entirely.
mtp_rollout_spec=${MTP_ROLLOUT_SPEC:-1}

# vLLM-only: number of speculative tokens (vLLM's MTP method draws this many
# tokens per verify step). Mirrors slime's effective draft budget.
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-3}

# sglang+EAGLE-only: matches slime/scripts/run-mimo-7B-rl-eagle.sh defaults.
spec_algorithm=${SPEC_ALGORITHM:-EAGLE}
spec_num_steps=${SPEC_NUM_STEPS:-3}
spec_eagle_topk=${SPEC_EAGLE_TOPK:-1}
spec_num_draft_tokens=${SPEC_NUM_DRAFT_TOKENS:-4}

total_epochs=${TOTAL_EPOCHS:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-3000}
save_freq=${SAVE_FREQ:-2000}
test_freq=${TEST_FREQ:-20}

project_name=${PROJECT_NAME:-mtp}
experiment_name=${EXPERIMENT_NAME:-mimo_7b_mtp_${rollout_backend}_megatron}
# ---- end user-adjustable ----

train_file=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
val_file=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${rollout_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.mtp.enable=True
    actor_rollout_ref.model.mtp.enable_train=True
    actor_rollout_ref.model.mtp.detach_encoder=True
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=${mtp_loss_scaling_factor}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.betas=[0.9,0.98]
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_backend}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${n_samples_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

if [ "${mtp_rollout_spec}" = "1" ]; then
    ROLLOUT+=(actor_rollout_ref.model.mtp.enable_rollout=True)
    if [ "${rollout_backend}" = "vllm" ]; then
        ROLLOUT+=(
            actor_rollout_ref.model.mtp.method=mtp
            actor_rollout_ref.model.mtp.num_speculative_tokens=${num_speculative_tokens}
        )
    elif [ "${rollout_backend}" = "sglang" ]; then
        ROLLOUT+=(
            actor_rollout_ref.model.mtp.speculative_algorithm=${spec_algorithm}
            actor_rollout_ref.model.mtp.speculative_num_steps=${spec_num_steps}
            actor_rollout_ref.model.mtp.speculative_eagle_topk=${spec_eagle_topk}
            actor_rollout_ref.model.mtp.speculative_num_draft_tokens=${spec_num_draft_tokens}
        )
    else
        echo "Unknown ROLLOUT_BACKEND=${rollout_backend}; expected 'vllm' or 'sglang'" >&2
        exit 1
    fi
fi

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.ref.megatron.param_offload=True
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${max_response_length}
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
    trainer.total_training_steps=${total_training_steps}
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
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
