#!/usr/bin/env bash
# GRPO | text | vLLM rollout | FSDP training | 1x NVIDIA H20 (96GB)
# Qwen2.5-0.5B-Instruct on GSM8K, with two memory/throughput tiers selected at
# runtime via the TIER env var (nothing extra in the filename):
#   TIER=recommended (default) — throughput: no actor offload, no gradient
#       checkpointing, dynamic token-based batching (32K/GPU), vLLM share 0.7,
#       chunked prefill.  Measured: 74.6 GB peak, ~21 s/step.
#   TIER=min — minimum resources: optimizer CPU offload + gradient
#       checkpointing + micro batch 2 + vLLM share 0.35.  Measured: 38.9 GB
#       peak, ~80 s/step (needs ~387 GB host RAM for the offload).
# Both tiers train the same task/lengths; learning curves are equivalent
# (GSM8K 0.547 vs 0.539 @100 steps). Details: examples/tuning/0.5b/h20_results.md
# (reproduce via scripts/run_h20_benchmark.sh).

set -xeuo pipefail

TIER=${TIER:-recommended}
case "$TIER" in
    min | recommended) ;;
    *)
        echo "TIER must be 'min' or 'recommended', got '$TIER'" >&2
        exit 2
        ;;
esac

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

if [ "$TIER" = "min" ]; then
    d_train_batch=32
    d_mini_batch=16
    d_rollout_n=4
    d_gpu_util=0.35
    d_save_freq=100
else
    d_train_batch=64
    d_mini_batch=32
    d_rollout_n=8
    d_gpu_util=0.7
    d_save_freq=20
fi

train_batch_size=${TRAIN_BATCH_SIZE:-$d_train_batch}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$d_mini_batch}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-$d_gpu_util}
rollout_n=${ROLLOUT_N:-$d_rollout_n}

total_epochs=${TOTAL_EPOCHS:-15}
total_training_steps=${TOTAL_TRAINING_STEPS:--1}
save_freq=${SAVE_FREQ:-$d_save_freq}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_tuning_h20}
experiment_name=${EXPERIMENT_NAME:-qwen2.5-0.5b_grpo_1xh20_${TIER}}
logger=${LOGGER:-'["console"]'}
# ---- end user-adjustable ----

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$HOME/data/gsm8k/train.parquet
    data.val_files=$HOME/data/gsm8k/test.parquet
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
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

if [ "$TIER" = "min" ]; then
    MODEL+=(actor_rollout_ref.model.enable_gradient_checkpointing=True)
    ACTOR+=(
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    )
    ROLLOUT+=(actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu})
    REF+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu})
else
    MODEL+=(actor_rollout_ref.model.enable_gradient_checkpointing=False)
    ACTOR+=(
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    )
    ROLLOUT+=(
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    )
    REF+=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    )
fi

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="${logger}"
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

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm x fsdp extras of the
# committed uv.lock. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

PYTHONUNBUFFERED=1 "${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@"
