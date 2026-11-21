#!/usr/bin/env bash
# GRPO | Qwen3-8B | gsm8k | Megatron training | MXFP8 end to end | NVIDIA Blackwell (SM100+)
#
# PRECISION=mxfp8 trains with Transformer Engine's MXFP8 recipe and serves the rollout in MXFP8;
# PRECISION=bf16 runs the same job in BF16, as the baseline. INFER_BACKEND controls the rollout backend: vllm | sglang.
# The defaults fit one B200. See docs/low_precision/mxfp8.md.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}
PRECISION=${PRECISION:-mxfp8}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}
# offload parameters and optimizer state while the rollout runs
offload=${OFFLOAD:-True}
# keep the whole optimizer on the CPU: on one GPU, Adam's FP32 state for 8B (~96 GiB) does not fit next to training
optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_mxfp8}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_megatron_${INFER_BACKEND}_${PRECISION}}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

case "${PRECISION}" in
    mxfp8)
        PRECISION_ARGS=(
            # MXFP8 GEMMs in training (Transformer Engine); the weights stay in BF16
            ++actor_rollout_ref.actor.megatron.override_transformer_config.fp8=e4m3
            ++actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=mxfp8
            # MXFP8 rollout: verl quantizes the synced weights; lm_head, embeddings and MoE routers stay in BF16
            actor_rollout_ref.rollout.quantization.linear=MXFP8
            actor_rollout_ref.rollout.quantization.moe=MXFP8
        )
        ;;
    bf16)
        PRECISION_ARGS=()
        ;;
    *)
        echo "PRECISION must be mxfp8 or bf16, got '${PRECISION}'" >&2
        exit 1
        ;;
esac

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
)

OPTIM=+actor_rollout_ref.actor.optim.override_optimizer_config
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    ${OPTIM}.optimizer_cpu_offload=${optimizer_cpu_offload}
    ${OPTIM}.overlap_cpu_optimizer_d2h_h2d=${optimizer_cpu_offload}
    ${OPTIM}.use_precision_aware_optimizer=${optimizer_cpu_offload}
    ${OPTIM}.optimizer_offload_fraction=1.0
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): GPU vllm/sglang × megatron run the driver and every Ray worker
# (runtime_env.py_executable) through `uv run` on the matching extras of the committed uv.lock;
# other backends / NPU fall back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${INFER_BACKEND}" = vllm ] || [ "${INFER_BACKEND}" = sglang ]; }; then
    LAUNCH=(uv run --frozen --all-packages --extra "${INFER_BACKEND}" --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${INFER_BACKEND} --extra megatron")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${PRECISION_ARGS[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
