#!/usr/bin/env bash
# GRPO | Qwen3-0.6B | GSM8K | TorchTitan Training & vLLM Rollout | TPU v6e-8 x2 Slices
# V1 PPOTrainer (Separate Async Overlap) — requires verl-hardware-plugin:
#   ray job submit --working-dir . --runtime-env-json '{
#     "py_modules": ["/path/to/verl-hardware-plugin/verl_hardware_plugin"],
#     "env_vars": {"VERL_PLATFORM": "tpu", "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin", ...}
#   }' -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
# See examples/tpu/grpo/README.md for full instructions.
#
# By default this runs a realistic 100-step GRPO job whose reward curve actually
# moves. Set SMOKE_TEST=1 for the 5-step configuration used to validate that the
# stack comes up (it trains nothing useful).
#
# The settings that separate the two modes, and why they matter:
#
#   train_batch_size       4  -> 32    4 prompts/step is far too noisy to show a trend.
#   max_response_length  512  -> 1024  At 512 the smoke test truncated 87.5% of responses
#                                      (`response_length/clip_ratio: 0.875`), so the model
#                                      was cut off before emitting the `#### <answer>`
#                                      line and scored zero regardless of correctness.
#   total_training_steps   5  -> 100   Enough steps for the reward curve to move.
#
# Both modes sample rollout.n=8 responses per prompt. GRPO estimates the advantage as the
# spread of rewards *within* a group of samples for the same prompt. With n=2 the group is
# almost always all-correct or all-wrong, the advantage collapses to zero and no gradient
# flows: one smoke run at n=2 had grad_norm 0.0 on all 5 steps, so it never updated the
# actor or sent changed weights to the rollout.
#
# Parallelism: the actor runs pure FSDP (tensor_parallel_size=1,
# data_parallel_shard_size=8). Do not re-enable tensor parallelism without re-testing.
# Under tensor_parallel_size=2 the actor produced non-finite gradients on most steps, and
# because optimizer_step() silently skips the update when grad_norm is not finite, the job
# still reported SUCCEEDED while the policy never changed. The smoke config (then at
# rollout.n=2) gave grad_norm 1.47 / 0.0 / 1.65 / 0.0 / 0.0 at tp=1 and
# inf / 8.3e37 / 3.8e24 at tp=2.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# JAX/XLA Launch Barrier Configuration
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

SMOKE_TEST="${SMOKE_TEST:-0}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
    exp_name="${EXPERIMENT_NAME:-qwen3_0.6b_fast_smoke_test}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
    TEST_FREQ="${TEST_FREQ:-2}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
else
    exp_name="${EXPERIMENT_NAME:-qwen3_0.6b_gsm8k}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
    TEST_FREQ="${TEST_FREQ:-10}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
fi

# Project details
project_name='verl_tpu_grpo'

# Paths
RAY_DATA_HOME="/data/jialei"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-0.6B}"

TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"

# TPU 2-slice v6e-8 configurations
export NNODES_TRAINER=2       # 2 physical VM hosts for training slice
export N_CHIPS_TRAINER=4      # 4 TPU chips per training host

export NNODES_ROLLOUT=2       # 2 physical VM hosts for rollout slice
export N_CHIPS_ROLLOUT=4      # 4 TPU chips per rollout host

TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))

# Sequence budget. max_model_len must cover prompt + response, otherwise vLLM
# silently truncates the generation and the reward is always zero.
MAX_PROMPT_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

# Actor parallelism. Pure FSDP is the only configuration that has been validated
# on TPU; see the note at the top of this file before changing these.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-8}"

if [[ "${TENSOR_PARALLEL_SIZE}" != "1" ]]; then
    set +x
    echo "=============================================================================" >&2
    echo "WARNING: tensor_parallel_size=${TENSOR_PARALLEL_SIZE} is NOT supported on TPU." >&2
    echo "  Tensor parallelism has not been properly tested with this stack and is known" >&2
    echo "  to produce non-finite (nan/inf) actor gradients. Because optimizer_step()" >&2
    echo "  skips the update whenever grad_norm is not finite, the job will still report" >&2
    echo "  SUCCEEDED while the policy silently never trains." >&2
    echo "  Use TENSOR_PARALLEL_SIZE=1 with DATA_PARALLEL_SHARD_SIZE=<num actor chips>." >&2
    echo "=============================================================================" >&2
    set -x
fi

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TOTAL_ROLLOUT_CHIPS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=/tmp/verl_dump/rollout \
    trainer.validation_data_dir=/tmp/verl_dump/validation \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"
