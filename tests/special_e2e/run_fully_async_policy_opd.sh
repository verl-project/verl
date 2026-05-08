#!/usr/bin/env bash
set -xeuo pipefail

# Workaround for NVIDIA driver bug (r560-r575) causing SIGSEGV in ncclCuMemHostEnable()
# on PCIe machines without P2P access. See: https://github.com/NVIDIA/nccl/issues/1838
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

# Test script for fully_async_policy + Multi-Teacher Online Policy Distillation (OPD)
# This script runs fully async training with Megatron backend and multiple standalone
# teacher models to verify that multi-teacher distillation works correctly in async mode.
#
# Follows PR #6051 (Multi-Teacher OPD) setup:
#   Student: Qwen3-VL-2B-Instruct (Megatron, tp=2, pp=2)
#   GSM8K Teacher: Qwen3-4B-Instruct-2507
#   Geo3K Teacher: Qwen3-VL-4B-Instruct
#
# GPU allocation (8 GPUs):
#   - 2 GPU: Rollout (student vLLM async, gen_tp=2)
#   - 4 GPU: Training (student Megatron, tp=2 x pp=2)
#   - 1 GPU: Teacher GSM8K (standalone vLLM)
#   - 1 GPU: Teacher Geo3K (standalone vLLM)
#
# Usage:
#   cd /root/verl && bash tests/special_e2e/run_fully_async_policy_opd.sh

############################ Quick Config ############################

ROLLOUT_NAME="vllm"
export VLLM_USE_V1=1

STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
GSM8K_TEACHER_MODEL_ID=${GSM8K_TEACHER_MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}
GEO3K_TEACHER_MODEL_ID=${GEO3K_TEACHER_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}

STUDENT_MODEL=${STUDENT_MODEL:-${HOME}/models/${STUDENT_MODEL_ID}}
GSM8K_TEACHER_MODEL=${GSM8K_TEACHER_MODEL:-${HOME}/models/${GSM8K_TEACHER_MODEL_ID}}
GEO3K_TEACHER_MODEL=${GEO3K_TEACHER_MODEL:-${HOME}/models/${GEO3K_TEACHER_MODEL_ID}}

DISTILLATION_LOSS_MODE="k1"
USE_POLICY_GRADIENT=True

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=2048
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# Fully async specific
N_GPUS_ROLLOUT=2
N_GPUS_TRAINING=4
N_GPUS_TEACHER_TOTAL=2  # 1 per teacher
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-128}

# Megatron parallelism
GEN_TP=2
TRAIN_TP=2
TRAIN_PP=2

STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4

############################ Data Preparation ############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GSM8K_DIR="${HOME}/data/gsm8k"
GEO3K_DIR="${HOME}/data/geo3k"

# Prepare GSM8K (idempotent)
if [ ! -f "${GSM8K_DIR}/train.parquet" ]; then
    echo "Preparing GSM8K dataset..."
    python3 "${VERL_ROOT}/examples/data_preprocess/gsm8k.py" --local_save_dir "$GSM8K_DIR"
fi

# Prepare Geo3K (idempotent)
if [ ! -f "${GEO3K_DIR}/train.parquet" ]; then
    echo "Preparing Geo3K dataset..."
    python3 "${VERL_ROOT}/examples/data_preprocess/geo3k.py" --local_save_dir "$GEO3K_DIR"
fi

GSM8K_TRAIN="${GSM8K_DIR}/train.parquet"
GSM8K_TEST="${GSM8K_DIR}/test.parquet"
GEO3K_TRAIN="${GEO3K_DIR}/train.parquet"
GEO3K_TEST="${GEO3K_DIR}/test.parquet"

TRAIN_FILES="['${GSM8K_TRAIN}','${GEO3K_TRAIN}']"
TEST_FILES="['${GSM8K_TEST}','${GEO3K_TEST}']"

############################ Detect Device ############################

device_name=$(python3 - <<'EOF'
from verl.utils.device import get_device_name
print(get_device_name())
EOF
)

ACTOR_OFFLOAD=True

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=0
    data.gen_batch_size=1
    data.return_raw_chat=True
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

STUDENT=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.lr_decay_steps=10000000
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_OFFLOAD}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl'
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

# Multi-teacher: one teacher per dataset, routed by the sample's `data_source` value.
DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=${N_GPUS_TEACHER_TOTAL}
    distillation.nnodes=1
    # --- gsm8k teacher (text-only) ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="${GSM8K_TEACHER_MODEL}"
    +distillation.teacher_models.gsm8k.num_replicas=1
    +distillation.teacher_models.gsm8k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False
    +distillation.teacher_models.gsm8k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=$MAX_NUM_TOKENS
    # --- geo3k teacher (vision-language) ---
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
    +distillation.teacher_models.geo3k.model_path="${GEO3K_TEACHER_MODEL}"
    +distillation.teacher_models.geo3k.num_replicas=1
    +distillation.teacher_models.geo3k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.geo3k.inference.enforce_eager=False
    +distillation.teacher_models.geo3k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.engine_kwargs.vllm.mm_processor_cache_gb=0
    # --- loss ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name='verl-test-fully-async-opd'
    trainer.experiment_name="qwen3-vl-2b-fully-async-multi-teacher-opd"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=10
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=2
    trainer.test_freq=-1
)

ASYNC_TRAINING=(
    rollout.nnodes=1
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}
    async_training.staleness_threshold=${STALENESS_THRESHOLD}
    async_training.partial_rollout=True
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}
    async_training.use_trainer_do_validate=False
)

############################ Launch ############################

echo "Running fully_async_policy + Multi-Teacher OPD"
echo "Student: ${STUDENT_MODEL}"
echo "Teacher GSM8K: ${GSM8K_TEACHER_MODEL}"
echo "Teacher Geo3K: ${GEO3K_TEACHER_MODEL}"
echo "GPUs: ${N_GPUS_ROLLOUT} rollout + ${N_GPUS_TRAINING} training + ${N_GPUS_TEACHER_TOTAL} teachers"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    actor_rollout_ref.hybrid_engine=False \
    critic.strategy=megatron \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${STUDENT[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "$@"

echo "Fully async policy + Multi-Teacher OPD E2E test completed successfully"
