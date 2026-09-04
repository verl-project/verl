#!/usr/bin/env bash
set -euo pipefail

# Run through run_qwen38_lora_dapo_slurm.sh inside the pinned Enroot image.
: "${TRAIN_MODEL_PATH:?set the BF16 trainer checkpoint}"
: "${ROLLOUT_MODEL_PATH:?set the matching FP8 rollout checkpoint}"
: "${TRAIN_FILE:?set the DAPO-Math training parquet}"
: "${VAL_FILE:?set the DAPO-Math validation parquet}"
: "${OUTPUT_DIR:?set the checkpoint directory}"
: "${UV_PROJECT_ENVIRONMENT:?set the baked uv environment}"

NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
FSDP_SIZE=${FSDP_SIZE:-8}
TRAINING_STEPS=${TRAINING_STEPS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
ENABLE_MTP=${ENABLE_MTP:-true}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.76}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}
USE_LIGER=${USE_LIGER:-false}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-false}
RESHARD_AFTER_FORWARD=${RESHARD_AFTER_FORWARD:-true}
USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION=${USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION:-false}
PAD_TO_LENGTH=${PAD_TO_LENGTH:-false}
PAD_TO_LENGTH_BUCKET=${PAD_TO_LENGTH_BUCKET:-1024}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
ROLLOUT_CACHE_DIR=${ROLLOUT_CACHE_DIR:-}
ROLLOUT_CACHE_STEPS=${ROLLOUT_CACHE_STEPS:-'[11,12]'}
ROLLOUT_CACHE_ACTION=${ROLLOUT_CACHE_ACTION:-cache}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-17917}
EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-30}
PROJECT_NAME=${PROJECT_NAME:-qwen38_full_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-fsdp16_lora_r32_mtp_eager_4k}

TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
if ((TOTAL_GPUS <= 0 || TOTAL_GPUS % ROLLOUT_TP != 0)); then
    echo "the positive GPU count must be divisible by rollout TP" >&2
    exit 2
fi
if ((FSDP_SIZE != -1 &&
      (FSDP_SIZE <= 0 || FSDP_SIZE > TOTAL_GPUS || TOTAL_GPUS % FSDP_SIZE != 0))); then
    echo "FSDP_SIZE must be -1 or a positive divisor of total GPUs" >&2
    exit 2
fi
if ((TRAIN_BATCH_SIZE <= 0 || PPO_MINI_BATCH_SIZE <= 0 ||
      PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE || ROLLOUT_N <= 1 ||
      TRAINING_STEPS <= 0)); then
    echo "invalid training batch, rollout, or step count" >&2
    exit 2
fi
if ((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN)); then
    echo "prompt plus response length exceeds model length" >&2
    exit 2
fi
if ((MAX_NUM_BATCHED_TOKENS <= 0 || MAX_NUM_SEQS <= 0)); then
    echo "rollout token and sequence limits must be positive" >&2
    exit 2
fi
if ((LORA_RANK <= 0 || LORA_ALPHA <= 0 ||
      SAVE_FREQ == 0 || SAVE_FREQ < -1 || TEST_FREQ == 0 || TEST_FREQ < -1)); then
    echo "LoRA dimensions must be positive; frequencies must be -1 or positive" >&2
    exit 2
fi
if [[ "$RESUME_MODE" != auto && "$RESUME_MODE" != disable &&
      "$RESUME_MODE" != resume_path ]]; then
    echo "RESUME_MODE must be auto, disable, or resume_path" >&2
    exit 2
fi
if [[ "$RESUME_MODE" == resume_path && -z "$RESUME_FROM_PATH" ]]; then
    echo "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 2
fi
if [[ "$ROLLOUT_CACHE_ACTION" != cache && "$ROLLOUT_CACHE_ACTION" != repeat ]]; then
    echo "ROLLOUT_CACHE_ACTION must be cache or repeat" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
UV_PY=(uv run --active --frozen --no-sync python)

read_rows() {
    "${UV_PY[@]}" -c \
        'import pyarrow.parquet as p, sys; print(p.ParquetFile(sys.argv[1]).metadata.num_rows)' \
        "$1"
}
train_rows=$(read_rows "$TRAIN_FILE")
val_rows=$(read_rows "$VAL_FILE")
if [[ "$train_rows" != "$EXPECTED_TRAIN_ROWS" ||
      "$val_rows" != "$EXPECTED_VAL_ROWS" ]]; then
    echo "dataset cardinality differs: train=$train_rows/$EXPECTED_TRAIN_ROWS val=$val_rows/$EXPECTED_VAL_ROWS" >&2
    exit 2
fi

args=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.dataloader_num_workers=0
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$TRAIN_MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING"
    actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING"
    actor_rollout_ref.model.use_liger="$USE_LIGER"
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS"
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj]'
    actor_rollout_ref.model.lora.merge=False
    actor_rollout_ref.model.mtp.enable="$ENABLE_MTP"
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout="$ENABLE_MTP"
    actor_rollout_ref.model.mtp.method=mtp
    actor_rollout_ref.model.mtp.num_speculative_tokens=3
    actor_rollout_ref.actor.strategy=fsdp2
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$FSDP_SIZE"
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward="$RESHARD_AFTER_FORWARD"
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation="$USE_NO_SYNC_FOR_GRADIENT_ACCUMULATION"
    actor_rollout_ref.actor.fsdp_config.pad_to_length="$PAD_TO_LENGTH"
    actor_rollout_ref.actor.fsdp_config.pad_to_length_bucket="$PAD_TO_LENGTH_BUCKET"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.model_path="$ROLLOUT_MODEL_PATH"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    actor_rollout_ref.rollout.seed=17
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config='{\"inductor_compile_config\":{\"triton.autotune_at_compile_time\":false}}'"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    trainer.logger='["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.total_epochs=1
    trainer.total_training_steps="$TRAINING_STEPS"
    trainer.val_before_train="$VAL_BEFORE_TRAIN"
    trainer.resume_mode="$RESUME_MODE"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.default_local_dir="$OUTPUT_DIR"
    'ray_kwargs.ray_init.runtime_env.py_executable=/opt/verl-uv-final/bin/python'
    hydra.run.dir=/run/hydra
)
if [[ "$RESUME_MODE" == resume_path ]]; then
    args+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi
if [[ "$USE_DYNAMIC_BSZ" == true ]]; then
    args+=(
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    )
else
    args+=(
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU"
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU"
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU"
    )
fi
if [[ -n "$ROLLOUT_CACHE_DIR" ]]; then
    args+=(
        skip.rollout_tq.enable=True
        skip.rollout_tq.dump_dir="$ROLLOUT_CACHE_DIR"
        skip.rollout_tq.steps="$ROLLOUT_CACHE_STEPS"
        skip.rollout_tq.action="$ROLLOUT_CACHE_ACTION"
    )
fi
if [[ -n "${RAY_ADDRESS:-}" ]]; then
    args+=(+ray_kwargs.ray_init.address="$RAY_ADDRESS")
else
    args+=(ray_kwargs.ray_init.num_cpus=48 +ray_kwargs.ray_init.num_gpus="$TOTAL_GPUS")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    args+=(--cfg job --resolve)
fi

exec "${UV_PY[@]}" -m verl.trainer.main_ppo "${args[@]}" "$@"
