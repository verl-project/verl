#!/usr/bin/env bash
# Short native FSDP2 Muon GRPO run for an already-downloaded, small dense text model.
# Run from the repository root in the prepared training environment.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to an absolute local model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to an absolute local training parquet file}"
: "${VAL_FILE:?Set VAL_FILE to an absolute local validation parquet file}"

if [[ "$MODEL_PATH" != /* || ! -f "$MODEL_PATH/config.json" ]]; then
    echo "MODEL_PATH must be an existing absolute model directory with config.json." >&2
    exit 1
fi
if [[ "$TRAIN_FILE" != /* || ! -f "$TRAIN_FILE" || "$VAL_FILE" != /* || ! -f "$VAL_FILE" ]]; then
    echo "TRAIN_FILE and VAL_FILE must be existing absolute local parquet paths." >&2
    exit 1
fi

# Keep this launcher local-only; prepare model, tokenizer, and data before running it.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TOTAL_STEPS=${TOTAL_STEPS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
ACTOR_LR=${ACTOR_LR:-1e-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dense_fsdp2_muon_smoke}

python3 -m verl.trainer.main_ppo \
    model_engine=dp \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "data.train_files=['$TRAIN_FILE']" \
    "data.val_files=['$VAL_FILE']" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    data.seed=42 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.optimizer=Muon \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    trainer.logger='["console"]' \
    trainer.project_name=fsdp2_muon \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@"
