#!/usr/bin/env bash
# GRPO on XR-chest with Qwen3-VL-4B (from SFT checkpoint).
#
# Usage:
#   bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh             # full run on train.parquet
#   SMOKE=1 bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh     # smoke on train_mini.parquet
#   DRY_RUN=1 bash recipe/xr_chest/run_qwen3vl_4b_grpo.sh   # startup-only (total_epochs=0)
#
# Must be launched from the verl repo root.
set -euxo pipefail

PYBIN=${PYBIN:-python3}
ENGINE=${ENGINE:-vllm}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}

DATA_DIR=${DATA_DIR:-"${HOME}/data/xr_chest_grpo"}

if [[ "${SMOKE}" == "1" ]]; then
    TRAIN_FILE="${DATA_DIR}/train_mini.parquet"
    EXP_TAG="smoke"
    SAVE_FREQ=10
    TOTAL_EPOCHS=1
    # train_mini is 32 rows; shrink batch so drop_last still yields a step.
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
else
    TRAIN_FILE="${DATA_DIR}/train.parquet"
    EXP_TAG="full"
    SAVE_FREQ=20
    TOTAL_EPOCHS=1
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
fi
VAL_FILE="${DATA_DIR}/valid_mini.parquet"

if [[ "${DRY_RUN}" == "1" ]]; then
    TOTAL_EPOCHS=0
    EXP_TAG="${EXP_TAG}-dryrun"
fi

MODEL_PATH=${MODEL_PATH:-"/mnt/nfs/home/michael/lf_outputs/qwen3vl/xr-qwen3vl-4b-full_all_lowres-seed42/checkpoint-6398"}

PROJECT_NAME=${PROJECT_NAME:-"pillar-rl"}
EXP_NAME=${EXP_NAME:-"xr-chest-qwen3vl-4b-grpo-${EXP_TAG}"}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${PROJECT_NAME}/${EXP_NAME}"}

# Pixel budget — must match SFT (full_all_unlocked_lowres.yaml).
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-131072}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-1024}

"${PYBIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    data.custom_cls.path=recipe/xr_chest/xr_dataset.py \
    data.custom_cls.name=XRChestRLHFDataset \
    +data.image_max_pixels=${IMAGE_MAX_PIXELS} \
    +data.image_min_pixels=${IMAGE_MIN_PIXELS} \
    custom_reward_function.path=recipe/xr_chest/reward_xr_report.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.val_before_train=True \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=10 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
