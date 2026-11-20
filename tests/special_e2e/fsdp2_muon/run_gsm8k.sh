#!/usr/bin/env bash
# Copyright 2026 Individual Contributor: yue-zeng-yue
# Licensed under the Apache License, Version 2.0.
# Run ONE matched comparison arm or resume in a prepared, dedicated GPU environment.
set -euo pipefail
: "${MODEL_PATH:?Provide the same local model snapshot for both arms}"
: "${TRAIN_FILE:?Provide the same GSM8K training parquet for both arms}"
: "${VAL_FILE:?Provide the same GSM8K validation parquet for both arms}"
: "${RUN_ROOT:?Provide a fresh absolute result directory}"
OPTIMIZER=${OPTIMIZER:-Muon}
RESUME_FROM=${RESUME_FROM:-}
if [[ "$OPTIMIZER" != Muon && "$OPTIMIZER" != AdamW ]]; then
    echo 'OPTIMIZER must be Muon or AdamW' >&2
    exit 1
fi
if [[ "$RUN_ROOT" != /* || -e "$RUN_ROOT" ]]; then
    echo 'RUN_ROOT must be a new absolute directory to exclude stale results' >&2
    exit 1
fi
python3 -c 'import torch; assert torch.cuda.device_count() >= 2, "Two CUDA GPUs are required"'
mkdir -p "$RUN_ROOT"
export VERL_FILE_LOGGER_PATH="$RUN_ROOT/metrics.jsonl"
export RAY_ADDRESS=local
export PYTHONHASHSEED=42
export HYDRA_FULL_ERROR=1
export TOTAL_STEPS=30
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export ROLLOUT_N=4
export ACTOR_LR=1e-6
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export EXPERIMENT_NAME="gsm8k_${OPTIMIZER}"
resume_mode=disable
if [[ -n "$RESUME_FROM" ]]; then
    [[ -d "$RESUME_FROM/actor" && -f "$RESUME_FROM/data.pt" ]]
    resume_mode=resume_path
fi
# SDPA is an explicit common setting for both arms, including resumed runs.
# Fixed numerical LR is an integration comparison, not matched update magnitudes.
set +e
timeout --signal=TERM --kill-after=30s "${RUN_TIMEOUT:-2400}" \
    bash examples/grpo_trainer/run_dense_muon_fsdp2.sh \
    actor_rollout_ref.actor.optim.optimizer="$OPTIMIZER" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    trainer.logger='["console","file"]' \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.val_before_train=False \
    trainer.default_local_dir="$RUN_ROOT/checkpoints" \
    trainer.rollout_data_dir="$RUN_ROOT/rollouts" \
    trainer.validation_data_dir="$RUN_ROOT/validation" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode="$resume_mode" \
    trainer.resume_from_path="${RESUME_FROM:-null}" \
    ray_kwargs.ray_init.num_cpus=16 \
    "$@" 2>&1 | tee "$RUN_ROOT/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_ROOT/exit_code"
exit "$status"
