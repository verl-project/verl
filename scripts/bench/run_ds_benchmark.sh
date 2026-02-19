#!/usr/bin/env bash
set -euo pipefail

# Simple driver for DeepSpeed PPO/GRPO benchmarks.
# Example:
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#   TRAIN=/data/gsm8k/train.parquet \
#   VAL=/data/gsm8k/test.parquet \
#   ./scripts/bench/run_ds_benchmark.sh

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN=${TRAIN:-/path/to/gsm8k/train.parquet}
VAL=${VAL:-/path/to/gsm8k/test.parquet}

# One of [ppo, grpo]
RUN=${RUN:-ppo}

# ZeRO and offload sweeps
ZERO_LIST=${ZERO_LIST:-"1 2 3"}
OFFLOAD_LIST=${OFFLOAD_LIST:-"none cpu"}

CONFIG_PATH=${CONFIG_PATH:-recipe}
CONFIG_NAME=${CONFIG_NAME:-ppo_deepspeed}
if [[ "${RUN}" == "grpo" ]]; then
  CONFIG_NAME=grpo_deepspeed
fi

for zero in ${ZERO_LIST}; do
  for offload in ${OFFLOAD_LIST}; do
    RUN_NAME="${RUN}-zs${zero}-offload-${offload}"
    echo ">>> Running ${RUN_NAME}"
    python3 -m verl.trainer.main_ppo \
      --config-path="${CONFIG_PATH}" \
      --config-name="${CONFIG_NAME}" \
      trainer.project_name=deepspeed-bench \
      trainer.experiment_name="${RUN_NAME}" \
      actor_rollout_ref.model.path="${MODEL}" \
      data.train_files="${TRAIN}" \
      data.val_files="${VAL}" \
      data.train_batch_size=256 \
      actor_rollout_ref.actor.deepspeed.zero_stage="${zero}" \
      actor_rollout_ref.actor.deepspeed.offload="${offload}" \
      actor_rollout_ref.ref.deepspeed.zero_stage="${zero}" \
      actor_rollout_ref.ref.deepspeed.offload="${offload}" \
      critic.deepspeed_config.zero_stage="${zero}" \
      critic.deepspeed_config.offload="${offload}"
  done
done
