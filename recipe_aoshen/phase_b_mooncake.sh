#!/usr/bin/env bash
# Phase B with Mooncake-store offload: fork of recipe_aoshen/phase_b.sh.
#
# Same Qwen3 GSM8K 5-step sanity loop, but the rollout vLLM engine is
# launched with --kv-transfer-config attaching a MooncakeStoreConnector
# pointing at a per-run Mooncake master. On every weight update verl
# drives a hard reset (engine.reset_prefix_cache(reset_connector=True))
# which cascades through scheduler.reset_connector_cache ->
# MooncakeStoreConnector.reset_cache -> store.remove_all(force=True).
#
# Pre-requisites (caller responsibility):
#   1. Mooncake master started via scripts/start_master.sh on this node.
#   2. MOONCAKE_CONFIG_PATH exported pointing at the JSON config used by
#      the master.
#   3. Ray cluster running (single-node or multi-node).
#   4. GSM8K parquet present at /root/data/gsm8k/{train,test}.parquet.
#
# Acceptance:
#   - 5 train steps complete without crash
#   - mooncake_master.log shows >= 5 RemoveAll calls
#   - In at least one cycle, external_prefix_cache_hits > 0 (set via
#     vLLM metrics)
set -xeuo pipefail

cd /workspace/verl

export MACHINE=gb200
export INFER_BACKEND=vllm
export MODEL_PATH=Qwen/Qwen3-0.6B

# Single-tray run for the first Mooncake-store integration smoke.
export NNODES=1
export NGPUS_PER_NODE=4

export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=16
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export PPO_MAX_TOKEN_LEN_PER_GPU=4096

export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.5
export ROLLOUT_N=2

export TOTAL_EPOCHS=1
export SAVE_FREQ=-1
export TEST_FREQ=10

export PROJECT_NAME=phase_b_mooncake
export EXPERIMENT_NAME=qwen3_06b_gsm8k_5steps_mooncake_${NNODES}nodes

# Make sure MOONCAKE_CONFIG_PATH is propagated to all rollout workers.
: "${MOONCAKE_CONFIG_PATH:?Set MOONCAKE_CONFIG_PATH before launching}"
export MOONCAKE_CONFIG_PATH

bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  "data.train_files=['/root/data/gsm8k/train.parquet']" \
  "data.val_files=['/root/data/gsm8k/test.parquet']" \
  trainer.total_training_steps=5 \
  "trainer.logger=['console']" \
  '~ray_kwargs.ray_init.num_gpus' \
  actor_rollout_ref.rollout.kv_store.enable=true \
  "actor_rollout_ref.rollout.kv_store.config_path=${MOONCAKE_CONFIG_PATH}" \
  actor_rollout_ref.rollout.kv_store.kv_role=kv_both
# NOTE: NEVER set NCCL_MNNVL_ENABLE=0 on this rack. Cross-tray traffic must
# go through the NVL72 NVLink switch (~1.8 TB/s/GPU) via MNNVL + IMEX.
