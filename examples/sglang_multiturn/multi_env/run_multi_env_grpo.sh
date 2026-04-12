#!/usr/bin/env bash
# Multi-environment tool-use GRPO training example.
set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"
EXAMPLE_DIR="$PROJECT_DIR/examples/sglang_multiturn/multi_env"

# Step 1: Prepare dataset
python3 "$EXAMPLE_DIR/prepare_multi_env_dataset.py" \
    --output_dir "$EXAMPLE_DIR/data" \
    --db_dir "$EXAMPLE_DIR/sample_dbs"

# Step 2: Train
python3 -m verl.trainer.main_ppo \
    --config-path="$EXAMPLE_DIR/config" \
    --config-name='multi_env_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$EXAMPLE_DIR/data/train.parquet" \
    data.val_files="$EXAMPLE_DIR/data/val.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.multi_turn.tool_env_manifest_path="$EXAMPLE_DIR/config/tool_env_manifest.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    "$@"
