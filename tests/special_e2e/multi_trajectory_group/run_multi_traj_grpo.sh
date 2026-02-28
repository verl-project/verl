#!/usr/bin/env bash
# Multi-Trajectory Group E2E Test
#
# This script tests the multi-trajectory group pipeline end-to-end:
# - An agent (multi_traj_test_agent) returns AgentLoopGroupOutput with 2 trajectories per rollout
# - _postprocess flattens them into a batch with trajectory_group_id
# - Batch expansion handles the larger-than-expected batch
# - _propagate_shared_reward_within_groups sets shared rewards
# - compute_advantage deduplicates for GRPO (or skips dedup for GAE)
# - Actor trains on all trajectories
#
# Prerequisites:
#   1. pip install -e ".[test,sglang]"  (or vllm)
#   2. Preprocess data:
#        python tests/special_e2e/multi_trajectory_group/preprocess_data.py \
#          --local_save_dir ~/data/gsm8k_multi_traj_test
#   3. Download model:
#        huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen/Qwen2.5-0.5B-Instruct
#
# Usage:
#   # Test with GRPO (outcome-level advantage, uses dedup-broadcast):
#   bash tests/special_e2e/multi_trajectory_group/run_multi_traj_grpo.sh
#
#   # Test with GAE (per-token advantage, no dedup):
#   ADV_ESTIMATOR=gae bash tests/special_e2e/multi_trajectory_group/run_multi_traj_grpo.sh
#
# Runs on 2+ GPUs. Uses a small model (0.5B) for fast testing.

set -xeuo pipefail

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Defaults — override via environment variables
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k_multi_traj_test"}
GPUS=${GPUS:-2}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-sglang}

# Preprocess data if not present
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preprocessing data to $DATA_DIR ..."
    python3 tests/special_e2e/multi_trajectory_group/preprocess_data.py \
        --local_save_dir "$DATA_DIR"
fi

# Determine critic/value-related settings based on estimator
USE_CRITIC=False
CRITIC_WARMUP=0
if [ "$ADV_ESTIMATOR" = "gae" ]; then
    USE_CRITIC=True
    CRITIC_WARMUP=0
fi

# Build command
CMD=(
    python3 -m verl.trainer.main_ppo
    --config-path="$CONFIG_PATH"
    --config-name='gsm8k_multiturn_grpo'

    # Algorithm
    algorithm.adv_estimator=$ADV_ESTIMATOR
    algorithm.use_kl_in_reward=False

    # Data — small batch for testing
    data.train_batch_size=16
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="$DATA_DIR/test.parquet"

    # Model
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # Actor
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    # Rollout
    actor_rollout_ref.rollout.name=$ROLLOUT_ENGINE
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.n=4

    # Agent loop — this is the key: point to our multi-trajectory test agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/tests/special_e2e/multi_trajectory_group/agent.yaml"
    actor_rollout_ref.rollout.agent.default_agent_loop=multi_traj_test_agent

    # Ref policy
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    # Trainer — 1 step, console only
    trainer.critic_warmup=$CRITIC_WARMUP
    trainer.logger=console
    trainer.project_name=multi_traj_group_test
    trainer.experiment_name="multi_traj_${ADV_ESTIMATOR}_test"
    trainer.n_gpus_per_node=$GPUS
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_training_steps=1
)

# Add critic config for GAE
if [ "$ADV_ESTIMATOR" = "gae" ]; then
    CMD+=(
        critic.optim.lr=1e-5
        critic.ppo_micro_batch_size_per_gpu=8
        critic.fsdp_config.param_offload=True
    )
fi

echo "=========================================="
echo "Multi-Trajectory Group E2E Test"
echo "  ADV_ESTIMATOR: $ADV_ESTIMATOR"
echo "  MODEL: $MODEL_PATH"
echo "  GPUS: $GPUS"
echo "  ROLLOUT_ENGINE: $ROLLOUT_ENGINE"
echo "=========================================="

"${CMD[@]}" "$@"

echo ""
echo "=========================================="
echo "  Multi-trajectory group test PASSED"
echo "=========================================="
