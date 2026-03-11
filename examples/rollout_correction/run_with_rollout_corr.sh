#!/usr/bin/env bash
# Example: RLOO (REINFORCE Leave-One-Out) with Rollout Correction
# This demonstrates self-normalized sequence-level IS with pure policy gradient
#
# References:
#   - Rollout Correction Docs: https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md
#   - Rollout Correction Math: https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr_math.md

set -xeuo pipefail

# ==============================================================================
# Rollout Correction Configuration (RLOO)
# ==============================================================================

# Importance Sampling (IS) weights configuration
rollout_is="sequence"                     # Self-normalized sequence-level IS
rollout_is_threshold=2.0                  # Upper threshold for IS weights
rollout_is_batch_normalize="true"        # Self-normalization (mean=1.0)

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # No rejection sampling for basic RLOO
rollout_rs_threshold="null"               # RS threshold spec (string or float)

# Bypass mode with REINFORCE loss (no PPO clipping)
bypass_mode="true"     # Skip old_log_prob computation
loss_type="reinforce"  # REINFORCE with explicit IS weights (alternative: "ppo_clip")

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
TRAIN_FILE=${TRAIN_FILE:-"data/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/test.parquet"}

max_prompt_length=2048
max_response_length=4096

# ==============================================================================
# Training Configuration
# ==============================================================================

train_batch_size=128
ppo_mini_batch_size=32
ppo_epochs=1
learning_rate=5e-7

# ==============================================================================
# Algorithm Configuration (RLOO)
# ==============================================================================

adv_estimator=rloo                        # RLOO advantage estimator
gamma=1.0

# ==============================================================================
# Launch Training
# ==============================================================================

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.truncation='left' \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gamma} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.loss_type=${loss_type} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_corr_rloo_example" \
    trainer.experiment_name="rloo_seq_is_pure" \
    trainer.total_epochs=10

echo "Training completed!"
echo ""
echo "RLOO Configuration:"
echo "  - Algorithm: RLOO (REINFORCE Leave-One-Out)"
echo "  - Advantage estimator: ${adv_estimator}"
echo "  - IS mode: ${rollout_is} (self-normalized: ${rollout_is_batch_normalize})"
echo "  - IS threshold: ${rollout_is_threshold}"
echo "  - Bypass mode: ${bypass_mode}, loss_type: ${loss_type}"
echo ""
echo "Monitor these key metrics in wandb:"
echo "  - rollout_corr/rollout_is_mean (should be ~1.0 before batch norm)"
echo "  - rollout_corr/rollout_is_batch_norm_factor (normalization factor applied)"
echo "  - rollout_corr/rollout_is_eff_sample_size (should be >0.5)"
