#!/usr/bin/env bash
# Example: PPO-clip with Rollout Correction using multiple RS criteria
# Demonstrates chaining token-level and sequence-level rejection sampling
# (token_k1 + seq_max_k2) alongside optional IS metrics.
#
# References:
#   - Rollout Correction Docs: https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md
#   - Rollout Correction Math: https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr_math.md

set -xeuo pipefail

# ==============================================================================
# Rollout Correction Configuration (PPO-clip + multi RS)
# ==============================================================================

# Importance Sampling (IS) weights configuration
rollout_is="token"                       # Token-level IS for metrics/analysis
rollout_is_threshold=2.0                 # Upper threshold for IS weights
rollout_is_batch_normalize="false"       # Keep raw truncated weights

# Rejection Sampling (RS) configuration (multi-criteria)
# - token_k1 keeps per-token ratios inside [lower, upper]
# - seq_max_k2 rejects sequences with extreme chi-square spikes
rollout_rs="token_k1,seq_max_k2"
rollout_rs_threshold="0.6_1.6,2.5"

# Bypass PPO mode (reuse rollout_log_prob)
bypass_mode="true"
loss_type="ppo_clip"

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
learning_rate=3e-6

# ==============================================================================
# Algorithm Configuration
# ==============================================================================

adv_estimator=grpo
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
    algorithm.rollout_correction.rollout_rs=\'${rollout_rs}\' \
    algorithm.rollout_correction.rollout_rs_threshold=\'${rollout_rs_threshold}\' \
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
    trainer.project_name="rollout_corr_multi_rs_example" \
    trainer.experiment_name="ppo_clip_multi_rs" \
    trainer.total_epochs=5

echo "Training completed!"
echo ""
echo "Multi-RS Configuration:"
echo "  - rollout_is: ${rollout_is} (threshold=${rollout_is_threshold}, batch_norm=${rollout_is_batch_normalize})"
echo "  - rollout_rs: ${rollout_rs}"
echo "  - rollout_rs_threshold: ${rollout_rs_threshold}"
echo "  - bypass_mode: ${bypass_mode}, loss_type: ${loss_type}"
echo ""
echo "Track these metrics in wandb:"
echo "  - rollout_corr/rollout_rs_token_k1_mean"
echo "  - rollout_corr/rollout_rs_seq_max_k2_mean"
echo "  - rollout_corr/rollout_rs_masked_fraction"
