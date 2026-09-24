#!/usr/bin/env bash
# Example: REINFORCE with Score Centering (arXiv 2609.20807)
# This demonstrates bypass-mode REINFORCE with token-level TIS and score centering,
# which removes the training-inference drift term from the policy gradient.
#
# References:
#   - Rollout Correction Docs: https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md
#   - Rollout Correction Math: https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr_math.md

set -xeuo pipefail

# ==============================================================================
# Rollout Correction Configuration (Score Centering)
# ==============================================================================

# Importance Sampling (IS) weights configuration
rollout_is="token"                        # Token-level TIS, composes with score centering
rollout_is_threshold=2.0                  # Upper threshold for IS weights

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # No rejection sampling
rollout_rs_threshold="null"               # RS threshold spec (string or float)

# Bypass mode with REINFORCE loss (no PPO clipping) + score centering
bypass_mode="true"      # Skip old_log_prob computation
loss_type="reinforce"   # REINFORCE with explicit IS weights (required for score centering)
score_centering="true"  # Subtract the sampler's expected score from every token's score

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"data/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/test.parquet"}

max_prompt_length=1024
max_response_length=512

# ==============================================================================
# Training Configuration
# ==============================================================================

train_batch_size=8
ppo_mini_batch_size=8
ppo_epochs=1
learning_rate=5e-7

# ==============================================================================
# Algorithm Configuration
# ==============================================================================

adv_estimator=grpo
gamma=1.0

# ==============================================================================
# Launch Training
# ==============================================================================
########################### parameter arrays ###########################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_batch_size}
    data.truncation='left'
    algorithm.adv_estimator=${adv_estimator}
    algorithm.gamma=${gamma}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
    algorithm.rollout_correction.bypass_mode=${bypass_mode}
    algorithm.rollout_correction.loss_type=${loss_type}
    algorithm.rollout_correction.score_centering=${score_centering}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${learning_rate}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs}
    actor_rollout_ref.actor.use_fused_kernels=False
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${loss_type}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=${score_centering}
)

ROLLOUT=(
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.topk_log_probs=128
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="rollout_corr_score_centering_example"
    trainer.experiment_name="qwen2_5_0_5b_token_tis_sc"
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.total_training_steps=2
)

EXTRA=(
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"

echo "Training completed!"
echo ""
echo "Score Centering Configuration:"
echo "  - Advantage estimator: ${adv_estimator}"
echo "  - IS mode: ${rollout_is}, threshold: ${rollout_is_threshold}"
echo "  - Bypass mode: ${bypass_mode}, loss_type: ${loss_type}"
echo "  - Score centering: ${score_centering}"
echo ""
echo "Monitor these key metrics in wandb:"
echo "  - actor/sc_correction (diagnostic: a surrogate whose gradient is the centering term)"
echo "  - actor/sc_sampler_head_mass, actor/sc_train_head_mass (top-k head mass coverage)"
echo "  - rollout_corr/rollout_is_mean (should be ~1.0)"
