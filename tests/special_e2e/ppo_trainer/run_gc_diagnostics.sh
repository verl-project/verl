#!/usr/bin/env bash
# One-GPU, one-step smoke test for weight-transfer GC diagnostics.
set -euo pipefail

training_steps=1
kernel_release=$(uname -r)
if [[ ${kernel_release,,} == *microsoft-standard-wsl2* ]]; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VERL_FORCE_SHM_WEIGHT_TRANSFER=1
    # Initial weight synchronization exercises the GC point. Stop there because
    # optimizer execution may be unavailable with some WSL CUDA driver stacks.
    training_steps=0
fi

log_file=$(mktemp /tmp/verl-gc-diagnostics-XXXXXX.log)

if ! NUM_GPUS=1 \
    bash tests/special_e2e/ppo_trainer/run_function_reward.sh \
        algorithm.adv_estimator=grpo \
        +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
        data.train_batch_size=1 \
        data.max_response_length=32 \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.gc_diagnostics=True \
        actor_rollout_ref.rollout.checkpoint_engine.gc_on_weight_transfer_cleanup=1 \
        trainer.total_training_steps="${training_steps}" \
        2>&1 | tee "${log_file}"; then
    echo "GC diagnostics smoke failed; log preserved at ${log_file}" >&2
    exit 1
fi

if ! grep -q '\[gc_diagnostics\] point=weight_transfer_cleanup .* generation=1 ' "${log_file}"; then
    echo "Missing GC diagnostics point=weight_transfer_cleanup generation=1; log preserved at ${log_file}" >&2
    exit 1
fi

rm -f "${log_file}"
echo "GC diagnostics smoke passed."
