#!/usr/bin/env bash
# Qwen3.5 dense/MoE MTP delta on the existing VeOmni GRPO example.
# Requires the matching local VeOmni changes (transformers==5.16.1).
# This example has been statically checked only.
set -euo pipefail

: "${model_path:?Set model_path to a local Qwen3.5 checkpoint with MTP weights}"
mtp_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mtp_fused="${USE_FUSED_KERNELS:-True}"

exec bash "$mtp_example_dir/../grpo_trainer/run_qwen3_5_35b_a3b_veomni.sh" \
    actor_rollout_ref.model.use_fused_kernels="$mtp_fused" \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_train=True \
    actor_rollout_ref.model.mtp.enable_rollout=False \
    actor_rollout_ref.model.mtp.detach_encoder=True \
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1 \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=chunk_loss \
    actor_rollout_ref.actor.veomni.router_replay.mode=disabled \
    trainer.experiment_name="qwen3_5_veomni_mtp_fused_${mtp_fused}" \
    "$@"
