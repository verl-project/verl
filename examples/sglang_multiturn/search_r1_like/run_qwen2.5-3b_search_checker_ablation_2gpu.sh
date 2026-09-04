#!/bin/bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <mode> [hydra overrides...]"
    echo "Modes: search_only | checker_explicit_only | checker_guarded | triage_guarded | triage_relaxed_guarded"
    exit 1
fi
shift

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "$NUM_GPUS" == "1" ]]; then
    DEFAULT_OVERRIDES=(
        trainer.n_gpus_per_node=1
        data.train_batch_size=2
        data.val_batch_size=2
        actor_rollout_ref.actor.ppo_mini_batch_size=2
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.rollout.agent.num_workers=1
        data.max_prompt_length=2304
        actor_rollout_ref.rollout.prompt_length=2304
        data.max_response_length=768
        actor_rollout_ref.rollout.max_model_len=4096
        actor_rollout_ref.rollout.gpu_memory_utilization=0.30
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=96
    )
else
    DEFAULT_OVERRIDES=(
        trainer.n_gpus_per_node=2
        data.train_batch_size=4
        data.val_batch_size=4
        actor_rollout_ref.actor.ppo_mini_batch_size=4
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.rollout.agent.num_workers=1
        data.max_prompt_length=2304
        actor_rollout_ref.rollout.prompt_length=2304
        data.max_response_length=768
        actor_rollout_ref.rollout.max_model_len=4096
        actor_rollout_ref.rollout.gpu_memory_utilization=0.35
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=96
    )
fi

exec bash "$BASE_DIR/run_qwen2.5-7b_search_checker_ablation_2gpu.sh" \
    "$MODE" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    trainer.experiment_name=qwen2.5-3b-${MODE}-ablation-$(date +%d-%H-%M) \
    "${DEFAULT_OVERRIDES[@]}" \
    "$@"
