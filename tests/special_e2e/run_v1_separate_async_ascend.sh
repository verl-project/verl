#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}
SKIP_DUMP_DIR=${SKIP_DUMP_DIR:-${HOME}/data/rollout_dump_v1_separate_async}

export NUM_GPUS=${NUM_GPUS:-8}
export N_GPUS_TRAINING=${N_GPUS_TRAINING:-4}
export N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-16}
export PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export VANILLA_MBRIDGE=${VANILLA_MBRIDGE:-True}

params=(
    data.filter_overlong_prompts=False
    algorithm.rollout_correction.bypass_mode=True
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    trainer.val_before_train=True
    trainer.total_epochs=2
    trainer.log_val_generations=10
    skip.rollout_tq.enable=True
    skip.rollout_tq.dump_dir="${SKIP_DUMP_DIR}"
    "skip.rollout_tq.steps=[1]"
    skip.rollout_tq.action=cache
)

if [[ "${ACTOR_STRATEGY}" == "fsdp2" ]]; then
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
    params+=(
        actor_rollout_ref.actor.grad_clip=1.0
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    )
elif [[ "${ACTOR_STRATEGY}" == "megatron" ]]; then
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
    params+=(
        actor_rollout_ref.actor.optim.lr_decay_steps=10000000
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True
    )
else
    echo "Unknown ACTOR_STRATEGY=${ACTOR_STRATEGY}; expected fsdp2 or megatron"
    exit 1
fi

exec bash "${SCRIPT_DIR}/run_v1_separate_async.sh" "${params[@]}" "$@"
