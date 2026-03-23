#!/bin/bash
set -euo pipefail
set -x

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <mode> [hydra overrides...]"
    echo "Modes: search_only | checker_explicit_only | checker_guarded | triage_guarded | triage_relaxed_guarded"
    exit 1
fi
shift

ulimit -n 65535

module load cuda
unset ROCR_VISIBLE_DEVICES

export XDG_CACHE_HOME=/ocean/projects/med230010p/yji3/.cache
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub
export TMPDIR=/ocean/projects/med230010p/yji3/.tmp
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3}"

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/train.parquet}"
VAL_DATA="${VAL_DATA:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet}"

function now() {
    date '+%d-%H-%M'
}

CLI_ARGS=(
    --config-path="$CONFIG_PATH"
)

COMMON_OVERRIDES=(
    +ray_kwargs.ray_init.object_store_memory=10000000000
    algorithm.adv_estimator=grpo
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
    data.shuffle=False
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.top_p=0.9
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.logger='["console","wandb"]'
    trainer.project_name='search_r1_like_async_rl'
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    trainer.total_epochs=1
)

case "$MODE" in
    search_only)
        EXPERIMENT_NAME="qwen2.5-7b-search-only-ablation-$(now)"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_tool_config.yaml"
        CONFIG_NAME="search_multiturn_grpo"
        MODE_OVERRIDES=(
            data.train_batch_size=16
            data.val_batch_size=8
            data.max_prompt_length=2048
            data.max_response_length=2000
            actor_rollout_ref.actor.ppo_mini_batch_size=16
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.actor.entropy_coeff=0
            actor_rollout_ref.rollout.max_model_len=8000
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4
            actor_rollout_ref.rollout.multi_turn.format=search_r1
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
            trainer.experiment_name="$EXPERIMENT_NAME"
            trainer.save_freq=50
            trainer.test_freq=25
        )
        ;;
    checker_explicit_only)
        EXPERIMENT_NAME="qwen2.5-7b-checker-explicit-only-ablation-$(now)"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        MODE_OVERRIDES=(
            data.train_batch_size=4
            data.val_batch_size=2
            data.max_prompt_length=2560
            data.max_response_length=1200
            actor_rollout_ref.actor.ppo_mini_batch_size=2
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.actor.entropy_coeff=0.001
            actor_rollout_ref.rollout.max_model_len=5000
            actor_rollout_ref.rollout.prompt_length=2560
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.gpu_memory_utilization=0.40
            actor_rollout_ref.rollout.agent.num_workers=2
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=False
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
            trainer.experiment_name="$EXPERIMENT_NAME"
            trainer.save_freq=50
            trainer.test_freq=25
        )
        ;;
    checker_guarded)
        EXPERIMENT_NAME="qwen2.5-7b-checker-guarded-ablation-$(now)"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        MODE_OVERRIDES=(
            data.train_batch_size=4
            data.val_batch_size=2
            data.max_prompt_length=2560
            data.max_response_length=1200
            actor_rollout_ref.actor.ppo_mini_batch_size=2
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.actor.entropy_coeff=0.001
            actor_rollout_ref.rollout.max_model_len=5000
            actor_rollout_ref.rollout.prompt_length=2560
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.gpu_memory_utilization=0.40
            actor_rollout_ref.rollout.agent.num_workers=2
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
            trainer.experiment_name="$EXPERIMENT_NAME"
            trainer.save_freq=50
            trainer.test_freq=25
        )
        ;;
    triage_guarded)
        EXPERIMENT_NAME="qwen2.5-7b-triage-guarded-ablation-$(now)"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        MODE_OVERRIDES=(
            data.train_batch_size=4
            data.val_batch_size=2
            data.max_prompt_length=2560
            data.max_response_length=1200
            actor_rollout_ref.actor.ppo_mini_batch_size=2
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.actor.entropy_coeff=0.001
            actor_rollout_ref.rollout.max_model_len=5000
            actor_rollout_ref.rollout.prompt_length=2560
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.gpu_memory_utilization=0.40
            actor_rollout_ref.rollout.agent.num_workers=2
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.online_escalation=True
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn=5
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn=7
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.contradiction_threshold=0.30
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.support_threshold=0.40
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
            trainer.experiment_name="$EXPERIMENT_NAME"
            trainer.save_freq=50
            trainer.test_freq=25
        )
        ;;
    triage_relaxed_guarded)
        EXPERIMENT_NAME="qwen2.5-7b-triage-relaxed-guarded-ablation-$(now)"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        MODE_OVERRIDES=(
            data.train_batch_size=4
            data.val_batch_size=2
            data.max_prompt_length=2560
            data.max_response_length=1200
            actor_rollout_ref.actor.ppo_mini_batch_size=2
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.actor.entropy_coeff=0.001
            actor_rollout_ref.rollout.max_model_len=5000
            actor_rollout_ref.rollout.prompt_length=2560
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.gpu_memory_utilization=0.40
            actor_rollout_ref.rollout.agent.num_workers=2
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.online_escalation=True
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn=6
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn=7
            +actor_rollout_ref.rollout.multi_turn.triage.heuristic.easy_threshold=0.20
            +actor_rollout_ref.rollout.multi_turn.triage.heuristic.hard_threshold=0.50
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.contradiction_threshold=0.30
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.support_threshold=0.40
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.on_checker_http_error=False
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.reset_counters_on_checker=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
            trainer.experiment_name="$EXPERIMENT_NAME"
            trainer.save_freq=50
            trainer.test_freq=25
        )
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

python3 -m verl.trainer.main_ppo \
    "${CLI_ARGS[@]}" \
    --config-name="$CONFIG_NAME" \
    -- \
    "${COMMON_OVERRIDES[@]}" \
    "${MODE_OVERRIDES[@]}" \
    "$@"
