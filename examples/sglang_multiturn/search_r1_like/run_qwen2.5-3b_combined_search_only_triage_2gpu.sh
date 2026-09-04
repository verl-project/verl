#!/bin/bash
set -x

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
export CUDA_VISIBLE_DEVICES=0,3

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/train.parquet}"
VAL_DATA="${VAL_DATA:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet}"
TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_tool_config.yaml"

function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="qwen2.5-3b-combined-search-only-triage-$(now)"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ray_kwargs.ray_init.object_store_memory=10000000000 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=8000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.multi_turn.format=search_r1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
    +actor_rollout_ref.rollout.multi_turn.triage.enable=True \
    +actor_rollout_ref.rollout.multi_turn.triage.online_escalation=True \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search=1 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check=0 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn=2 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search=2 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check=1 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn=4 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search=4 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check=2 \
    +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn=6 \
    +actor_rollout_ref.rollout.multi_turn.triage.escalation.contradiction_threshold=0.30 \
    +actor_rollout_ref.rollout.multi_turn.triage.escalation.support_threshold=0.40 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100000 \
    trainer.test_freq=20 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 \
    "$@"
