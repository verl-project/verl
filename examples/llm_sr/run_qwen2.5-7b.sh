#!/bin/bash
# =============================================================================
# LLM-SR GRPO Training with Qwen2.5-7B-Instruct
# Single-turn equation discovery on LLM-SR benchmark
#
# Requirements:
#   - 1x A100/H100 80GB GPU (or adjust n_gpus_per_node)
#   - pip install scipy
#   - Data prepared via: python examples/llm_sr/data_preprocess/prepare_llm_sr.py
#
# Usage:
#   bash examples/llm_sr/run_qwen2.5-7b.sh
# =============================================================================

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/llm_sr/config"
DATA_DIR="${DATA_DIR:-$HOME/data/llm_sr}"

# ── Tunable parameters ──
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-8}              # G=8 parallel rollouts per prompt
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
N_GPUS=${N_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
OFFLOAD=${OFFLOAD:-False}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='llm_sr_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/llm_sr/config/interaction_config/equation_fitting.yaml" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm_sr_grpo' \
    trainer.experiment_name='qwen2.5-7b_equation_discovery' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS $@
