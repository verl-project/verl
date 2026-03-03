#!/bin/bash
# Minimal RL training: Qwen2.5-0.5B + AgentBay code_interpreter on 1x V100 GPU
#
# Purpose: Verify the end-to-end pipeline (model → tool call → AgentBay → reward)
#
# V100 (sm70) adaptations vs original T4 script:
#   - vllm instead of sglang (sgl-kernel lacks sm70 support)
#   - float16 instead of bfloat16 (V100 has no bf16 hardware)
#   - sdpa attention instead of flash_attention_2
#   - enforce_eager=True (skip CUDA graph / FlashInfer)
#   - reduced max_model_len / max_response_length / micro_batch for 16GB VRAM
#
# Prerequisites:
#   1. pip install wuying-agentbay-sdk vllm
#   2. export AGENTBAY_API_KEY=your_key
#   3. python examples/data_preprocess/gsm8k_agentbay.py --local_save_dir ~/data/gsm8k_agentbay
#   4. Download model: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
#
# Usage:
#   bash examples/sglang_multiturn/run_qwen0.5b_gsm8k_agentbay_1gpu.sh

set -x

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate verl

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-/root/data/gsm8k_agentbay}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='gsm8k_agentbay_verify' \
    trainer.experiment_name='qwen0.5b-agentbay-1gpu-verify' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=50 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/agentbay_tool_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    trainer.total_epochs=1 $@
