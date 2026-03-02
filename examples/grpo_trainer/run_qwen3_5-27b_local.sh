#!/bin/bash
# Qwen3.5-27B GRPO training script for FSDP backend using local transformers
# This script uses local transformers code and skips weight loading

set -x

# Set PYTHONPATH to use local transformers
export PYTHONPATH="/home/t00906153/project/verl-qwen3.5/transformers/src:$PYTHONPATH"
export PYTHONPATH="/home/t00906153/project/verl-qwen3.5/verl:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# Model path - using local model directory
MODEL_PATH="/home/t00906153/project/verl-qwen3.5/model/Qwen3.5-27B"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Check if model files exist (at least config.json)
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: Model config file not found: $MODEL_PATH/config.json"
    exit 1
fi

echo "Using model from: $MODEL_PATH"
echo "Note: Will skip weight loading due to incomplete weight files"

# For testing with small dataset
DATA_PATH="/home/t00906153/project/verl-qwen3.5/data"
if [ ! -d "$DATA_PATH" ]; then
    echo "Creating data directory..."
    mkdir -p "$DATA_PATH"
fi

# Create dummy data if needed
DUMMY_TRAIN_FILE="$DATA_PATH/train.parquet"
DUMMY_VAL_FILE="$DATA_PATH/test.parquet"

if [ ! -f "$DUMMY_TRAIN_FILE" ]; then
    echo "Creating dummy training data..."
    python3 -c "
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'prompt': ['What is 2+2?'] * 10,
    'response': ['4'] * 10,
    'reward': [1.0] * 10
}

df = pd.DataFrame(data)
df.to_parquet('$DUMMY_TRAIN_FILE')
print(f'Created dummy training data with {len(df)} samples')
"
fi

if [ ! -f "$DUMMY_VAL_FILE" ]; then
    echo "Creating dummy validation data..."
    python3 -c "
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'prompt': ['What is 3+3?'] * 5,
    'response': ['6'] * 5,
    'reward': [1.0] * 5
}

df = pd.DataFrame(data)
df.to_parquet('$DUMMY_VAL_FILE')
print(f'Created dummy validation data with {len(df)} samples')
"
fi

# Set environment variable to skip weight loading
export VERL_SKIP_WEIGHT_LOADING="true"
export VERL_DEBUG_MODE="true"

echo "Starting training with local transformers and skipping weight loading..."

# First, apply patches to skip weight loading
echo "Applying patches to skip weight loading..."
python3 -c "
import sys
sys.path.insert(0, '/home/t00906153/project/verl-qwen3.5/verl')
sys.path.insert(0, '/home/t00906153/project/verl-qwen3.5/transformers/src')

# Apply weight loading patches
from verl.models.transformers.skip_weights import apply_all_patches
apply_all_patches()
print('Patches applied successfully')
"

# Run training with FSDP backend
# Note: We're using minimal settings for testing
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DUMMY_TRAIN_FILE" \
    data.val_files="$DUMMY_VAL_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=64 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.model_type=qwen3_5 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_qwen3_5_27b_local_test' \
    trainer.experiment_name='qwen3_5_27b_grpo_fsdp_local' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.max_steps=2 $@

echo "Training script completed"

# Notes:
# 1. Using minimal batch sizes and steps for testing
# 2. tensor_model_parallel_size=1 for single GPU testing
# 3. Skipping weight loading via environment variable
# 4. Using local transformers code