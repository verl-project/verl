#!/bin/bash
# Qwen3.5-27B GRPO training script for FSDP backend
# This script demonstrates how to train Qwen3.5-27B with GRPO algorithm using FSDP

set -x

# Model path - using local model directory
MODEL_PATH="/home/t00906153/project/verl-qwen3.5/model/Qwen3.5-27B"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Please download the model weights first"
    exit 1
fi

# Check if model files exist (at least config.json)
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: Model config file not found: $MODEL_PATH/config.json"
    exit 1
fi

echo "Using model from: $MODEL_PATH"

# For testing with small dataset
DATA_PATH="/home/t00906153/project/verl-qwen3.5/data"
if [ ! -d "$DATA_PATH" ]; then
    echo "Warning: Data directory not found: $DATA_PATH"
    echo "Creating dummy data directory..."
    mkdir -p "$DATA_PATH"
fi

# Create dummy data if needed
DUMMY_TRAIN_FILE="$DATA_PATH/train.parquet"
DUMMY_VAL_FILE="$DATA_PATH/test.parquet"

if [ ! -f "$DUMMY_TRAIN_FILE" ]; then
    echo "Creating dummy training data..."
    # We'll create a simple script to generate dummy data
    python3 -c "
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'prompt': ['What is 2+2?'] * 100,
    'response': ['4'] * 100,
    'reward': [1.0] * 100
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
    'prompt': ['What is 3+3?'] * 20,
    'response': ['6'] * 20,
    'reward': [1.0] * 20
}

df = pd.DataFrame(data)
df.to_parquet('$DUMMY_VAL_FILE')
print(f'Created dummy validation data with {len(df)} samples')
"
fi

# Run training with FSDP backend
# Note: For 27B model, we need more conservative settings
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DUMMY_TRAIN_FILE" \
    data.val_files="$DUMMY_VAL_FILE" \
    data.train_batch_size=64 \
    data.max_prompt_length=128 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.model_type=qwen3_5 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_qwen3_5_27b_test' \
    trainer.experiment_name='qwen3_5_27b_grpo_fsdp_test' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=3 \
    trainer.max_steps=10 $@

echo "Training script completed"

# Notes for Qwen3.5-27B specific configurations:
# 1. Reduced batch sizes due to larger model size
# 2. Increased tensor model parallel size (4 for 27B)
# 3. Enabled parameter and optimizer offloading
# 4. Lower learning rate for stability
# 5. Reduced number of epochs/steps for testing
# 6. Using dummy data for initial testing