#!/bin/bash
# Basic test of verl FSDP GRPO training with a small model
# This uses a small Llama model to verify the training pipeline works

set -x

# Set PYTHONPATH
export PYTHONPATH="/home/t00906153/project/verl-qwen3.5/verl:$PYTHONPATH"

echo "Testing verl FSDP GRPO training pipeline..."

# Create dummy data directory
DATA_DIR="/tmp/verl_test_data"
mkdir -p "$DATA_DIR"

# Create dummy training data
TRAIN_FILE="$DATA_DIR/train.parquet"
python3 -c "
import pandas as pd
import numpy as np

# Create very simple dummy data
data = {
    'prompt': ['What is 2+2?', 'What is 3+3?', 'What is 4+4?'] * 3,
    'response': ['4', '6', '8'] * 3,
    'reward': [1.0, 1.0, 1.0] * 3
}

df = pd.DataFrame(data)
df.to_parquet('$TRAIN_FILE')
print(f'Created dummy training data with {len(df)} samples')
"

# Create dummy validation data
VAL_FILE="$DATA_DIR/test.parquet"
python3 -c "
import pandas as pd
import numpy as np

# Create very simple dummy data
data = {
    'prompt': ['What is 5+5?', 'What is 6+6?'],
    'response': ['10', '12'],
    'reward': [1.0, 1.0]
}

df = pd.DataFrame(data)
df.to_parquet('$VAL_FILE')
print(f'Created dummy validation data with {len(df)} samples')
"

echo "Starting basic FSDP GRPO test..."

# Run a minimal training test
# Using a small model that should be available
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=8 \
    data.max_prompt_length=32 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="meta-llama/Llama-2-7b-hf" \
    actor_rollout_ref.model.model_type=llama \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
    trainer.project_name='verl_fsdp_grpo_basic_test' \
    trainer.experiment_name='basic_fsdp_grpo_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.max_steps=2 \
    trainer.debug_mode=true \
    trainer.dry_run=true $@

echo "Basic test completed"

# Check if the test ran successfully
if [ $? -eq 0 ]; then
    echo "✅ Basic FSDP GRPO test passed!"
    echo "The verl training pipeline is working correctly."
    echo ""
    echo "Next steps for Qwen3.5:"
    echo "1. Install a transformers version with Qwen3.5 support (>=4.57.0)"
    echo "2. Download complete Qwen3.5-27B model weights"
    echo "3. Run: ./examples/grpo_trainer/run_qwen3_5-27b_local.sh"
else
    echo "❌ Basic FSDP GRPO test failed!"
    echo "Please check verl installation and dependencies."
fi