#!/bin/bash
#SBATCH --job-name=mt_dapo_7b_2tenant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --account=infra01
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=env
#SBATCH --output=logs/multi-tenant/mt_dapo_7b_2tenant_%j.out


REPO_ROOT="/users/${USER}/scratch/rl-as-a-service"

# Tenant definitions: name:train_file:val_file (comma-separated)
export TENANTS="alice:${HOME}/data/alice_train.parquet:${HOME}/data/alice_val.parquet,bob:${HOME}/data/bob_train.parquet:${HOME}/data/bob_val.parquet"

# LoRA config (same for all tenants)
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export EXP_NAME=${EXP_NAME:-'MT-DAPO-Qwen2.5-7b-vllm-fully-async-2-2-2tenants'}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-4}

bash "${REPO_ROOT}/multi_tenant_run.sh" "$@"
