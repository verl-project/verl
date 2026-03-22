#!/bin/bash
#SBATCH --job-name=dapo_7b_math_vllm_2_2_ra32_st0_ts1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --account=infra01
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=env
#SBATCH --output=logs/lora-vllm/dapo_7b_math_vllm_2_2_ra32_st0_ts1%j.out


REPO_ROOT="/users/${USER}/scratch/rl-as-a-service"

# Small override set for experiment variants.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export EXP_NAME=${EXP_NAME:-'DAPO-Qwen2.5-7b-vllm-fully-async-2-2-ra32-st0-ts1'}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
bash "${REPO_ROOT}/dapo_7b_math_vllm_2_2_lora_run.sh" "$@"
