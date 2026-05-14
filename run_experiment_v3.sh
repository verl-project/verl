#!/bin/bash
# Experiment launcher v4 - EXACT copy of working parameters
# ONLY changes: paths (model/data/reward). All other params identical to working command.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/experiments/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# PATHS - MODIFY THESE FOR YOUR ENVIRONMENT
# ============================================================
export WORKING_DIR="${WORKING_DIR:-/home/lingquh1xx/L2598/Temp/Spec-RL}"
export MODEL_PATH="${MODEL_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/model}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-base}"
export DATA_PATH="${DATA_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/data}"
export DATASET_NAME="${DATASET_NAME:-deepmath}"
export TRAIN_FILE="${TRAIN_FILE:-train_sample_6144}"
export NUM_GPU="${NUM_GPU:-8}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WORKING_DIR}/checkpoints}"

# ============================================================
# H100 NCCL FIXES
# ============================================================
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# ============================================================
# GPU CLEANUP
# ============================================================
cleanup_gpu() {
    echo "Cleaning up..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "python" 2>/dev/null || true
    sleep 5
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

verify_paths() {
    for p in "${MODEL_PATH}/${MODEL_NAME}" "${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet" "${WORKING_DIR}/custom_reward/verl_math_verify.py"; do
        [ -e "$p" ] || { echo "ERROR: $p not found"; exit 1; }
        echo "  [OK] $p"
    done
}

# ============================================================
# RUN EXPERIMENT - EXACT working parameters, only paths changed
# ============================================================
run_experiment() {
    local phase="$1"
    local spec_decoding="${2:-True}"
    local bias="${3:-0.5}"
    local steps="${4:-1}"      # Default 1 step for quick test
    local ngpu="${5:-${NUM_GPU}}"
    local batch="${6:-128}"    # Default 128 (working value)
    local resp_len="${7:-512}" # Default 512 (working value)
    local rollout_n_val="${8:-4}" # Default 4 (working value)
    
    local log_dir="${EXPERIMENT_DIR}/${phase}_${TIMESTAMP}"
    mkdir -p "${log_dir}"
    mkdir -p "${CHECKPOINT_PATH}/${phase}_${TIMESTAMP}"
    
    cleanup_gpu
    cd "${WORKING_DIR}"
    
    # ========================================================
    # EXACT copy of your working command, only paths changed
    # ========================================================
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet" \
        data.val_files="${DATA_PATH}/${DATASET_NAME}/test.parquet" \
        data.train_batch_size="${batch}" \
        data.max_prompt_length=1024 \
        data.max_response_length="${resp_len}" \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        actor_rollout_ref.model.path="${MODEL_PATH}/${MODEL_NAME}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n="${rollout_n_val}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        custom_reward_function.path="${WORKING_DIR}/custom_reward/verl_math_verify.py" \
        trainer.critic_warmup=0 \
        trainer.logger="['console']" \
        trainer.project_name=verl-test \
        trainer.experiment_name="${phase}_${TIMESTAMP}" \
        +trainer.spec_decoding="${spec_decoding}" \
        +trainer.spec_bias="${bias}" \
        trainer.n_gpus_per_node="${ngpu}" \
        trainer.nnodes=1 \
        trainer.save_freq=1 \
        trainer.test_freq=1 \
        trainer.default_local_dir="${CHECKPOINT_PATH}/${phase}_${TIMESTAMP}" \
        trainer.total_training_steps="${steps}" \
        trainer.total_epochs=1 \
        2>&1 | tee "${log_dir}/training.log"
    
    echo "${phase} complete. Results: ${log_dir}/"
}

# ============================================================
# MAIN
# ============================================================
PHASE="${1:-short_test}"
verify_paths
mkdir -p "${EXPERIMENT_DIR}"

case "${PHASE}" in
    short_test)
        # EXACT working parameters: 1 step, 8 GPU, batch=128, resp=512
        run_experiment "short_test" True 0.5 1 8 128 512 4
        ;;
    phase1_small)
        # Small scale: 60 steps, 8 GPU, batch=128, resp=512
        run_experiment "phase1_small" True 0.5 60 8 128 512 4
        ;;
    phase1_full)
        # Full scale: 60 steps, 8 GPU, batch=512, resp=4096
        run_experiment "phase1_full" True 0.5 60 8 512 4096 8
        ;;
    *)
        echo "Usage: bash run_experiment_v4.sh [short_test|phase1_small|phase1_full]"
        exit 1
        ;;
esac