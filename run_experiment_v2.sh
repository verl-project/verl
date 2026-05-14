#!/bin/bash
# Experiment launcher - uses ORIGINAL train_grpo-spec-sampling.sh
# Keeps ALL original environment settings, only modifies paths

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/experiments/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# PATH CONFIGURATION - MODIFY THESE FOR YOUR ENVIRONMENT
# ============================================================
export WORKING_DIR="${WORKING_DIR:-/home/lingquh1xx/L2598/Temp/Spec-RL}"
export MODEL_PATH="${MODEL_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/model}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-base}"
export DATA_PATH="${DATA_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/data}"
export DATASET_NAME="${DATASET_NAME:-deepmath}"
export TRAIN_FILE="${TRAIN_FILE:-train_sample_6144}"
export NUM_GPU="${NUM_GPU:-8}"

# ============================================================
# GPU CLEANUP
# ============================================================
cleanup_gpu() {
    echo "Cleaning up GPU processes..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 3
    
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "GPU memory after cleanup: ${used}MB"
}

# ============================================================
# VERIFY PATHS
# ============================================================
verify_paths() {
    echo "Verifying paths..."
    local errors=0
    
    # Model path (MODEL_PATH is parent dir, MODEL_NAME is subdir)
    local model_full="${MODEL_PATH}/${MODEL_NAME}"
    if [ ! -d "${model_full}" ]; then
        echo "ERROR: Model not found: ${model_full}"
        errors=$((errors + 1))
    else
        echo "  [OK] Model: ${model_full}"
    fi
    
    # Data file
    local train_file="${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet"
    if [ ! -f "${train_file}" ]; then
        echo "ERROR: Train data not found: ${train_file}"
        errors=$((errors + 1))
    else
        echo "  [OK] Train data: ${train_file}"
    fi
    
    # Reward function
    if [ ! -f "${WORKING_DIR}/custom_reward/verl_math_verify.py" ]; then
        echo "ERROR: Reward function not found"
        errors=$((errors + 1))
    else
        echo "  [OK] Reward function"
    fi
    
    # Training script
    if [ ! -f "${WORKING_DIR}/training_scripts/train_grpo-spec-sampling.sh" ]; then
        echo "ERROR: train_grpo-spec-sampling.sh not found"
        errors=$((errors + 1))
    else
        echo "  [OK] Training script"
    fi
    
    if [ ${errors} -gt 0 ]; then
        echo "Please fix the paths and try again."
        exit 1
    fi
}

# ============================================================
# RUN EXPERIMENT using ORIGINAL train_grpo-spec-sampling.sh
# ============================================================
run_experiment() {
    local phase="$1"
    local spec_decoding="${2:-True}"
    local bias="${3:-0.5}"
    local steps="${4:-60}"
    
    local log_dir="${EXPERIMENT_DIR}/${phase}_${TIMESTAMP}"
    mkdir -p "${log_dir}"
    
    echo ""
    echo "============================================================"
    echo "  Running: ${phase}"
    echo "  Spec Decoding: ${spec_decoding}"
    echo "  Model: ${MODEL_PATH}/${MODEL_NAME}"
    echo "  Data: ${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet"
    echo "============================================================"
    
    cleanup_gpu
    cd "${WORKING_DIR}"
    
    # ========================================================
    # CRITICAL: These exports are REQUIRED by the original script
    # They were missing in the previous version!
    # ========================================================
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export HYDRA_FULL_ERROR=1
    export RAY_memory_usage_threshold=0.99
    
    # Call ORIGINAL train_grpo-spec-sampling.sh
    # It will construct: actor_rollout_ref.model.path=$MODEL_PATH/$MODEL_NAME
    # And: data.train_files=$DATA_PATH/$DATASET_NAME/$TRAIN_FILE.parquet
    # Use EXACT same parameters as original SPEC-RL
    # Only change: model_path, train_file_name, spec_decoding, bias, steps, suffix
    # ONLY pass parameters that train_grpo-spec-sampling.sh explicitly supports
    # All other parameters use the script's internal defaults
    bash training_scripts/train_grpo-spec-sampling.sh \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --dataset_name "${DATASET_NAME}" \
        --train_file_name "${TRAIN_FILE}" \
        --train_batch_size 512 \
        --max_response_length 4096 \
        --rollout_gpu_memory_util 0.7 \
        --rollout_tp 2 \
        --ppo_micro_batch_size 2 \
        --total_steps "${steps}" \
        --spec_decoding "${spec_decoding}" \
        --bias "${bias}" \
        --num_gpu "${NUM_GPU}" \
        --suffix "${phase}_${TIMESTAMP}" \
        2>&1 | tee "${log_dir}/training.log"
    
    echo ""
    echo "${phase} complete. Results: ${log_dir}/"
}

# ============================================================
# MAIN DISPATCH
# ============================================================
PHASE="${1:-phase1}"

# Verify paths first
verify_paths
mkdir -p "${EXPERIMENT_DIR}"

echo "============================================================"
echo "  Dynamic Spec Experiments (v2 - Original Script Wrapper)"
echo "  Phase: ${PHASE}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================================"

case "${PHASE}" in
    phase1)
        # Phase 1: SPEC-RL baseline (spec_decoding=True)
        run_experiment "phase1" True 0.5 60
        ;;
    
    baseline_no_spec)
        # Baseline WITHOUT speculative decoding
        run_experiment "baseline_no_spec" False 0.0 60
        ;;
    
    phase2)
        # Phase 2: Dynamic spec (requires code modification first)
        export DYNAMIC_SPEC_VERIFY="incremental"
        export DYNAMIC_SPEC_PREDICTOR="per_prompt_p75"
        run_experiment "phase2" True 0.5 60
        ;;
    
    short_test)
        # Quick 1-step test to verify setup
        echo "Running 1-step quick test..."
        run_experiment "short_test" True 0.5 1
        ;;
    
    analyze)
        if [ -f "experiments/analyze_results.py" ]; then
            python experiments/analyze_results.py \
                --results_dir "${EXPERIMENT_DIR}" \
                --output "${EXPERIMENT_DIR}/report_${TIMESTAMP}.md"
        else
            echo "analyze_results.py not found. Skipping analysis."
        fi
        ;;
    
    *)
        echo "Unknown phase: ${PHASE}"
        echo ""
        echo "Usage: bash run_experiment_v2.sh [phase1|baseline_no_spec|phase2|short_test|analyze]"
        echo ""
        echo "Recommended workflow:"
        echo "  1. bash run_experiment_v2.sh short_test      # 1-step sanity check"
        echo "  2. bash run_experiment_v2.sh baseline_no_spec # No spec baseline (60 steps)"
        echo "  3. bash run_experiment_v2.sh phase1           # SPEC-RL baseline (60 steps)"
        echo "  4. bash run_experiment_v2.sh phase2           # Your optimized version"
        echo "  5. bash run_experiment_v2.sh analyze          # Compare results"
        exit 1
        ;;
esac