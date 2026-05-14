#!/bin/bash
# Main experiment launcher for Dynamic Speculative Decoding
#
# Usage:
#   bash run_experiment.sh [phase]
#
# Phases:
#   phase1   - Baseline: Run SPEC-RL with full logging
#   phase2_1 - Offline: Compare window predictors on historical data
#   phase2_2 - Online: Run with incremental verification
#   phase2_3 - Grid search: Tune segmentation parameters
#   phase2_4 - Final: Best combination
#   phase3   - Adam signal validation
#   analyze  - Analyze all results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/experiments/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration - MODIFY THESE PATHS FOR YOUR ENVIRONMENT
# ALL variables are exported so sub-scripts (train_grpo-spec-sampling.sh) can see them
export WORKING_DIR="${WORKING_DIR:-/home/lingquh1xx/L2598/Temp/Spec-RL}"
export MODEL_PATH="${MODEL_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/model}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-base}"
export DATA_PATH="${DATA_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/data}"
export DATASET_NAME="${DATASET_NAME:-deepmath}"
export TRAIN_FILE="${TRAIN_FILE:-train_sample_6144}"
export NUM_GPU="${NUM_GPU:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export TOTAL_STEPS="${TOTAL_STEPS:-60}"

# IMPORTANT: Set these environment variables if you don't want to edit this file:
#   export WORKING_DIR=/your/working/dir
#   export MODEL_PATH=/your/model/dir
#   export MODEL_NAME=Qwen3-1.7B-base
#   export DATA_PATH=/your/data/dir
#   export DATASET_NAME=deepmath
#   export TRAIN_FILE=train_sample_6144

echo "============================================================"
echo "  Dynamic Speculative Decoding Experiment Suite"
echo "  Phase: ${1:-all}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================================"
echo "  Model Path: ${MODEL_PATH}"
echo "  Model: ${MODEL_NAME}"
echo "  Data Path: ${DATA_PATH}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Train File: ${TRAIN_FILE}"
echo "  GPUs: ${NUM_GPU}"
echo "  Batch Size: ${TRAIN_BATCH_SIZE}"
echo "  Max Response: ${MAX_RESPONSE_LENGTH}"
echo "  Total Steps: ${TOTAL_STEPS}"
echo "  Working Dir: ${WORKING_DIR}"
echo "============================================================"

mkdir -p "${EXPERIMENT_DIR}"

# ============================================================
# Phase 1: Baseline with Full Logging
# ============================================================
run_phase1() {
    echo ""
    echo "============================================================"
    echo "  Phase 1: Baseline SPEC-RL with Full Logging"
    echo "============================================================"

    local run_name="phase1_baseline_${TIMESTAMP}"
    local log_dir="${EXPERIMENT_DIR}/phase1_${TIMESTAMP}"
    mkdir -p "${log_dir}"

    cd "${WORKING_DIR}"

    # ABSOLUTE PATHS - no dependency on WORKING_DIR env var
    local model_full_path="${MODEL_PATH}/${MODEL_NAME}"
    local train_file_path="${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet"
    local val_file_path="${DATA_PATH}/${DATASET_NAME}/test.parquet"
    local reward_fn_path="${WORKING_DIR}/custom_reward/verl_math_verify.py"
    local rollout_dir="${WORKING_DIR}/rollouts/${run_name}"
    local checkpoint_dir="${WORKING_DIR}/checkpoints/${run_name}"

    # Verify all paths exist
    echo "Verifying paths..."
    echo "  Model: ${model_full_path}"
    echo "  Train: ${train_file_path}"
    echo "  Reward: ${reward_fn_path}"
    for p in "${model_full_path}" "${train_file_path}" "${reward_fn_path}"; do
        if [ ! -e "${p}" ]; then
            echo "ERROR: Path does not exist: ${p}"
            exit 1
        fi
    done

    # Kill stale Ray processes (they may hold old env vars)
    echo "Cleaning up stale Ray processes..."
    ray stop --force 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    sleep 3

    # Run SPEC-RL DIRECTLY (bypass train_grpo-spec-sampling.sh)
    # This ensures all paths are absolute and not dependent on env vars
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${train_file_path}" \
        data.val_files="${val_file_path}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length=1024 \
        data.max_response_length="${MAX_RESPONSE_LENGTH}" \
        data.filter_overlong_prompts=True \
        data.shuffle=False \
        data.truncation='error' \
        actor_rollout_ref.model.path="${model_full_path}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        custom_reward_function.path="${reward_fn_path}" \
        trainer.critic_warmup=0 \
        trainer.logger="['console','wandb']" \
        trainer.project_name="verl-0.5-train" \
        trainer.experiment_name="${run_name}" \
        +trainer.spec_decoding=True \
        +trainer.spec_bias=0.5 \
        trainer.rollout_data_dir="${rollout_dir}" \
        trainer.n_gpus_per_node="${NUM_GPU}" \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.total_training_steps="${TOTAL_STEPS}" \
        trainer.total_epochs=15 \
        2>&1 | tee "${log_dir}/training.log"

    echo "Phase 1 complete. Results: ${log_dir}/"
}

# ============================================================
# Common: Run SPEC-RL with given config
# ============================================================
run_spec_rl() {
    local run_name="$1"
    local log_dir="$2"
    local spec_decoding="${3:-True}"
    local spec_bias="${4:-0.5}"
    local total_steps="${5:-${TOTAL_STEPS}}"
    local predictor="${6:-}"

    mkdir -p "${log_dir}"
    cd "${WORKING_DIR}"

    # ABSOLUTE PATHS
    local model_full_path="${MODEL_PATH}/${MODEL_NAME}"
    local train_file_path="${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet"
    local val_file_path="${DATA_PATH}/${DATASET_NAME}/test.parquet"
    local reward_fn_path="${WORKING_DIR}/custom_reward/verl_math_verify.py"
    local rollout_dir="${WORKING_DIR}/rollouts/${run_name}"
    local checkpoint_dir="${WORKING_DIR}/checkpoints/${run_name}"

    # Verify paths
    for p in "${model_full_path}" "${train_file_path}" "${reward_fn_path}"; do
        if [ ! -e "${p}" ]; then
            echo "ERROR: Path does not exist: ${p}"
            exit 1
        fi
    done

    # Kill stale Ray processes
    echo "Cleaning up stale Ray processes..."
    ray stop --force 2>/dev/null || true
    pkill -f "ray::" 2>/dev/null || true
    sleep 3

    # Build dynamic spec args
    local spec_args=""
    if [ -n "${predictor}" ]; then
        spec_args="+trainer.dynamic_spec_verify=incremental +trainer.dynamic_spec_predictor=${predictor}"
    fi

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${train_file_path}" \
        data.val_files="${val_file_path}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length=1024 \
        data.max_response_length="${MAX_RESPONSE_LENGTH}" \
        data.filter_overlong_prompts=True \
        data.shuffle=False \
        data.truncation='error' \
        actor_rollout_ref.model.path="${model_full_path}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        custom_reward_function.path="${reward_fn_path}" \
        trainer.critic_warmup=0 \
        trainer.logger="['console','wandb']" \
        trainer.project_name="verl-0.5-train" \
        trainer.experiment_name="${run_name}" \
        +trainer.spec_decoding="${spec_decoding}" \
        +trainer.spec_bias="${spec_bias}" \
        trainer.rollout_data_dir="${rollout_dir}" \
        trainer.n_gpus_per_node="${NUM_GPU}" \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.total_training_steps="${total_steps}" \
        trainer.total_epochs=15 \
        ${spec_args} \
        2>&1 | tee "${log_dir}/training.log"

    echo "Run complete: ${log_dir}/"
}

# ============================================================
# Phase 2.1: Offline Predictor Comparison
# ============================================================
run_phase2_1() {
    echo ""
    echo "============================================================"
    echo "  Phase 2.1: Offline Window Predictor Comparison"
    echo "============================================================"

    local log_dir="${EXPERIMENT_DIR}/phase1_${TIMESTAMP}"

    # Check if Phase 1 data exists
    if [ ! -f "${log_dir}/steps.parquet" ]; then
        echo "Error: Phase 1 data not found at ${log_dir}/steps.parquet"
        echo "Run Phase 1 first: bash run_experiment.sh phase1"
        exit 1
    fi

    python experiments/run_phase2_1.py \
        --log_file "${log_dir}/steps.parquet" \
        --output_dir "${EXPERIMENT_DIR}/phase2_1_${TIMESTAMP}" \
        --num_samples 10000

    echo "Phase 2.1 complete. Results: ${EXPERIMENT_DIR}/phase2_1_${TIMESTAMP}/"
}

# ============================================================
# Phase 2.2: Online Incremental Verification
# ============================================================
run_phase2_2() {
    echo ""
    echo "============================================================"
    echo "  Phase 2.2: Online Incremental Verification"
    echo "============================================================"

    local run_name="phase2_2_incremental_${TIMESTAMP}"
    local log_dir="${EXPERIMENT_DIR}/phase2_2_${TIMESTAMP}"

    export DYNAMIC_SPEC_VERIFY="incremental"
    export DYNAMIC_SPEC_PREDICTOR="per_prompt_p75"
    export DYNAMIC_SPEC_SAFETY="1.2"

    run_spec_rl "${run_name}" "${log_dir}" True 0.5 "${TOTAL_STEPS}" "per_prompt_p75"

    echo "Phase 2.2 complete. Results: ${log_dir}/"
}

# ============================================================
# Phase 2.3: Grid Search
# ============================================================
run_phase2_3() {
    echo ""
    echo "============================================================"
    echo "  Phase 2.3: Grid Search for Best Parameters"
    echo "============================================================"

    # Safety factors to test
    local safety_factors=(1.0 1.1 1.2 1.3 1.5)
    local predictors=("per_prompt_ema" "per_prompt_p75" "adaptive_sf")

    for predictor in "${predictors[@]}"; do
        for safety in "${safety_factors[@]}"; do
            local run_name="phase2_3_${predictor}_s${safety}_${TIMESTAMP}"
            local log_dir="${EXPERIMENT_DIR}/phase2_3_${TIMESTAMP}/${predictor}_s${safety}"
            mkdir -p "${log_dir}"

            echo "  Running: predictor=${predictor}, safety=${safety}"

            export DYNAMIC_SPEC_VERIFY="incremental"
            export DYNAMIC_SPEC_PREDICTOR="${predictor}"
            export DYNAMIC_SPEC_SAFETY="${safety}"

            run_spec_rl "${run_name}" "${log_dir}" True 0.5 "${TOTAL_STEPS}" "${predictor}"
        done
    done

    echo "Phase 2.3 complete. Results: ${EXPERIMENT_DIR}/phase2_3_${TIMESTAMP}/"
}

# ============================================================
# Phase 2.4: Final Best Configuration
# ============================================================
run_phase2_4() {
    echo ""
    echo "============================================================"
    echo "  Phase 2.4: Final Best Configuration"
    echo "============================================================"

    # Read best config from Phase 2.3 results
    local best_config_file="${EXPERIMENT_DIR}/phase2_3_${TIMESTAMP}/best_config.json"

    if [ ! -f "${best_config_file}" ]; then
        echo "Warning: No best config found. Using defaults."
        local best_predictor="per_prompt_p75"
    else
        local best_predictor=$(cat "${best_config_file}" | python3 -c "import json,sys; print(json.load(sys.stdin)['predictor'])")
    fi

    echo "  Using best config: predictor=${best_predictor}"

    local run_name="phase2_4_final_${TIMESTAMP}"
    local log_dir="${EXPERIMENT_DIR}/phase2_4_${TIMESTAMP}"

    run_spec_rl "${run_name}" "${log_dir}" True 0.5 120 "${best_predictor}"

    echo "Phase 2.4 complete. Results: ${log_dir}/"
}

# ============================================================
# Phase 3: Adam Signal Validation
# ============================================================
run_phase3() {
    echo ""
    echo "============================================================"
    echo "  Phase 3: Adam Signal Validation (Optional)"
    echo "============================================================"

    local run_name="phase3_adam_${TIMESTAMP}"
    local log_dir="${EXPERIMENT_DIR}/phase3_${TIMESTAMP}"

    export DYNAMIC_SPEC_VERIFY="incremental"
    export DYNAMIC_SPEC_PREDICTOR="adaptive_sf"
    export DYNAMIC_SPEC_SAFETY="1.2"
    export DYNAMIC_SPEC_ADAM_ASSIST="true"

    run_spec_rl "${run_name}" "${log_dir}" True 0.5 "${TOTAL_STEPS}" "adaptive_sf"

    echo "Phase 3 complete. Results: ${log_dir}/"
}

# ============================================================
# Analyze: Generate comparison report
# ============================================================
run_analyze() {
    echo ""
    echo "============================================================"
    echo "  Analyzing All Results"
    echo "============================================================"

    python experiments/analyze_results.py \
        --results_dir "${EXPERIMENT_DIR}" \
        --output "${EXPERIMENT_DIR}/final_report_${TIMESTAMP}.md"

    echo "Analysis complete. Report: ${EXPERIMENT_DIR}/final_report_${TIMESTAMP}.md"
}

# ============================================================
# Main dispatch
# ============================================================
PHASE="${1:-all}"

case "${PHASE}" in
    phase1)
        run_phase1
        ;;
    phase2_1)
        run_phase2_1
        ;;
    phase2_2)
        run_phase2_2
        ;;
    phase2_3)
        run_phase2_3
        ;;
    phase2_4)
        run_phase2_4
        ;;
    phase3)
        run_phase3
        ;;
    analyze)
        run_analyze
        ;;
    all)
        echo "Running full experiment pipeline..."
        run_phase1
        run_phase2_1
        run_phase2_2
        run_phase2_3
        run_phase2_4
        run_analyze
        echo ""
        echo "============================================================"
        echo "  All phases complete!"
        echo "  Results: ${EXPERIMENT_DIR}/"
        echo "============================================================"
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: bash run_experiment.sh [phase1|phase2_1|phase2_2|phase2_3|phase2_4|phase3|analyze|all]"
        exit 1
        ;;
esac