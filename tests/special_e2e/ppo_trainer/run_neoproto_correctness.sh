#!/usr/bin/env bash
#
# Deterministic, fail-closed DataProto vs NeoProto E2E correctness gate.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
result_parent="${RUNNER_TEMP:-/tmp}"
if [ -n "${NEOPROTO_CORRECTNESS_DIR:-}" ]; then
    result_dir="${NEOPROTO_CORRECTNESS_DIR}"
    if [ -e "${result_dir}" ]; then
        echo "Refusing to overwrite existing correctness directory: ${result_dir}" >&2
        exit 2
    fi
    mkdir -p "${result_dir}"
else
    result_dir="$(mktemp -d "${result_parent}/neoproto-correctness.XXXXXX")"
fi

export PYTHONHASHSEED=42
export VLLM_DISABLE_COMPILE_CACHE=1
export TOKENIZERS_PARALLELISM=true
num_gpus="${NUM_GPUS:-8}"
num_nodes="${NNODES:-1}"
manage_ray="${NEOPROTO_MANAGE_RAY:-True}"
attention_backend="${ATTENTION_BACKEND:-FLASH_ATTN}"

run_case() {
    local name="$1"
    local data_plane="$2"
    local strict_mode="$3"
    local case_dir="${result_dir}/${name}"

    mkdir -p "${case_dir}/rollouts" "${case_dir}/tensors"
    if [ "${manage_ray}" = "True" ]; then
        ray stop --force
    fi

    NUM_GPUS="${num_gpus}" \
    DATA_PLANE="${data_plane}" \
    NEOPROTO_STRICT_MODE="${strict_mode}" \
    ADV_ESTIMATOR=gae \
    USE_KL=True \
    CUSTOM_REWARD_FN=True \
    CUSTOM_REWARD_FN_FILE="${repo_root}/tests/special_e2e/ppo_trainer/neoproto_test_reward.py" \
    TOTAL_TRAIN_STEPS=2 \
    DATA_SHUFFLE=False \
    DATA_SEED=42 \
    ROLLOUT_SEED=42 \
    ROLLOUT_FULL_DETERMINISM=True \
    ROLLOUT_SCHEDULING_POLICY=priority \
    ROLLOUT_ENFORCE_EAGER=True \
    ACTOR_FULL_DETERMINISM=True \
    REF_FULL_DETERMINISM=True \
    CRITIC_FULL_DETERMINISM=True \
    LOAD_FORMAT=auto \
    VAL_BEFORE_TRAIN=False \
    TEST_FREQ=-1 \
    SAVE_FREQ=2 \
    SAVE_HF_MODEL=False \
    RESUME_MODE=disable \
    KEEP_OUTPUT_FILE=True \
    OUTPUT_FILE="${case_dir}/training.log" \
    ROLLOUT_DATA_DIR="${case_dir}/rollouts" \
    CORRECTNESS_DUMP_DIR="${case_dir}/tensors" \
    VERL_EXP_NAME="neoproto-correctness-${name}" \
    bash "${repo_root}/tests/special_e2e/ppo_trainer/run_function_reward.sh" \
        data.dataloader_num_workers=0 \
        actor_rollout_ref.actor.shuffle=False \
        critic.shuffle=False \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${attention_backend}" \
        trainer.nnodes="${num_nodes}" \
        trainer.default_local_dir="${case_dir}/checkpoint"
}

echo "NEOPROTO_CORRECTNESS_RESULT_DIR=${result_dir}"
run_case dataproto classic False
run_case neoproto neoproto True

checker_args=()
if [ "${num_nodes}" -gt 1 ]; then
    checker_args+=(--distributed-ray-address auto)
fi

python3 "${repo_root}/tests/special_e2e/ppo_trainer/check_neoproto_equivalence.py" \
    --baseline-log "${result_dir}/dataproto/training.log" \
    --neo-log "${result_dir}/neoproto/training.log" \
    --baseline-dump-dir "${result_dir}/dataproto/tensors" \
    --neo-dump-dir "${result_dir}/neoproto/tensors" \
    --baseline-rollout-dir "${result_dir}/dataproto/rollouts" \
    --neo-rollout-dir "${result_dir}/neoproto/rollouts" \
    --baseline-checkpoint-dir "${result_dir}/dataproto/checkpoint/global_step_2" \
    --neo-checkpoint-dir "${result_dir}/neoproto/checkpoint/global_step_2" \
    --expected-steps 2 \
    "${checker_args[@]}"

echo "NEOPROTO_CORRECTNESS_STATUS=PASS"
