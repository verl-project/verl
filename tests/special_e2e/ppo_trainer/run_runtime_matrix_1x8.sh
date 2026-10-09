#!/usr/bin/env bash
#
# One-click 1x8 Runtime correctness and performance suite for a single server.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
matrix_mode="${RUNTIME_MATRIX_MODE:-all}"
result_root="${RUNTIME_MATRIX_RESULT_DIR:-${TMPDIR:-/tmp}/runtime-matrix-1x8-$(date +%Y%m%dT%H%M%S)}"
baseline_repo="${BASELINE_REPO:-}"
baseline_commit="${BASELINE_COMMIT:-b8d99e5db3e340d91a0da341251d05c9ccb9e34e}"
model_path="${MODEL_PATH:-}"
model_id="${MODEL_ID:-Qwen/Qwen3-8B}"
train_files="${TRAIN_FILES:-}"
val_files="${VAL_FILES:-}"
runtime_python="${RUNTIME_MATRIX_PYTHON:-python3}"
dependency_pythonpath="${RUNTIME_MATRIX_PYTHONPATH:-}"
wandb_project="${WANDB_PROJECT:-verl_runtime_ab_qwen3_8b}"
enable_wandb="${RUNTIME_MATRIX_ENABLE_WANDB:-0}"
dry_run=0

usage() {
    cat <<'EOF'
Run the 1x8 Runtime correctness and performance matrix on one server.

The feature checkout is the repository containing this script. A separate
baseline checkout using NeoProto before the Runtime migration, a Python
environment with the Runtime dependencies, and model/dataset paths are required. Exactly eight CUDA devices must
be visible to the process.

Usage:
  run_runtime_matrix_1x8.sh [options]

Options:
  --baseline-repo PATH     Baseline Git checkout.
  --baseline-commit SHA    Expected baseline commit (default: b8d99e5db3e340d91a0da341251d05c9ccb9e34e).
  --model-path PATH        Model path passed to the trainer.
  --model-id ID            Model identifier (default: Qwen/Qwen3-8B).
  --train-files PATH       Training dataset path.
  --val-files PATH         Validation dataset path.
  --result-dir PATH        New local result directory.
  --mode MODE              all, correctness, or performance (default: all).
  --python PATH            Python environment's python3 (default: python3).
  --pythonpath PATH        Optional dependency prefix for PYTHONPATH.
  --enable-wandb           Add wandb to the console and file loggers.
  --wandb-project NAME     Project used with --enable-wandb.
  --dry-run                Print effective rollout settings and cases without running them.
  -h, --help               Show this help.

The same values may be supplied through BASELINE_REPO, BASELINE_COMMIT,
MODEL_PATH, MODEL_ID, TRAIN_FILES, VAL_FILES, RUNTIME_MATRIX_RESULT_DIR,
RUNTIME_MATRIX_MODE, RUNTIME_MATRIX_PYTHON, RUNTIME_MATRIX_PYTHONPATH,
RUNTIME_MATRIX_ENABLE_WANDB, and WANDB_PROJECT.

Correctness sets max_num_seqs=1 and vLLM optimization level 0 for deterministic
comparison; performance keeps the default engine optimizations. Set
ROLLOUT_ENFORCE_EAGER=True only when explicitly testing eager execution on both
baseline and feature paths.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --baseline-repo) baseline_repo=${2:?missing value for --baseline-repo}; shift 2 ;;
        --baseline-commit) baseline_commit=${2:?missing value for --baseline-commit}; shift 2 ;;
        --model-path) model_path=${2:?missing value for --model-path}; shift 2 ;;
        --model-id) model_id=${2:?missing value for --model-id}; shift 2 ;;
        --train-files) train_files=${2:?missing value for --train-files}; shift 2 ;;
        --val-files) val_files=${2:?missing value for --val-files}; shift 2 ;;
        --result-dir) result_root=${2:?missing value for --result-dir}; shift 2 ;;
        --mode) matrix_mode=${2:?missing value for --mode}; shift 2 ;;
        --python) runtime_python=${2:?missing value for --python}; shift 2 ;;
        --pythonpath) dependency_pythonpath=${2:?missing value for --pythonpath}; shift 2 ;;
        --enable-wandb) enable_wandb=1; shift ;;
        --wandb-project) wandb_project=${2:?missing value for --wandb-project}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${matrix_mode}" in
    all|correctness|performance) ;;
    *) echo "--mode must be all, correctness, or performance" >&2; exit 2 ;;
esac

: "${baseline_repo:?--baseline-repo or BASELINE_REPO is required}"
: "${baseline_commit:?--baseline-commit or BASELINE_COMMIT is required}"
: "${model_path:?--model-path or MODEL_PATH is required}"
: "${train_files:?--train-files or TRAIN_FILES is required}"
: "${val_files:?--val-files or VAL_FILES is required}"
case "${enable_wandb}" in
    0|1) ;;
    *) echo "RUNTIME_MATRIX_ENABLE_WANDB must be 0 or 1" >&2; exit 2 ;;
esac

attention_backend="${ATTENTION_BACKEND:-FLASH_ATTN}"
runtime_matrix_python="$(command -v "${runtime_python}")"
runtime_matrix_python_dir="$(dirname "${runtime_matrix_python}")"
torchstore_strategy="${RUNTIME_MATRIX_TORCHSTORE_STRATEGY:-host}"
case "${torchstore_strategy}" in
    host|local_rank) ;;
    *) echo "RUNTIME_MATRIX_TORCHSTORE_STRATEGY must be host or local_rank" >&2; exit 2 ;;
esac
trainer_loggers='["console","file"]'
if [ "${enable_wandb}" = 1 ]; then
    trainer_loggers='["console","wandb","file"]'
fi

runtime_matrix_init() {
    result_root="$1"
    index="${result_root}/matrix.tsv"
    if [ "${dry_run}" = 1 ]; then return; fi
    if [ -e "${result_root}" ]; then
        echo "Refusing to overwrite existing matrix directory: ${result_root}" >&2
        exit 2
    fi
    if [ "$(git -C "${baseline_repo}" rev-parse HEAD)" != "${baseline_commit}" ]; then
        echo "BASELINE_REPO must be checked out at ${baseline_commit}" >&2
        exit 2
    fi
    if [ ! -e "${model_path}" ]; then
        echo "MODEL_PATH does not exist: ${model_path}" >&2
        exit 2
    fi
    if [ ! -e "${train_files}" ]; then
        echo "TRAIN_FILES does not exist: ${train_files}" >&2
        exit 2
    fi
    if [ ! -e "${val_files}" ]; then
        echo "VAL_FILES does not exist: ${val_files}" >&2
        exit 2
    fi
    "${runtime_matrix_python}" -c \
        'import torch; count = torch.cuda.device_count(); assert count == 8, f"expected 8 CUDA devices, found {count}"'
    mkdir -p "${result_root}"
    printf '%s\n' "${torchstore_strategy}" >"${result_root}/torchstore-strategy.txt"
    printf 'mode\trepetition\torder\tvariant\tbackend\tdata_plane\texperiment_time\trun_name\tsource_commit\n' \
        >"${index}"
}

runtime_cleanup_backend() {
    local backend="$1"
    local run_dir="$2"
    if [ "${backend}" = ray ]; then
        "${runtime_matrix_python}" -m ray.scripts.scripts stop --force
    elif [ "${backend}" = monarch ]; then
        local job_state="${run_dir}/job-context/.monarch/job_state.pkl"
        if [ -e "${job_state}" ]; then
            local state_target
            state_target="$(readlink -f "${job_state}")" || return $?
            (cd "${run_dir}/job-context" && "${runtime_matrix_python}" -m monarch.tools.cli kill) || return $?
            if [ -e "${state_target}" ] || [ -e "${job_state}" ]; then
                echo "Monarch job state remains after cleanup: ${state_target}" >&2
                return 1
            fi
        fi
    else
        echo "Unsupported backend: ${backend}" >&2
        return 2
    fi
}

runtime_run_case() {
    local mode="$1" repetition="$2" order="$3" variant="$4" backend="$5" experiment_time="$6"
    local data_plane=neoproto
    local run_name="${variant}-${backend}-${data_plane}-1x8-${mode}-${experiment_time}"
    local run_dir="${result_root}/${run_name}"
    local source_repo="${repo_root}"
    local source_commit
    local runtime_pythonpath
    local total_steps=10
    local save_freq=-1
    local custom_reward=False
    local data_seed=1234
    local rollout_full_determinism=False
    local actor_full_determinism=False
    local ref_full_determinism=False
    local critic_full_determinism=False
    local rollout_scheduling_policy=fcfs
    local rollout_enforce_eager=False
    local ignore_eos=True
    local rollout_data_dir=null
    local skip_reward_check=False
    local ppo_config_name=ppo_trainer
    local run_rc=0

    if [ "${variant}" = baseline ]; then
        source_repo="${baseline_repo}"
    fi
    source_commit="$(git -C "${source_repo}" rev-parse HEAD)"
    runtime_pythonpath="${dependency_pythonpath:+${dependency_pythonpath}:}${source_repo}"
    if [ "${mode}" = correctness ]; then
        total_steps=2
        save_freq=2
        custom_reward=True
        data_seed=42
        # The vLLM server enables VLLM_BATCH_INVARIANT for full_determinism.
        rollout_full_determinism=True
        actor_full_determinism=True
        ref_full_determinism=True
        critic_full_determinism=True
        rollout_scheduling_policy=priority
        ignore_eos=False
        rollout_data_dir="${run_dir}/rollouts"
    fi
    rollout_enforce_eager="${ROLLOUT_ENFORCE_EAGER:-${rollout_enforce_eager}}"
    if [ "${backend}" = monarch ]; then
        skip_reward_check=True
    fi

    backend_args=()
    # Worker-process environment for deterministic rollout. These run in Ray/
    # Monarch worker actors, which do NOT inherit the driver shell's exports;
    # the env must be injected through the backend's own channel. In particular
    # GlobalRequestLoadBalancer routes with hash(request_id), so PYTHONHASHSEED
    # must be identical across worker processes or full-determinism routing (and
    # therefore rollout output) diverges. baseline (no Runtime) uses the legacy
    # ray_init.runtime_env.env_vars channel; feature (ray and monarch) uses the
    # unified Runtime root env_vars channel.
    worker_env_keys=(
        "PYTHONPATH=${runtime_pythonpath}"
        "PYTHONHASHSEED=42"
        "VLLM_DISABLE_COMPILE_CACHE=1"
        "TOKENIZERS_PARALLELISM=true"
        "TORCHSTORE_GLOO_ENABLED=0"
        "TORCHSTORE_RDMA_ENABLED=1"
        "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING=${HYPERACTOR_MESH_ENABLE_LOG_FORWARDING:-false}"
    )
    if [ "${mode}" = performance ]; then
        worker_env_keys+=("VLLM_ATTENTION_BACKEND=${attention_backend}")
    fi
    env_args=()
    # Quote every value so Hydra parses it as a string: ray runtime_env.env_vars
    # requires Dict[str, str], but Hydra would otherwise infer int for values
    # like PYTHONHASHSEED=42 or TORCHSTORE_GLOO_ENABLED=0 and Ray rejects them.
    if [ "${variant}" = feature ]; then
        for kv in "${worker_env_keys[@]}"; do
            env_args+=("+runtime.env_vars.${kv%%=*}='${kv#*=}'")
        done
    else
        for kv in "${worker_env_keys[@]}"; do
            env_args+=("+ray_kwargs.ray_init.runtime_env.env_vars.${kv%%=*}='${kv#*=}'")
        done
    fi
    extra_hydra_args=()
    if [ "${mode}" = correctness ]; then
        # vLLM 0.23.0/0.24.0 fused RMSNorm changes reduction order at 256 tokens,
        # even with VLLM_BATCH_INVARIANT=1: https://github.com/vllm-project/vllm/issues/48271
        # Keep per-replica batches fixed until the pinned vLLM build includes the fix.
        extra_hydra_args+=(
            "actor_rollout_ref.rollout.max_num_seqs=1"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=${attention_backend}"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.optimization_level=0"
        )
    fi
    if [ "${variant}" = feature ] && [ "${backend}" = ray ]; then
        backend_args+=(runtime.backend=ray)
    elif [ "${backend}" = monarch ]; then
        ppo_config_name=ppo_monarch_neoproto
        backend_args+=(
            topology=monarch_neoproto
            runtime.backend=monarch
            runtime.monarch.job_mode=current
            runtime.monarch.worker_ready_timeout_s=600
            runtime.monarch.shutdown_timeout_s=600
            runtime.monarch.object_store.store_name_prefix="${run_name//-/_}"
            runtime.monarch.object_store.timeout_s=300.0
            "runtime.monarch.object_store.strategy=${torchstore_strategy}"
        )
        if [ -n "${TORCHSTORE_LOCAL_CACHE_BYTES:-}" ]; then
            backend_args+=("runtime.monarch.object_store.local_cache_bytes=${TORCHSTORE_LOCAL_CACHE_BYTES}")
        fi
    fi

    if [ "${dry_run}" = 1 ]; then
        printf 'case=%s repetition=%s source=%s\n' "${run_name}" "${repetition}" "${source_commit}"
        printf 'full_determinism=%s enforce_eager=%s scheduling_policy=%s steps=%s\n' \
            "${rollout_full_determinism}" "${rollout_enforce_eager}" "${rollout_scheduling_policy}" "${total_steps}"
        printf 'hydra_args='
        printf '%q ' "${backend_args[@]}" "${env_args[@]}" "${extra_hydra_args[@]}"
        printf '\n'
        return
    fi

    mkdir -p "${run_dir}/wandb"
    if [ "${mode}" = correctness ]; then
        mkdir -p "${rollout_data_dir}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${mode}" "${repetition}" "${order}" "${variant}" "${backend}" "${data_plane}" \
        "${experiment_time}" "${run_name}" "${source_commit}" >>"${index}"

    runtime_cleanup_backend "${backend}" "${run_dir}" || return $?

    (
        cd "${source_repo}"
        export PATH="${runtime_matrix_python_dir}:${PATH}"
        export PYTHONPATH="${runtime_pythonpath}"
        if [ "${backend}" = monarch ]; then
            mkdir -p "${run_dir}/job-context"
            cd "${run_dir}/job-context"
            "${runtime_matrix_python}" -c \
                'from monarch.job import set_current_job; set_current_job("tests.single_controller.monarch.local_job.job")'
        fi
        export PYTHONHASHSEED=42
        export VLLM_DISABLE_COMPILE_CACHE=1
        if [ "${mode}" = performance ]; then
            export VLLM_ATTENTION_BACKEND="${attention_backend}"
        fi
        export TOKENIZERS_PARALLELISM=true
        export WANDB_DIR="${run_dir}/wandb"
        export WANDB_CONSOLE=off
        export WANDB_RUN_GROUP="qwen3-8b-${mode}-1x8-${experiment_time}"
        export VERL_FILE_LOGGER_PATH="${run_dir}/metrics.jsonl"
        export TORCHSTORE_GLOO_ENABLED=0
        export TORCHSTORE_RDMA_ENABLED=1
        unset WANDB_DISABLED PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

        PPO_CONFIG_NAME="${ppo_config_name}" \
        NUM_GPUS=8 \
        MODEL_ID="${model_id}" \
        MODEL_PATH="${model_path}" \
        TRAIN_FILES="${train_files}" \
        VAL_FILES="${val_files}" \
        MAX_PROMPT_LEN=512 \
        MAX_RESPONSE_LEN=512 \
        N_RESP_PER_PROMPT=4 \
        TRAIN_TRAJ_MICRO_BSZ_PER_GPU=2 \
        ADV_ESTIMATOR=gae \
        USE_KL=True \
        TOTAL_TRAIN_STEPS="${total_steps}" \
        CUSTOM_REWARD_FN="${custom_reward}" \
        CUSTOM_REWARD_FN_FILE="${repo_root}/tests/special_e2e/ppo_trainer/neoproto_test_reward.py" \
        SKIP_CUSTOM_REWARD_STDOUT_CHECK="${skip_reward_check}" \
        DATA_SHUFFLE=False \
        DATA_SEED="${data_seed}" \
        ROLLOUT_SEED=42 \
        ROLLOUT_FULL_DETERMINISM="${rollout_full_determinism}" \
        ROLLOUT_SCHEDULING_POLICY="${rollout_scheduling_policy}" \
        ROLLOUT_ENFORCE_EAGER="${rollout_enforce_eager}" \
        ACTOR_FULL_DETERMINISM="${actor_full_determinism}" \
        REF_FULL_DETERMINISM="${ref_full_determinism}" \
        CRITIC_FULL_DETERMINISM="${critic_full_determinism}" \
        LOAD_FORMAT=auto \
        VAL_BEFORE_TRAIN=False \
        TEST_FREQ=-1 \
        SAVE_FREQ="${save_freq}" \
        SAVE_HF_MODEL=False \
        RESUME_MODE=disable \
        KEEP_OUTPUT_FILE=True \
        OUTPUT_FILE="${run_dir}/training.log" \
        ROLLOUT_DATA_DIR="${rollout_data_dir}" \
        VERL_EXP_NAME="${run_name}" \
        bash "${source_repo}/tests/special_e2e/ppo_trainer/run_function_reward.sh" \
            "${backend_args[@]}" \
            "${env_args[@]}" \
            "${extra_hydra_args[@]}" \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=8 \
            trainer.use_v1=False \
            trainer.logger="${trainer_loggers}" \
            trainer.project_name="${wandb_project}" \
            trainer.experiment_name="${run_name}" \
            trainer.default_local_dir="${run_dir}/checkpoint" \
            data.dataloader_num_workers=0 \
            actor_rollout_ref.actor.shuffle=False \
            critic.shuffle=False \
            actor_rollout_ref.rollout.ignore_eos="${ignore_eos}"
    ) || run_rc=$?

    local cleanup_rc=0
    runtime_cleanup_backend "${backend}" "${run_dir}" || cleanup_rc=$?
    if [ "${run_rc}" = 0 ]; then
        run_rc="${cleanup_rc}"
    fi
    return "${run_rc}"
}

runtime_forward_cases=('baseline ray' 'feature ray' 'feature monarch')
runtime_reverse_cases=('feature monarch' 'feature ray' 'baseline ray')
suite_root="${result_root}"
modes=("${matrix_mode}")
if [ "${matrix_mode}" = all ]; then modes=(correctness performance); fi
if [ "${dry_run}" = 0 ] && [ -e "${suite_root}" ]; then
    echo "Refusing to overwrite existing matrix directory: ${suite_root}" >&2
    exit 2
fi
for mode in "${modes[@]}"; do
    mode_root="${suite_root}"
    if [ "${matrix_mode}" = all ]; then mode_root="${suite_root}/${mode}"; fi
    runtime_matrix_init "${mode_root}"
    repetitions=(1)
    if [ "${mode}" = performance ]; then repetitions=(1 2); fi
    for repetition in "${repetitions[@]}"; do
        experiment_time="$(date +%Y%m%dT%H%M%S)"
        cases=("${runtime_forward_cases[@]}")
        if [ "${repetition}" = 2 ]; then cases=("${runtime_reverse_cases[@]}"); fi
        order=0
        for spec in "${cases[@]}"; do
            order=$((order + 1))
            read -r variant backend <<<"${spec}"
            runtime_run_case "${mode}" "${repetition}" "${order}" "${variant}" "${backend}" "${experiment_time}"
        done
    done
    if [ "${dry_run}" = 0 ]; then
        checker_args=(--repetitions 2)
        if [ "${mode}" = correctness ]; then checker_args=(--repo "${repo_root}"); fi
        "${runtime_matrix_python}" "${repo_root}/tests/special_e2e/ppo_trainer/check_runtime_matrix.py" \
            "${mode}" --root "${result_root}" --index "${index}" \
            --output "${result_root}/${mode}-summary.json" "${checker_args[@]}"
    fi
done
if [ "${dry_run}" = 1 ]; then exit 0; fi
echo "RUNTIME_MATRIX_RESULT_DIR=${suite_root}"
echo "RUNTIME_MATRIX_STATUS=PASS"
