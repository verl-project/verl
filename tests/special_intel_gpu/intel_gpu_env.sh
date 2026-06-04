#!/usr/bin/env bash

# Shared runtime environment setup for Intel GPU special tests.
#
# Optional overrides can be placed in:
#   tests/special_intel_gpu/.env.intel_gpu
# or provided with:
#   XPU_ENV_FILE=/path/to/file
#
# Supported modes:
#   configure_xpu_runtime sft
#   configure_xpu_runtime vllm

configure_xpu_runtime() {
    local mode="${1:-sft}"

    if [[ -z "${NUM_GPUS:-}" ]]; then
        echo "NUM_GPUS must be set before calling configure_xpu_runtime" >&2
        return 1
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local env_file="${XPU_ENV_FILE:-${script_dir}/.env.intel_gpu}"

    if [[ -f "${env_file}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi

    # Temporary oneCCL workarounds for multi-GPU XPU, to be removed after PyTorch 2.13 release
    export CCL_ATL_SHM="${CCL_ATL_SHM:-1}"
    export CCL_BUFFER_CACHE="${CCL_BUFFER_CACHE:-0}"
    export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK="${CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-0}"
    export CCL_TOPO_ALGO="${CCL_TOPO_ALGO:-0}"

    # Restrict visible GPU devices for this process tree.
    local devices
    devices="$(seq 0 $((NUM_GPUS - 1)) | paste -sd',')"
    export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-${devices}}"

    if [[ "${mode}" == "vllm" ]]; then
        # vLLM XPU uses ZE_AFFINITY_MASK as device control variable.
        if [[ "${XPU_UNSET_ONEAPI_DEVICE_SELECTOR:-1}" == "1" ]]; then
            unset ONEAPI_DEVICE_SELECTOR
        fi

        # Ray/XPU workarounds: reduce idle contexts and disable host-memory false OOM checks.
        export RAY_NUM_PRESTART_PYTHON_WORKERS="${RAY_NUM_PRESTART_PYTHON_WORKERS:-0}"
        export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    fi
}
