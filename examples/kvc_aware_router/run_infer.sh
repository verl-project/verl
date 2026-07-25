#!/bin/bash
# KV-cache-aware router inference runner — a thin wrapper around parallel_infer.py.
# Defaults (set in parallel_infer.py) are a 2-GPU single-replica smoke test
# (1 sample). Pass extra flags after the positional args to scale up.
#
# Usage:
#   bash examples/kvc_aware_router/run_infer.sh MODEL_PATH [DATA_PATH] [AGENT_CONFIG] [...extra flags...]
#
# Examples:
#   # smoke test
#   bash examples/kvc_aware_router/run_infer.sh /data/models/Qwen3-4B --kv-events
#
#   # full 8-GPU data-parallel
#   bash examples/kvc_aware_router/run_infer.sh /data/models/Qwen3-4B \
#       --num-workers 8 --n-gpus-per-node 8 --tensor-parallel-size 2 --max-samples -1 --kv-events
#
#   # with MooncakeStoreConnector
#   bash examples/kvc_aware_router/run_infer.sh /data/models/Qwen3-4B --enable-mooncake --kv-events

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Positional args: MODEL_PATH (required), DATA_PATH / AGENT_CONFIG (default to same dir).
MODEL_PATH=${1:?ERROR: MODEL_PATH is required}
DATA_PATH=${2:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
AGENT_CONFIG=${3:-${SCRIPT_DIR}/agent_config_localdocker.yaml}
shift 3 2>/dev/null || set --  # drop the first 3; "$@" = remaining flags

# Pre-flight checks
for var_name in DATA_PATH MODEL_PATH AGENT_CONFIG; do
    path="${!var_name}"
    if [ ! -e "$path" ]; then
        echo "ERROR: ${var_name} not found: ${path}" >&2
        exit 1
    fi
done

python "${SCRIPT_DIR}/parallel_infer.py" \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --agent-config-path "${AGENT_CONFIG}" \
    "$@"
