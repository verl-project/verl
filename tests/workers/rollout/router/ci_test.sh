#!/usr/bin/env bash
# ci_test.sh — full CI test suite for llm_router.
#
# Four test stages, ordered by increasing weight:
#   ut      — pure unit tests (config, strategies, balancer logic). No Ray, no GPU.
#   st-cpu  — system tests on CPU (balancer integration over a real Ray cluster).
#   st-gpu  — integration tests against a real vLLM service (PollingCollector &
#             KVEventCollector). Needs GPU + a local model.
#   e2e     — end-to-end tests via run_infer.sh (KVCAware router + mooncake).
#             Needs GPU + full vLLM + Ray agent loop. Standalone (no conftest sharing).
#
# Each test carries pytest markers (registered in conftest.py) on two dimensions:
#   type:     ``ut`` / ``st`` / ``e2e``
#   resource: ``cpu`` / ``gpu``
# Stages are selected with ``-m``:
#   pytest -m ut                  # the ut stage
#   pytest -m "st and cpu"        # the st-cpu stage
#   pytest -m "st and gpu"        # the st-gpu stage
#   pytest -m e2e                 # the e2e stage
#
# Stages are independent. Residual Ray/vLLM processes are cleaned up between
# stages so a prior stage never leaves GPU/IPC state that breaks the next.
#
# Environment variables (st-gpu / e2e):
#   VLLM_MODEL       — model ID or local path (default: Qwen/Qwen3-4B)
#   VLLM_HOST        — host to bind vLLM server (default: 127.0.0.1)
#   VLLM_PORT        — port for vLLM server (default: 8000)
#   ZMQ_SUB_PORT     — ZMQ PUB socket port (default: 5555)
#   ZMQ_REPLAY_PORT  — ZMQ ROUTER replay port (default: 5556)
#   CUDA_VISIBLE_DEVICES — GPUs visible to st-gpu/e2e (e.g. 0,1,2,3,4,5,6,7)
#
# Usage:
#   ./ci_test.sh                          # run all stages
#   ./ci_test.sh --ut                     # unit tests only
#   ./ci_test.sh --st-cpu                 # CPU system tests only
#   ./ci_test.sh --st-gpu                 # GPU integration tests only
#   ./ci_test.sh --e2e                    # end-to-end tests only
#   ./ci_test.sh --ut --st-cpu            # unit + CPU system tests

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export ZMQ_SUB_PORT="${ZMQ_SUB_PORT:-5555}"
export ZMQ_REPLAY_PORT="${ZMQ_REPLAY_PORT:-5556}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Stage selection ───────────────────────────────────────────────────────
RUN_UT=false
RUN_ST_CPU=false
RUN_ST_GPU=false
RUN_E2E=false
RUN_ALL=true

for arg in "$@"; do
    case "$arg" in
        --ut)     RUN_ALL=false; RUN_UT=true ;;
        --st-cpu) RUN_ALL=false; RUN_ST_CPU=true ;;
        --st-gpu) RUN_ALL=false; RUN_ST_GPU=true ;;
        --e2e)    RUN_ALL=false; RUN_E2E=true ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--ut] [--st-cpu] [--st-gpu] [--e2e]" >&2
            exit 2
            ;;
    esac
done

if $RUN_ALL; then RUN_UT=true; RUN_ST_CPU=true; RUN_ST_GPU=true; RUN_E2E=true; fi

# ── Helpers ───────────────────────────────────────────────────────────────
cleanup_processes() {
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
    sleep 2
}

print_summary() {
    local label="$1" code="$2"
    if [ "$code" -eq 0 ]; then
        echo ">>> $label: PASS"
    else
        echo ">>> $label: FAIL (exit $code)"
    fi
}

OVERALL=0

echo "=== llm_router CI tests ==="
STAGES=""
$RUN_UT     && STAGES="${STAGES}ut "
$RUN_ST_CPU && STAGES="${STAGES}st-cpu "
$RUN_ST_GPU && STAGES="${STAGES}st-gpu "
$RUN_E2E    && STAGES="${STAGES}e2e"
echo "  Stages       : ${STAGES}"
echo "  Model        : ${VLLM_MODEL}"
echo "  vLLM Host    : ${VLLM_HOST}:${VLLM_PORT}"
echo "  ZMQ Sub/Repl : ${ZMQ_SUB_PORT}/${ZMQ_REPLAY_PORT}"
echo ""

# ── ut: pure unit tests (no Ray, no GPU) ─────────────────────────────────
if $RUN_UT; then
    echo "--- ut: unit tests (config / strategies / balancer logic) ---"
    cleanup_processes
    set +e
    python -m pytest "${SCRIPT_DIR}" -m "ut" -v
    code=$?
    set -e
    print_summary "ut" "$code"
    [ "$code" -ne 0 ] && OVERALL=$code
fi

# ── st-cpu: Ray actor integration on CPU ─────────────────────────────────
if $RUN_ST_CPU; then
    echo ""
    echo "--- st-cpu: balancer integration over Ray (CPU) ---"
    cleanup_processes
    set +e
    python -m pytest "${SCRIPT_DIR}" -m "st and cpu" -v
    code=$?
    set -e
    print_summary "st-cpu" "$code"
    [ "$code" -ne 0 ] && OVERALL=$code
fi

# ── st-gpu: real vLLM + GPU integration (collector tests) ────────────────
if $RUN_ST_GPU; then
    echo ""
    echo "--- st-gpu: PollingCollector + KVEventCollector against real vLLM ---"
    cleanup_processes
    set +e
    VLLM_PORT=${VLLM_PORT} ZMQ_SUB_PORT=${ZMQ_SUB_PORT} ZMQ_REPLAY_PORT=${ZMQ_REPLAY_PORT} \
        python -m pytest "${SCRIPT_DIR}" -m "st and gpu" -v
    code=$?
    set -e
    cleanup_processes
    print_summary "st-gpu" "$code"
    [ "$code" -ne 0 ] && OVERALL=$code
fi

# ── e2e: end-to-end via run_infer.sh (KVCAware router + mooncake) ────────
if $RUN_E2E; then
    echo ""
    echo "--- e2e: KVCAware router + mooncake end-to-end via run_infer.sh ---"
    cleanup_processes
    set +e
    python -m pytest "${SCRIPT_DIR}/e2e/" -m "e2e" -v
    code=$?
    set -e
    cleanup_processes
    print_summary "e2e" "$code"
    [ "$code" -ne 0 ] && OVERALL=$code
fi

echo ""
echo "=== All tests done (overall exit ${OVERALL}) ==="
exit $OVERALL
