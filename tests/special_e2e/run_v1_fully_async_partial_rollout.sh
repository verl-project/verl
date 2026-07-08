#!/usr/bin/env bash
set -xeuo pipefail

# GPU E2E test for the V1 `fully_async` streaming trainer that specifically exercises the
# partial-rollout path: it forces FREQUENT weight syncs (small parameter_sync_step) with LONG
# responses, so generation is in flight when a sync fires. On each sync the trainer:
#   - aborts in-flight requests on the standalone rollout server and continues them with the
#     generated prefix preserved (partial rollout, inherited from separate_async via
#     FullyAsyncLLMServerClient + CheckpointEngineManager.update_weights), and
#   - pauses the streaming feeder for the duration of the sync (so it does not dispatch into a
#     server that is mid-sync), then resumes it.
#
# After the run it asserts, from the captured logs, that the feeder started and that it paused
# for at least one weight sync -- i.e. the streaming + partial-rollout wiring actually fired.
#
# Usage:   NUM_GPUS=2 bash tests/special_e2e/run_v1_fully_async_partial_rollout.sh
# Requires >= 2 GPUs (>=1 training + >=1 standalone rollout).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-$(mktemp -t fully_async_partial.XXXXXX.log)}"

# Aggressive knobs to guarantee syncs land mid-generation:
export PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-2}"   # sync every 2 steps
export STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-6}"  # -> syncs at steps 2, 4, 6
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}" # longer gen -> straddles syncs
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"        # more in-flight work per prompt

echo "Running fully_async partial-rollout GPU test; logging to ${LOG_FILE}"

# pipefail (set above) ensures the python exit code propagates through tee.
bash "${HERE}/run_v1_fully_async.sh" "$@" 2>&1 | tee "${LOG_FILE}"

echo "=== verifying streaming feeder + partial-rollout (pause-on-sync) markers ==="

assert_grep() {
    local pattern="$1" msg="$2"
    if ! grep -q "${pattern}" "${LOG_FILE}"; then
        echo "FAIL: ${msg} (pattern not found: '${pattern}')"
        exit 1
    fi
    echo "OK: ${msg}"
}

assert_grep "Streaming rollout feeder started" "feeder thread started in on_train_begin"
assert_grep "Pausing streaming feeder for weight sync" "feeder paused for a weight sync (partial-rollout window)"
assert_grep "Resumed streaming feeder after weight sync" "feeder resumed after the weight sync"

echo "PASS: fully_async streaming + partial-rollout pause-on-sync exercised successfully"
