#!/usr/bin/env bash
# Usage in a single-node Actions job: bash .github/scripts/run_with_hccl_diagnostics.sh COMMAND [ARG ...]
set -eu

# Start the workload (and its local Ray workers) with a fresh native log directory.
capture=$(mktemp -d "${RUNNER_TEMP:?}/hccl-${GITHUB_ACTION:-launch}.XXXXXXXX")
export ASCEND_PROCESS_LOG_PATH="$capture/plog"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

set +e
"$@" 2>&1 | tee "$capture/console.log"
status=${PIPESTATUS[0]}
set -e
# Collection errors must preserve the workload's exit code.
trap 'exit "$status"' EXIT

signature='ERR02200|HCCL function error|DIST call hccl api failed|\b(EI|EJ)[0-9]{4}\b|\[ERROR\][[:space:]]+HCCL\('
if (( status != 0 )) && grep -Eirq "$signature" "$capture"; then
    # The upload step consumes only retained failures from this job attempt.
    artifacts="$RUNNER_TEMP/hccl-diagnostics-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$GITHUB_JOB"
    mkdir -p "$artifacts"
    if [[ -z $(find "$ASCEND_PROCESS_LOG_PATH" -type f -print -quit) ]]; then
        echo "HCCL failure detected, but the worker log directory was empty." | tee -a "$GITHUB_STEP_SUMMARY"
    fi
    mv "$capture" "$artifacts/"
    echo "HCCL_DIAGNOSTIC_DIR=${artifacts##*/}" >> "$GITHUB_ENV"
else
    rm -rf -- "$capture"
fi
