#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1090
source "${RN4PT_MANIFEST_OVERRIDE:-$(dirname "$0")/manifest.sh}"
rn4pt_validate_static
mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"

readonly PHASE=${1:-}
case "$PHASE" in probe|build|preflight|smoke|control) ;; *) rn4pt_die "usage: $0 probe|build|preflight|smoke|control" ;; esac

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q '^verl-rn4pt-moedbg'; then
  rn4pt_die "a BF16 control job is already running or pending"
fi

exports="ALL,RN4PT_MANIFEST_OVERRIDE=$RN4PT_BUNDLE/manifest.sh"
case "$PHASE" in
  probe)
    job_file=$RN4PT_BUILD_JOB_IMPL/probe_runtime.job; nodes=1; time_limit=00:15:00 ;;
  build)
    [[ -s "$RN4PT_PROBE_PASS" ]] || rn4pt_die "probe.pass is required"
    job_file=$RN4PT_BUILD_JOB_IMPL/build_runtime.job; nodes=1; time_limit=01:00:00 ;;
  preflight)
    [[ -s "$RN4PT_BUILD_PASS" ]] || rn4pt_die "build.pass is required"
    job_file=$RN4PT_BUILD_JOB_IMPL/preflight.job; nodes=1; time_limit=00:45:00 ;;
  smoke)
    [[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "preflight.pass is required"
    rn4pt_require_runtime_image
    job_file=$RN4PT_JOB_IMPL/train.job; nodes=1; time_limit=01:00:00
    exports+=",ARM=smoke" ;;
  control)
    [[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "preflight.pass is required"
    rn4pt_require_runtime_image
    # The take-off in every reference curve happens over steps 41-80, so the
    # control has to run well past it to discriminate anything.
    job_file=$RN4PT_JOB_IMPL/train.job; nodes=8; time_limit=05:00:00
    exports+=",ARM=short,TOTAL_STEPS=${CONTROL_STEPS:-100}" ;;
esac

job_id=$(sbatch --parsable --nodes="$nodes" --time="$time_limit" \
  --output="$RN4PT_LOGS/%x_%j.out" --export="$exports" "$job_file")
printf '%s\n' "$job_id" >"$RN4PT_STATE/$PHASE.jobid"
echo "REAL_NVFP4_BF16_CONTROL_SUBMITTED phase=$PHASE job_id=$job_id"
