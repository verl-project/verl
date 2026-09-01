#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/manifest.sh"
rn4pt_validate_static
mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"

readonly PHASE=${1:-}
case "$PHASE" in
  probe|build|preflight|smoke|recover-smoke|short) ;;
  *) rn4pt_die "usage: $0 probe|build|preflight|smoke|recover-smoke|short" ;;
esac

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q '^verl-rn4pt-'; then
  rn4pt_die "an existing real-NVFP4 bundle job is already running or pending"
fi

case "$PHASE" in
  probe)
    job_file=$RN4PT_BUNDLE/probe_runtime.job
    nodes=1
    time_limit=00:15:00
    ;;
  build)
    [[ -s "$RN4PT_PROBE_PASS" ]] || rn4pt_die "probe.pass is required"
    job_file=$RN4PT_BUNDLE/build_runtime.job
    nodes=1
    time_limit=03:00:00
    ;;
  preflight)
    [[ -s "$RN4PT_BUILD_PASS" ]] || rn4pt_die "build.pass is required"
    job_file=$RN4PT_BUNDLE/preflight.job
    nodes=1
    time_limit=00:45:00
    ;;
  smoke)
    [[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "preflight.pass is required"
    job_file=$RN4PT_BUNDLE/train.job
    nodes=1
    time_limit=01:30:00
    ;;
  recover-smoke)
    [[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "preflight.pass is required"
    [[ ! -e "$RN4PT_SMOKE_PASS" ]] || rn4pt_die "smoke.pass already exists"
    [[ "${RECOVER_JOB_ID:-}" =~ ^[0-9]+$ ]] || rn4pt_die "set RECOVER_JOB_ID to the completed smoke job"
    job_file=$RN4PT_BUNDLE/recover_smoke.job
    nodes=1
    time_limit=00:15:00
    ;;
  short)
    [[ -s "$RN4PT_SMOKE_PASS" ]] || rn4pt_die "smoke.pass is required"
    job_file=$RN4PT_BUNDLE/train.job
    nodes=8
    time_limit=05:00:00
    ;;
esac

sbatch_args=(
  --parsable
  --nodes="$nodes"
  --time="$time_limit"
  --output="$RN4PT_LOGS/%x_%j.out"
)
if [[ "$PHASE" = smoke || "$PHASE" = short ]]; then
  sbatch_args+=(--export="ALL,ARM=$PHASE")
elif [[ "$PHASE" = recover-smoke ]]; then
  sbatch_args+=(--export="ALL,RECOVER_JOB_ID=$RECOVER_JOB_ID")
fi
job_id=$(sbatch "${sbatch_args[@]}" "$job_file")
printf '%s\n' "$job_id" >"$RN4PT_STATE/$PHASE.jobid"
echo "REAL_NVFP4_PERTOKEN_SUBMITTED phase=$PHASE job_id=$job_id"
