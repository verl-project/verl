#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1090
source "${RN4PT_MANIFEST_OVERRIDE:-$(dirname "$0")/manifest.sh}"
rn4pt_validate_static
rn4pt_require_runtime_image
mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"

readonly PHASE=${1:-}
case "$PHASE" in smoke|control) ;; *) rn4pt_die "usage: $0 smoke|control" ;; esac
[[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "the shared v12 preflight.pass is required"

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q '^verl-rn4pt-bf16ctl'; then
  rn4pt_die "a BF16 control job is already running or pending"
fi

# PRECISION_MODE=bf16 had never actually been executed before job 2702860, which
# hung in Megatron WorkerDict init and burned the full 5h wall clock without
# logging a step. Prove the BF16 path on one node first; only then spend 8.
if [[ "$PHASE" = smoke ]]; then
  job_id=$(sbatch --parsable --nodes=1 --time=01:00:00 \
    --output="$RN4PT_LOGS/%x_%j.out" \
    --export="ALL,RN4PT_MANIFEST_OVERRIDE=$RN4PT_BUNDLE/manifest.sh,ARM=smoke" \
    "$RN4PT_JOB_IMPL/train.job")
  printf '%s\n' "$job_id" >"$RN4PT_STATE/$PHASE.jobid"
  echo "REAL_NVFP4_BF16_CONTROL_SUBMITTED phase=smoke job_id=$job_id exp=$RN4PT_SMOKE_EXP"
  exit 0
fi

# Match the W4A4 arm step for step: the take-off in every reference curve
# happens over steps 41-80, so the control has to run well past it to be
# informative.
readonly CONTROL_STEPS=${CONTROL_STEPS:-100}
job_id=$(sbatch --parsable --nodes=8 --time=05:00:00 \
  --output="$RN4PT_LOGS/%x_%j.out" \
  --export="ALL,RN4PT_MANIFEST_OVERRIDE=$RN4PT_BUNDLE/manifest.sh,ARM=short,TOTAL_STEPS=$CONTROL_STEPS" \
  "$RN4PT_JOB_IMPL/train.job")
printf '%s\n' "$job_id" >"$RN4PT_STATE/$PHASE.jobid"
echo "REAL_NVFP4_BF16_CONTROL_SUBMITTED job_id=$job_id steps=$CONTROL_STEPS exp=$RN4PT_SHORT_EXP"
