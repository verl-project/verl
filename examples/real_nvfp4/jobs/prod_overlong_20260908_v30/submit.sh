#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/manifest.sh"

# The image phases are arm-independent: both arms share one runtime.
readonly USAGE="usage: $0 probe|build|preflight | $0 w4a4|bf16 audit|smoke|smoke8|release"
if [[ "${1:-}" =~ ^(probe|build|preflight)$ ]]; then
  readonly IMAGE_PHASE=$1
  rn4pt_validate_static
  mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"
  if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q '^verl-rn4pt-v30'; then
    rn4pt_die "a v30 job is already running or pending"
  fi
  case "$IMAGE_PHASE" in
    probe)
      job_file=$RN4PT_BUILD_JOB_IMPL/probe_runtime.job; nodes=1; time_limit=00:15:00 ;;
    build)
      [[ -s "$RN4PT_PROBE_PASS" ]] || rn4pt_die "probe.pass is required"
      job_file=$RN4PT_BUILD_JOB_IMPL/build_runtime.job; nodes=1; time_limit=01:00:00 ;;
    preflight)
      [[ -s "$RN4PT_BUILD_PASS" ]] || rn4pt_die "build.pass is required"
      job_file=$RN4PT_BUILD_JOB_IMPL/preflight.job; nodes=1; time_limit=00:45:00 ;;
  esac
  job_id=$(sbatch --parsable --nodes="$nodes" --time="$time_limit" \
    --job-name="verl-rn4pt-v30-$IMAGE_PHASE" \
    --output="$RN4PT_LOGS/%x_%j.out" \
    --export="ALL,RN4PT_MANIFEST_OVERRIDE=$RN4PT_BUNDLE/manifest.sh" "$job_file")
  printf '%s\n' "$job_id" >"$RN4PT_STATE/$IMAGE_PHASE.jobid"
  echo "REAL_NVFP4_V30_SUBMITTED phase=$IMAGE_PHASE job_id=$job_id"
  exit 0
fi

readonly ARM=${1:-}
readonly ACTION=${2:-audit}
case "$ARM" in w4a4|bf16) ;; *) rn4pt_die "$USAGE" ;; esac
case "$ACTION" in audit|smoke|smoke8|release) ;; *) rn4pt_die "$USAGE" ;; esac

rn4pt_arm_settings "$ARM"
rn4pt_validate_static
[[ -s "$RN4PT_PREFLIGHT_PASS" ]] || rn4pt_die "preflight.pass is required"
rn4pt_require_runtime_image

echo "REAL_NVFP4_V30_AUDIT_PASS arm=$ARM exp=$LONG_EXP precision=$LONG_PRECISION_MODE" \
  "bf16_layers=${LONG_BF16_LAYERS_AT_START}/${LONG_BF16_LAYERS_AT_END}" \
  "first_last=$LONG_FIRST_LAST_BF16 ds=$LONG_FILTER_GROUPS gen_mult=$LONG_GEN_PROMPT_BSZ_MULT" \
  "tis=$LONG_ROLLOUT_IS targets=${LONG_TARGET_STEPS[*]}"
if [[ "$ACTION" = audit ]]; then
  exit 0
fi

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q "^verl-rn4pt-v30-$ARM"; then
  rn4pt_die "a v30 $ARM job is already running or pending"
fi
mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"

if [[ "$ACTION" = smoke ]]; then
  job_id=$(sbatch --parsable --nodes=1 --time=01:30:00 \
    --job-name="verl-rn4pt-v30-$ARM-smoke" \
    --output="$RN4PT_LOGS/%x_%j.out" \
    --export="ALL,ARM=$ARM,PHASE=smoke" \
    "$RN4PT_JOB_IMPL/train.job")
  printf '%s\n' "$job_id" >"$RN4PT_STATE/${ARM}_smoke.jobid"
  echo "REAL_NVFP4_V30_SUBMITTED arm=$ARM phase=smoke job_id=$job_id"
  exit 0
fi

# Production-shape validation: same 8-node formal profile as a chain chunk, six
# steps, so several live refits are exercised before any chain is released.
if [[ "$ACTION" = smoke8 ]]; then
  [[ -s "$RN4PT_STATE/${ARM}_chunk_0.pass" ]] || \
    rn4pt_die "run '$0 $ARM smoke' first: $RN4PT_STATE/${ARM}_chunk_0.pass is missing"
  job_id=$(sbatch --parsable --nodes=8 --time=01:30:00 \
    --job-name="verl-rn4pt-v30-$ARM-smoke8" \
    --output="$RN4PT_LOGS/%x_%j.out" \
    --export="ALL,ARM=$ARM,PHASE=smoke8" \
    "$RN4PT_JOB_IMPL/train.job")
  printf '%s\n' "$job_id" >"$RN4PT_STATE/${ARM}_smoke8.jobid"
  echo "REAL_NVFP4_V30_SUBMITTED arm=$ARM phase=smoke8 job_id=$job_id"
  exit 0
fi

# The smoke is the only thing that has ever caught a silently desynced rollout
# early, so a long chain may not start without one.
readonly SMOKE_PASS=$RN4PT_STATE/${ARM}_chunk_0.pass
[[ -s "$SMOKE_PASS" ]] || rn4pt_die "run '$0 $ARM smoke' first: $SMOKE_PASS is missing"
# The 1-node smoke cleared v29c, whose W4A4 rollout then went NaN at step 2 on
# 8 nodes. A chain may not start until the same shapes have been exercised.
readonly SMOKE8_PASS=$RN4PT_STATE/${ARM}_smoke8.pass
[[ -s "$SMOKE8_PASS" ]] || rn4pt_die "run '$0 $ARM smoke8' first: $SMOKE8_PASS is missing"

readonly CHAIN_STATE=$RN4PT_STATE/${ARM}_chain.tsv
[[ ! -e "$CHAIN_STATE" ]] || rn4pt_die "chain state collision: $CHAIN_STATE"
[[ ! -e "$RN4PT_CHECKPOINTS/$LONG_EXP" ]] || \
  rn4pt_die "fresh checkpoint collision: $RN4PT_CHECKPOINTS/$LONG_EXP"

: >"$CHAIN_STATE"
previous_job=
for index in "${!LONG_TARGET_STEPS[@]}"; do
  chunk=$((index + 1))
  target=${LONG_TARGET_STEPS[$index]}
  sbatch_args=(
    --parsable
    --nodes=8
    --time=05:00:00
    --job-name="verl-rn4pt-v30-$ARM-c${chunk}"
    --output="$RN4PT_LOGS/%x_%j.out"
    --export="ALL,ARM=$ARM,PHASE=long,LONG_CHUNK_INDEX=$chunk,LONG_TARGET_STEP=$target"
  )
  if [[ -n "$previous_job" ]]; then
    sbatch_args+=(--dependency="afterok:$previous_job")
  fi
  job_id=$(sbatch "${sbatch_args[@]}" "$RN4PT_JOB_IMPL/train.job")
  printf '%s\t%s\t%s\t%s\n' "$chunk" "$target" "$job_id" "${previous_job:-none}" >>"$CHAIN_STATE"
  echo "REAL_NVFP4_V30_SUBMITTED arm=$ARM phase=long chunk=$chunk target_step=$target job_id=$job_id dependency=${previous_job:-none}"
  previous_job=$job_id
done
echo "REAL_NVFP4_V30_CHAIN_RELEASED arm=$ARM exp=$LONG_EXP final_job=$previous_job state=$CHAIN_STATE"
