#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/manifest.sh"

readonly ARM=${1:-}
readonly ACTION=${2:-audit}
case "$ARM" in w4a4|bf16) ;; *) rn4pt_die "usage: $0 w4a4|bf16 audit|smoke|release" ;; esac
case "$ACTION" in audit|smoke|release) ;; *) rn4pt_die "usage: $0 w4a4|bf16 audit|smoke|release" ;; esac

rn4pt_arm_settings "$ARM"
rn4pt_validate_static
rn4pt_require_runtime_image

echo "REAL_NVFP4_V25_AUDIT_PASS arm=$ARM exp=$LONG_EXP precision=$LONG_PRECISION_MODE" \
  "bf16_layers=${LONG_BF16_LAYERS_AT_START}/${LONG_BF16_LAYERS_AT_END}" \
  "first_last=$LONG_FIRST_LAST_BF16 ds=$LONG_FILTER_GROUPS gen_mult=$LONG_GEN_PROMPT_BSZ_MULT" \
  "tis=$LONG_ROLLOUT_IS targets=${LONG_TARGET_STEPS[*]}"
if [[ "$ACTION" = audit ]]; then
  exit 0
fi

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q "^verl-rn4pt-v25-$ARM"; then
  rn4pt_die "a v25 $ARM job is already running or pending"
fi
mkdir -p "$RN4PT_STATE" "$RN4PT_LOGS"

if [[ "$ACTION" = smoke ]]; then
  job_id=$(sbatch --parsable --nodes=1 --time=01:30:00 \
    --job-name="verl-rn4pt-v25-$ARM-smoke" \
    --output="$RN4PT_LOGS/%x_%j.out" \
    --export="ALL,ARM=$ARM,PHASE=smoke" \
    "$RN4PT_JOB_IMPL/train.job")
  printf '%s\n' "$job_id" >"$RN4PT_STATE/${ARM}_smoke.jobid"
  echo "REAL_NVFP4_V25_SUBMITTED arm=$ARM phase=smoke job_id=$job_id"
  exit 0
fi

# The smoke is the only thing that has ever caught a silently desynced rollout
# early, so a long chain may not start without one.
readonly SMOKE_PASS=$RN4PT_STATE/${ARM}_chunk_0.pass
[[ -s "$SMOKE_PASS" ]] || rn4pt_die "run '$0 $ARM smoke' first: $SMOKE_PASS is missing"

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
    --job-name="verl-rn4pt-v25-$ARM-c${chunk}"
    --output="$RN4PT_LOGS/%x_%j.out"
    --export="ALL,ARM=$ARM,PHASE=long,LONG_CHUNK_INDEX=$chunk,LONG_TARGET_STEP=$target"
  )
  if [[ -n "$previous_job" ]]; then
    sbatch_args+=(--dependency="afterok:$previous_job")
  fi
  job_id=$(sbatch "${sbatch_args[@]}" "$RN4PT_JOB_IMPL/train.job")
  printf '%s\t%s\t%s\t%s\n' "$chunk" "$target" "$job_id" "${previous_job:-none}" >>"$CHAIN_STATE"
  echo "REAL_NVFP4_V25_SUBMITTED arm=$ARM phase=long chunk=$chunk target_step=$target job_id=$job_id dependency=${previous_job:-none}"
  previous_job=$job_id
done
echo "REAL_NVFP4_V25_CHAIN_RELEASED arm=$ARM exp=$LONG_EXP final_job=$previous_job state=$CHAIN_STATE"
