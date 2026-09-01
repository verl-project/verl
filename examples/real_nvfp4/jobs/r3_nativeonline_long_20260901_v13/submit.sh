#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/manifest.sh"

readonly ACTION=${1:-audit}
case "$ACTION" in
  audit|release) ;;
  *) long_die "usage: $0 audit|release" ;;
esac

long_validate_static
rn4pt_require_runtime_image

readonly -a TARGET_STEPS=(80 160 240 320)
echo "REAL_NVFP4_LONG_AUDIT_PASS exp=$LONG_EXP targets=${TARGET_STEPS[*]} dependency=afterok"
if [[ "$ACTION" = audit ]]; then
  exit 0
fi

if squeue -h -u "$USER" -t RUNNING,PENDING -o '%j' | grep -q '^verl-rn4pt-long-v13-'; then
  long_die "a v13 long-run job is already running or pending"
fi
[[ ! -e "$LONG_CHAIN_STATE" ]] || long_die "chain state collision: $LONG_CHAIN_STATE"
[[ ! -e "$LONG_CHECKPOINTS/$LONG_EXP" ]] || \
  long_die "fresh checkpoint collision: $LONG_CHECKPOINTS/$LONG_EXP"

mkdir -p "$LONG_STATE" "$LONG_LOGS"
: >"$LONG_CHAIN_STATE"
previous_job=
for index in "${!TARGET_STEPS[@]}"; do
  chunk=$((index + 1))
  target=${TARGET_STEPS[$index]}
  sbatch_args=(
    --parsable
    --nodes=8
    --time=05:00:00
    --job-name="verl-rn4pt-long-v13-c${chunk}"
    --output="$LONG_LOGS/%x_%j.out"
    --export="ALL,LONG_CHUNK_INDEX=$chunk,LONG_TARGET_STEP=$target"
  )
  if [[ -n "$previous_job" ]]; then
    sbatch_args+=(--dependency="afterok:$previous_job")
  fi
  job_id=$(sbatch "${sbatch_args[@]}" "$LONG_BUNDLE/train.job")
  printf '%s\t%s\t%s\t%s\n' "$chunk" "$target" "$job_id" "${previous_job:-none}" >>"$LONG_CHAIN_STATE"
  echo "REAL_NVFP4_LONG_SUBMITTED chunk=$chunk target_step=$target job_id=$job_id dependency=${previous_job:-none}"
  previous_job=$job_id
done

echo "REAL_NVFP4_LONG_CHAIN_RELEASED exp=$LONG_EXP final_job=$previous_job state=$LONG_CHAIN_STATE"
