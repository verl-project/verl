#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
PPO_RUN_ROOT=${PPO_RUN_ROOT:-outputs/ppo_zero_six_60_seed7777_lt_fix_20260213_023008}
GRPO_RUN_ROOT=${GRPO_RUN_ROOT:-outputs/grpo_zero_six_60_seed7777_lt_fix_20260213_045247}
OUT_DIR=${OUT_DIR:-outputs/pr_curves_${STAMP}}

CASES=(
  zero1_no_offload
  zero2_no_offload
  zero3_no_offload
  zero1_cpu_offload
  zero2_cpu_offload
  zero3_cpu_offload
)

mkdir -p "$OUT_DIR"

for root in "$PPO_RUN_ROOT" "$GRPO_RUN_ROOT"; do
  if [[ ! -d "$root" ]]; then
    echo "[error] run root not found: $root" >&2
    exit 1
  fi
done

for case_name in "${CASES[@]}"; do
  log_file="$GRPO_RUN_ROOT/$case_name/train.log"
  if [[ ! -f "$log_file" ]]; then
    echo "[error] missing GRPO log: $log_file" >&2
    exit 1
  fi
  step=$(rg -o 'training/global_step:[0-9]+' "$log_file" | tail -n1 | cut -d: -f2)
  step=${step:-0}
  if (( step < 30 )); then
    echo "[error] $case_name only reached step=$step (<30), cannot export GRPO-30 curve" >&2
    exit 1
  fi
done

GRPO30_SUMMARY="$OUT_DIR/grpo_30_summary.tsv"
{
  printf 'case\n'
  for case_name in "${CASES[@]}"; do
    printf '%s\n' "$case_name"
  done
} > "$GRPO30_SUMMARY"

python3 scripts/bench/plot_ds_zero_six_grpo_curves.py \
  --run-root "$GRPO_RUN_ROOT" \
  --summary "$GRPO30_SUMMARY" \
  --max-step 30 \
  --output "$OUT_DIR/grpo_zero_six_30step_curves.png" \
  --metrics-tsv "$OUT_DIR/grpo_zero_six_30step_metrics.tsv"

python3 scripts/bench/plot_ds_zero_six_curves.py \
  --run-root "$PPO_RUN_ROOT" \
  --summary "$PPO_RUN_ROOT/summary.tsv" \
  --max-step 60 \
  --output "$OUT_DIR/ppo_zero_six_60step_curves.png" \
  --metrics-tsv "$OUT_DIR/ppo_zero_six_60step_metrics.tsv"

cat > "$OUT_DIR/README.txt" <<EOF
Generated PR curves
===================
PPO run root:  $PPO_RUN_ROOT
GRPO run root: $GRPO_RUN_ROOT

Artifacts:
- grpo_zero_six_30step_curves.png
- grpo_zero_six_30step_metrics.tsv
- ppo_zero_six_60step_curves.png
- ppo_zero_six_60step_metrics.tsv
- grpo_30_summary.tsv
EOF

echo "OUT_DIR=$OUT_DIR"
echo "GRPO_30_PNG=$OUT_DIR/grpo_zero_six_30step_curves.png"
echo "PPO_60_PNG=$OUT_DIR/ppo_zero_six_60step_curves.png"
