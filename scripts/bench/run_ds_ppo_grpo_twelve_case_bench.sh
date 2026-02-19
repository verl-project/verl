#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
SEED=${SEED:-7777}
STEPS=${STEPS:-60}
DS_ROLLOUT_SYNC_PROFILE=${DS_ROLLOUT_SYNC_PROFILE:-0}

PPO_TAG=${PPO_TAG:-ppo_zero_six_${STEPS}_seed${SEED}_lt}
GRPO_TAG=${GRPO_TAG:-grpo_zero_six_${STEPS}_seed${SEED}_lt}

PPO_LAUNCH_LOG="outputs/${PPO_TAG}_launch_${STAMP}.log"
GRPO_LAUNCH_LOG="outputs/${GRPO_TAG}_launch_${STAMP}.log"

echo "[info] start twelve-case bench stamp=$STAMP seed=$SEED steps=$STEPS"
echo "[info] ds_rollout_sync_profile=$DS_ROLLOUT_SYNC_PROFILE"
echo "[info] repo_root=$REPO_ROOT"

echo "[info] running PPO six-case benchmark..."
SEED="$SEED" STEPS="$STEPS" RUN_TAG="$PPO_TAG" DS_ROLLOUT_SYNC_PROFILE="$DS_ROLLOUT_SYNC_PROFILE" \
  ./scripts/bench/run_ds_zero_six_case_bench.sh | tee "$PPO_LAUNCH_LOG"

PPO_RUN_ROOT=$(rg '^RUN_ROOT=' "$PPO_LAUNCH_LOG" | tail -n1 | cut -d= -f2-)
PPO_SUMMARY=$(rg '^SUMMARY=' "$PPO_LAUNCH_LOG" | tail -n1 | cut -d= -f2-)
if [[ -z "${PPO_RUN_ROOT:-}" || -z "${PPO_SUMMARY:-}" ]]; then
  echo "[error] failed to parse PPO run root/summary from $PPO_LAUNCH_LOG" >&2
  exit 1
fi

python3 ./scripts/bench/plot_ds_zero_six_curves.py \
  --run-root "$PPO_RUN_ROOT" \
  --summary "$PPO_SUMMARY" \
  --output "$PPO_RUN_ROOT/zero_six_curves.png" \
  --metrics-tsv "$PPO_RUN_ROOT/zero_six_curves_metrics.tsv"

echo "[info] running GRPO six-case benchmark..."
SEED="$SEED" STEPS="$STEPS" RUN_TAG="$GRPO_TAG" REUSE_BASELINE=0 DS_ROLLOUT_SYNC_PROFILE="$DS_ROLLOUT_SYNC_PROFILE" \
  ./scripts/bench/run_ds_zero_six_case_grpo_bench.sh | tee "$GRPO_LAUNCH_LOG"

GRPO_RUN_ROOT=$(rg '^RUN_ROOT=' "$GRPO_LAUNCH_LOG" | tail -n1 | cut -d= -f2-)
GRPO_SUMMARY=$(rg '^SUMMARY=' "$GRPO_LAUNCH_LOG" | tail -n1 | cut -d= -f2-)
if [[ -z "${GRPO_RUN_ROOT:-}" || -z "${GRPO_SUMMARY:-}" ]]; then
  echo "[error] failed to parse GRPO run root/summary from $GRPO_LAUNCH_LOG" >&2
  exit 1
fi

MANIFEST="outputs/twelve_case_manifest_${STAMP}.txt"
cat >"$MANIFEST" <<EOF
stamp=$STAMP
seed=$SEED
steps=$STEPS
ds_rollout_sync_profile=$DS_ROLLOUT_SYNC_PROFILE
ppo_run_root=$PPO_RUN_ROOT
ppo_summary=$PPO_SUMMARY
ppo_curve_png=$PPO_RUN_ROOT/zero_six_curves.png
ppo_curve_metrics=$PPO_RUN_ROOT/zero_six_curves_metrics.tsv
grpo_run_root=$GRPO_RUN_ROOT
grpo_summary=$GRPO_SUMMARY
grpo_curve_png=$GRPO_RUN_ROOT/grpo_zero_six_curves.png
grpo_curve_metrics=$GRPO_RUN_ROOT/grpo_zero_six_curves_metrics.tsv
EOF

cp "$PPO_RUN_ROOT/zero_six_curves.png" "/home/ubuntu/ppo_zero_six_curves_${STAMP}.png"
cp "$PPO_RUN_ROOT/zero_six_curves_metrics.tsv" "/home/ubuntu/ppo_zero_six_curves_metrics_${STAMP}.tsv"
cp "$GRPO_RUN_ROOT/grpo_zero_six_curves.png" "/home/ubuntu/grpo_zero_six_curves_${STAMP}.png"
cp "$GRPO_RUN_ROOT/grpo_zero_six_curves_metrics.tsv" "/home/ubuntu/grpo_zero_six_curves_metrics_${STAMP}.tsv"
cp "$MANIFEST" "/home/ubuntu/twelve_case_manifest_${STAMP}.txt"

echo "[done] manifest=$MANIFEST"
