#!/usr/bin/env bash
set -uo pipefail
W=/mnt/share/y00938198/kimi-k31/test_wxx_skill
attempt="$1"
[[ "$attempt" =~ ^[0-9]+$ ]] || exit 2
export KIMI_ALIGNMENT_RECURRENT_KDA=1 KIMI_ALIGNMENT_V2_SCORING=1 KIMI_ALIGNMENT_CONV_SPLIT_SILU=1
printf '%s\n' "$$" > "$W/results/launcher-423-attempt${attempt}.pid"
bash "$W/scripts/launch.sh" > "$W/logs/driver-423-attempt${attempt}.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$W/results/driver-423-attempt${attempt}.exit"
exit "$rc"
