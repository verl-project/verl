#!/usr/bin/env bash
# 3.2 benchmark driver: 1x NVIDIA H20 (96GB), Qwen2.5-0.5B-Instruct GRPO on GSM8K.
# Runs the MIN and RECOMMENDED tiers of
# examples/tuning/0.5b/run_qwen2-0.5b_grpo_1_h20_fsdp.sh (selected via TIER) and
# records peak GPU memory, wall time per step and the last validation score into
# examples/tuning/0.5b/h20_results.md.
#
# Designed to run detached from any frontend (survives disconnects):
#   cd /path/to/verl
#   mkdir -p logs
#   nohup setsid bash scripts/run_h20_benchmark.sh > logs/3_2_h20_benchmark.nohup 2>&1 &
#
# Progress: logs/3_2_h20/STATUS.md   Stop: touch logs/3_2_h20/STOP
# Env knobs: TOTAL_TRAINING_STEPS (default 10), MODEL_PATH, SAMPLE_INTERVAL,
#            SKIP_PREFLIGHT/SKIP_DATA/SKIP_TESTS/SKIP_MIN/SKIP_REC=1

set -uo pipefail

# scripts/ is one level below the repo root.
cd "$(dirname "$0")/.."
ROOT=$PWD
RUN_ROOT=${RUN_ROOT:-$ROOT/logs/3_2_h20}
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$RUN_ROOT/run_$RUN_ID
mkdir -p "$RUN_DIR"
STATUS=$RUN_ROOT/STATUS.md
STEPS=${TOTAL_TRAINING_STEPS:-10}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-5}
TUNING_DIR=$ROOT/examples/tuning/0.5b
TIER_SCRIPT=$TUNING_DIR/run_qwen2-0.5b_grpo_1_h20_fsdp.sh
# REPARSE_RUN_DIR=<logs/3_2_h20/run_xxx>: skip all live phases and rebuild the
# report from that run's existing logs (no GPU needed).
if [ -n "${REPARSE_RUN_DIR:-}" ]; then
    SKIP_PREFLIGHT=1
    SKIP_DATA=1
    SKIP_TESTS=1
    SKIP_MIN=1
    SKIP_REC=1
    RUN_DIR=$REPARSE_RUN_DIR
    RUN_ID=$(basename "$RUN_DIR" | sed 's/^run_//')
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/driver.log"; }
phase() { echo "- $(date '+%F %T')  $1: $2" >> "$STATUS"; log "PHASE $1 -> $2"; }
check_stop() {
    if [ -e "$RUN_ROOT/STOP" ]; then
        log "STOP file found, exiting."
        phase driver STOPPED
        exit 0
    fi
}
rm -f "$RUN_ROOT/STOP"
{
    echo "# 3.2 H20 benchmark status"
    echo ""
    echo "- $(date '+%F %T')  run: started ($RUN_ID, steps=$STEPS, model=$MODEL_PATH)"
} >> "$STATUS"
log "=== 3.2 H20 benchmark $RUN_ID (steps=$STEPS model=$MODEL_PATH) ==="

# ---------------- phase 1: preflight ----------------
if [ "${SKIP_PREFLIGHT:-0}" = 1 ]; then
    phase preflight SKIPPED
else
    {
        echo "nvidia-smi -L:"; nvidia-smi -L 2>&1; echo; echo "memory:"
        nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1
        echo; echo "disk:"; df -h "$ROOT" | tail -1; echo; echo "uv:"
        command -v uv || echo "uv NOT FOUND"
    } > "$RUN_DIR/preflight.txt" 2>&1
    if nvidia-smi -L >/dev/null 2>&1; then
        phase preflight OK
    else
        phase preflight "NO GPU VISIBLE (see preflight.txt) — continuing, runs will fail"
    fi
fi
check_stop

# ---------------- phase 2: data ----------------
if [ "${SKIP_DATA:-0}" = 1 ] || { [ -f "$HOME/data/gsm8k/train.parquet" ] && [ -f "$HOME/data/gsm8k/test.parquet" ]; }; then
    phase data "SKIPPED (parquet present or SKIP_DATA=1)"
else
    log "preparing gsm8k parquet"
    if uv run --frozen --all-packages --extra vllm --extra fsdp python3 \
        examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k" \
        > "$RUN_DIR/data_prep.log" 2>&1; then
        phase data OK
    else
        phase data FAILED
        tail -20 "$RUN_DIR/data_prep.log" | tee -a "$RUN_DIR/driver.log"
    fi
fi
check_stop

# ---------------- phase 3: cpu unit tests (3.2 config validation) ----------------
if [ "${SKIP_TESTS:-0}" = 1 ]; then
    phase unittests SKIPPED
else
    if uv run --frozen --all-packages --extra fsdp pytest tests/utils/test_validate_config_on_cpu.py -q \
        > "$RUN_DIR/unittests.log" 2>&1; then
        phase unittests OK
    else
        phase unittests FAILED
        tail -30 "$RUN_DIR/unittests.log" | tee -a "$RUN_DIR/driver.log"
    fi
fi
check_stop

# ---------------- phase 4/5: tiers ----------------
# extract_metrics <tier> <logf> <gpuf> <dur_seconds>
extract_metrics() {
    tier=$1
    logf=$2
    gpuf=$3
    dur=${4:-0}
    # Step lines look like "(TaskRunnerV1 pid=...) step:10 - global_seqlen/...".
    nsteps=$(grep -cE 'step:[0-9]+ - ' "$logf" 2>/dev/null || true)
    nsteps=${nsteps:-0}
    # tqdm tail line: "Training Progress: 100%|...| 10/10 [04:34<00:00, 29.54s/it]".
    sps=$(grep -oE '[0-9]+\.[0-9]+s/it' "$logf" 2>/dev/null | tail -1)
    sps=${sps%s/it}
    if [ -z "$sps" ] && [ "$nsteps" -gt 0 ] && [ "$dur" -gt 0 ]; then
        sps=$(awk "BEGIN{printf \"%.1f\", $dur/$nsteps}")
    fi
    sps=${sps:-0}
    peak=$(sort -n "$gpuf" 2>/dev/null | tail -1)
    peak=${peak:-0}
    peakgb=$(awk "BEGIN{printf \"%.1f\", $peak/1024}")
    # Metric naming differs across verl versions: legacy 'val/test_score/...:0.42'
    # vs current 'val-aux/openai/gsm8k/reward/mean@1:np.float64(0.42)'.
    score=$(sed 's/np\.float64(//g; s/)//g' "$logf" 2>/dev/null \
        | grep -oE 'val[a-zA-Z0-9/_@-]*gsm8k[a-zA-Z0-9/_@-]*:[0-9.]+' | tail -1)
    torchalloc=$(grep -oE 'actor/perf/max_memory_allocated_gb:[0-9.]+' "$logf" 2>/dev/null | tail -1 | cut -d: -f2)
    grep -oE 'timing/[a-z_]+:[0-9.]+' "$logf" 2>/dev/null \
        | awk -F: '{s[$1]+=$2} END {for (k in s) printf "%s %.1f\n", k, s[$1]}' | sort \
        > "$RUN_DIR/timings_$tier.txt" || true
    echo "$tier $nsteps $dur $sps $peakgb ${score:-none} torch_alloc_gb=${torchalloc:-na}" >> "$RUN_DIR/metrics.txt"
    phase "tier_$tier" "steps=$nsteps sec/step=$sps peak=${peakgb}GB torch_alloc=${torchalloc:-na}GB ${score:-}"
}

# run_tier <tier>
run_tier() {
    tier=$1
    logf=$RUN_DIR/train_$tier.log
    gpuf=$RUN_DIR/gpu_$tier.csv
    log "[$tier] launching $TIER_SCRIPT (TIER=$tier)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l "$SAMPLE_INTERVAL" \
        > "$gpuf" 2>/dev/null &
    sampler=$!
    t0=$(date +%s)
    MODEL_PATH="$MODEL_PATH" TOTAL_TRAINING_STEPS="$STEPS" TIER="$tier" \
        bash "$TIER_SCRIPT" \
        trainer.logger='["console"]' \
        "trainer.experiment_name=qwen2.5-0.5b_grpo_1xh20_${tier}_${RUN_ID}" \
        > "$logf" 2>&1
    rc=$?
    t1=$(date +%s)
    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null
    extract_metrics "$tier" "$logf" "$gpuf" "$((t1 - t0))"
    log "[$tier] finished rc=$rc wall_sec=$((t1 - t0))"
}

[ "${SKIP_MIN:-0}" = 1 ] || { run_tier min; check_stop; }
ray stop --force >/dev/null 2>&1 || true
[ "${SKIP_REC:-0}" = 1 ] || { run_tier recommended; check_stop; }
ray stop --force >/dev/null 2>&1 || true

# ---------------- phase 6: report ----------------
if [ -n "${REPARSE_RUN_DIR:-}" ]; then
    rm -f "$RUN_DIR/metrics.txt"
    extract_metrics min "$RUN_DIR/train_min.log" "$RUN_DIR/gpu_min.csv" 0
    extract_metrics recommended "$RUN_DIR/train_recommended.log" "$RUN_DIR/gpu_recommended.csv" 0
fi
{
    echo "# 1x H20 (96GB) measured results — Qwen2.5-0.5B-Instruct GRPO on GSM8K"
    echo ""
    echo "Regenerated by scripts/run_h20_benchmark.sh (run $RUN_ID, steps=$STEPS, model=$MODEL_PATH)."
    echo ""
    echo "| Tier | peak GPU mem (GB, nvidia-smi) | sec/step (tqdm) | steps | last val score | actor torch alloc (GB) |"
    echo "|------|-------------------------------|------------------|-------|----------------|-------------------------|"
    while read -r t n d s p sc rest; do
        echo "| $t | $p | $s | $n | $sc | $rest |"
    done < "$RUN_DIR/metrics.txt" 2>/dev/null
    echo ""
    echo '\* wall sec/step includes startup overhead for short runs; per-phase timing/* sums:'
    for f in "$RUN_DIR"/timings_*.txt; do
        [ -e "$f" ] || continue
        echo ""
        echo '```'
        echo "$(basename "$f"):"
        cat "$f"
        echo '```'
    done
    echo ""
    echo "Full logs: $RUN_DIR (train_min.log / train_recommended.log / gpu_*.csv)."
} > "$TUNING_DIR/h20_results.md"
phase report DONE
log "=== benchmark finished; see examples/tuning/0.5b/h20_results.md and $RUN_DIR ==="
