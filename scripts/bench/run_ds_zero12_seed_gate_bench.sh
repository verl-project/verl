#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ubuntu/verl
STAMP=$(date +%Y%m%d_%H%M%S)
STEPS=${STEPS:-60}
SEEDS=(${SEEDS:-1234 2024 3407})
STAGES=(1 2)
GPU_PAIRS_DEFAULT=("0,1" "2,3" "4,5" "6,7")
GPU_PAIRS_OVERRIDE=${GPU_PAIRS_OVERRIDE:-}
if [[ -n "$GPU_PAIRS_OVERRIDE" ]]; then
  read -r -a GPU_PAIRS <<<"$GPU_PAIRS_OVERRIDE"
else
  GPU_PAIRS=("${GPU_PAIRS_DEFAULT[@]}")
fi
MAX_RETRIES=${MAX_RETRIES:-2}
SCORE_DELTA_THRESH=${SCORE_DELTA_THRESH:-0.020000}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
VERL_PATCH=${VERL_DS_ZERO2_FP32_ACCUM_PATCH:-1}
ZERO2_STEP_EACH_MICRO=${ZERO2_STEP_EACH_MICRO:-0}

RUN_TAG=${RUN_TAG:-ds_stage12_seed${STEPS}_gate}
RUN_ROOT=${RUN_ROOT:-$BASE/outputs/${RUN_TAG}_${STAMP}}
mkdir -p "$RUN_ROOT" /home/ubuntu/ray_tmp

cat > "$RUN_ROOT/run_config.env" <<CFG
RUN_ROOT=$RUN_ROOT
RUN_TAG=$RUN_TAG
STEPS=$STEPS
SEEDS=${SEEDS[*]}
ROLLOUT_MODE=$ROLLOUT_MODE
VERL_DS_ZERO2_FP32_ACCUM_PATCH=$VERL_PATCH
VERL_DS_ZERO2_STEP_EACH_MICRO=$ZERO2_STEP_EACH_MICRO
SCORE_DELTA_THRESH=$SCORE_DELTA_THRESH
CFG

COMMON_ARGS=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE"
  actor_rollout_ref.rollout.load_format=auto
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.max_model_len=512
  actor_rollout_ref.rollout.max_num_seqs=8
  actor_rollout_ref.rollout.max_num_batched_tokens=1024
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B
  critic.model.path=Qwen/Qwen3-0.6B
  actor_rollout_ref.model.trust_remote_code=True
  critic.model.trust_remote_code=True
  data.trust_remote_code=True
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
  +critic.model.override_config.attn_implementation=flash_attention_2
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  critic.ppo_mini_batch_size=32
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  critic.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
  critic.ppo_max_token_len_per_gpu=2048
  actor_rollout_ref.actor.use_kl_loss=False
  algorithm.use_kl_in_reward=False
  data.train_files=/home/ubuntu/data/gsm8k_ppo/train.parquet
  data.val_files=/home/ubuntu/data/gsm8k_ppo/test.parquet
  data.train_batch_size=64
  data.max_prompt_length=512
  data.max_response_length=512
  trainer.project_name="$RUN_TAG"
  trainer.save_freq=0
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.resume_mode=disable
  trainer.total_training_steps="$STEPS"
  trainer.logger=[console]
  +ray_kwargs.ray_init.address=local
  +ray_kwargs.ray_init.num_gpus=2
  +ray_kwargs.ray_init.include_dashboard=False
)

kill_case_gpu_processes() {
  local gpus=$1
  local ids
  ids=$(echo "$gpus" | tr ',' ' ')
  for gid in $ids; do
    local bus
    bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F',' -v id="$gid" '$1+0==id {gsub(/ /,"",$2); print $2}')
    if [[ -z "$bus" ]]; then
      continue
    fi
    local pids
    pids=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | awk -F',' -v b="$bus" '$1 ~ b {gsub(/ /,"",$2); print $2}')
    if [[ -n "$pids" ]]; then
      echo "$pids" | xargs -r kill -9 || true
    fi
  done
}

run_case_once() {
  local case_name=$1
  local stage=$2
  local seed=$3
  local gpus=$4
  local attempt=$5

  local log_dir="$RUN_ROOT/$case_name"
  local log_file="$log_dir/train.log"
  mkdir -p "$log_dir"
  [[ "$attempt" == "1" ]] && : >"$log_file"

  local ray_tmp="/tmp/r$(printf '%s' "${case_name}_a${attempt}" | md5sum | cut -c1-12)"
  rm -rf "$ray_tmp"
  mkdir -p "$ray_tmp"

  kill_case_gpu_processes "$gpus"

  local mode_args=(
    trainer.experiment_name="$case_name"
    trainer.default_local_dir="$log_dir/checkpoints"
    +ray_kwargs.ray_init._temp_dir="$ray_tmp"
    trainer.use_legacy_worker_impl=enable
    data.seed="$seed"
    actor_rollout_ref.actor.data_loader_seed="$seed"
    critic.data_loader_seed="$seed"
    actor@actor_rollout_ref.actor=ds_actor
    critic=ds_critic
    actor_rollout_ref.actor.strategy=deepspeed
    actor_rollout_ref.actor.deepspeed.zero_stage="$stage"
    critic.deepspeed_config.zero_stage="$stage"
    actor_rollout_ref.actor.deepspeed.offload=none
    critic.deepspeed_config.offload=none
  )

  echo "[start] case=$case_name attempt=$attempt stage=$stage seed=$seed gpus=$gpus $(date -Is)" | tee -a "$log_file" "$RUN_ROOT/driver.log"

  set +e
  env CUDA_VISIBLE_DEVICES="$gpus" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    RAY_TMPDIR="$ray_tmp" \
    TMPDIR="$ray_tmp" \
    XDG_CACHE_HOME=/home/ubuntu/.cache \
    TRITON_CACHE_DIR=/home/ubuntu/.cache/triton \
    TORCH_EXTENSIONS_DIR=/home/ubuntu/.cache/torch_extensions \
    PYTHONPATH="$BASE" \
    HYDRA_FULL_ERROR=1 \
    VLLM_SKIP_MEMORY_PROFILING=1 \
    DS_SKIP_CUDA_CHECK=1 \
    VERL_DS_ZERO2_FP32_ACCUM_PATCH="$VERL_PATCH" \
    VERL_DS_ZERO2_STEP_EACH_MICRO="$ZERO2_STEP_EACH_MICRO" \
    NCCL_DEBUG=WARN \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    stdbuf -oL -eL python3 -m verl.trainer.main_ppo -cn ppo_trainer \
    "${COMMON_ARGS[@]}" "${mode_args[@]}" >>"$log_file" 2>&1
  local rc=$?
  set -e

  local step_line
  step_line=$(rg "step:${STEPS} -" "$log_file" | tail -n 1 || true)
  if [[ $rc -eq 0 && -n "$step_line" ]]; then
    echo "[done] case=$case_name attempt=$attempt rc=0 $(date -Is)" | tee -a "$log_file" "$RUN_ROOT/driver.log"
    return 0
  fi

  echo "[fail] case=$case_name attempt=$attempt rc=$rc $(date -Is)" | tee -a "$log_file" "$RUN_ROOT/driver.log"
  return 1
}

run_case_with_retry() {
  local case_name=$1
  local stage=$2
  local seed=$3
  local gpus=$4

  local attempt
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    if run_case_once "$case_name" "$stage" "$seed" "$gpus" "$attempt"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

declare -a JOB_CASES=()
declare -a JOB_STAGES=()
declare -a JOB_SEEDS=()

for seed in "${SEEDS[@]}"; do
  for stage in "${STAGES[@]}"; do
    case_name="ds_s${stage}_seed${seed}"
    JOB_CASES+=("$case_name")
    JOB_STAGES+=("$stage")
    JOB_SEEDS+=("$seed")
  done
done

worker_loop() {
  local worker_id=$1
  local gpus=${GPU_PAIRS[$worker_id]}
  local n=${#JOB_CASES[@]}
  local idx=$worker_id
  local failed=0

  while [[ $idx -lt $n ]]; do
    if ! run_case_with_retry "${JOB_CASES[$idx]}" "${JOB_STAGES[$idx]}" "${JOB_SEEDS[$idx]}" "$gpus"; then
      failed=1
    fi
    idx=$((idx + ${#GPU_PAIRS[@]}))
  done
  return $failed
}

# Static strided assignment keeps stage/seed scheduling deterministic while
# utilizing all configured GPU pairs.
declare -a PIDS=()
for wid in "${!GPU_PAIRS[@]}"; do
  worker_loop "$wid" &
  PIDS+=($!)
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

summary_file="$RUN_ROOT/summary.tsv"
compare_file="$RUN_ROOT/compare.tsv"
overall_file="$RUN_ROOT/overall.tsv"
gate_file="$RUN_ROOT/gate_status.txt"

python3 - <<'PY' "$RUN_ROOT" "$summary_file" "$compare_file" "$overall_file" "$gate_file" "$STEPS" "$SCORE_DELTA_THRESH"
import csv
import os
import re
import statistics
import sys

run_root, summary_file, compare_file, overall_file, gate_file, steps, score_thresh = sys.argv[1:]
steps = int(steps)
score_thresh = float(score_thresh)
start_tail = max(1, steps - 4)

case_dirs = sorted([d for d in os.listdir(run_root) if d.startswith('ds_s') and os.path.isdir(os.path.join(run_root, d))])
summary_rows = []

for case in case_dirs:
    m = re.match(r'ds_s(\d+)_seed(\d+)$', case)
    if not m:
        continue
    stage = int(m.group(1))
    seed = int(m.group(2))
    log = os.path.join(run_root, case, 'train.log')
    txt = open(log, 'rb').read().decode('utf-8', 'ignore') if os.path.exists(log) else ''

    rows = []
    for line in txt.splitlines():
        mm = re.search(r'step:(\d+) -', line)
        if not mm:
            continue
        step = int(mm.group(1))
        sm = re.search(r'critic/score/mean:([-+0-9.eE]+)', line)
        tm = re.search(r'perf/throughput:([-+0-9.eE]+)', line)
        ma = re.search(r'perf/max_memory_allocated_gb:([-+0-9.eE]+)', line)
        mr = re.search(r'perf/max_memory_reserved_gb:([-+0-9.eE]+)', line)
        score = float(sm.group(1)) if sm else None
        thru = float(tm.group(1)) if tm else None
        mem_alloc = float(ma.group(1)) if ma else None
        mem_reserved = float(mr.group(1)) if mr else None
        rows.append((step, score, thru, mem_alloc, mem_reserved))

    last_step = max([r[0] for r in rows], default=0)
    step_target_score = None
    step_target_mem_alloc = None
    step_target_mem_reserved = None
    for step, score, _, mem_alloc, mem_reserved in rows:
        if step == steps and score is not None:
            step_target_score = score
        if step == steps and mem_alloc is not None:
            step_target_mem_alloc = mem_alloc
        if step == steps and mem_reserved is not None:
            step_target_mem_reserved = mem_reserved

    tail = [r for r in rows if start_tail <= r[0] <= steps]
    tail_scores = [r[1] for r in tail if r[1] is not None]
    tail_thru = [r[2] for r in tail if r[2] is not None]
    tail_mem_alloc = [r[3] for r in tail if r[3] is not None]
    tail_mem_reserved = [r[4] for r in tail if r[4] is not None]
    peak_mem_reserved = [r[4] for r in rows if r[4] is not None]

    summary_rows.append({
        'case': case,
        'seed': seed,
        'stage': stage,
        'last_step': last_step,
        'step_target_score': step_target_score,
        'avg_score_tail5': (sum(tail_scores) / len(tail_scores)) if tail_scores else None,
        'avg_throughput_tail5': (sum(tail_thru) / len(tail_thru)) if tail_thru else None,
        'step_target_mem_alloc': step_target_mem_alloc,
        'step_target_mem_reserved': step_target_mem_reserved,
        'avg_mem_alloc_tail5': (sum(tail_mem_alloc) / len(tail_mem_alloc)) if tail_mem_alloc else None,
        'avg_mem_reserved_tail5': (sum(tail_mem_reserved) / len(tail_mem_reserved)) if tail_mem_reserved else None,
        'peak_mem_reserved': max(peak_mem_reserved) if peak_mem_reserved else None,
        'status': 'ok' if last_step >= steps else 'incomplete',
    })

with open(summary_file, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow([
        'case','seed','stage','last_step',
        f'step{steps}_score_mean',f'avg_score_{start_tail}_{steps}',
        f'avg_throughput_{start_tail}_{steps}',
        f'step{steps}_max_memory_allocated_gb',f'avg_max_memory_allocated_{start_tail}_{steps}_gb',
        f'step{steps}_max_memory_reserved_gb',f'avg_max_memory_reserved_{start_tail}_{steps}_gb',
        'peak_max_memory_reserved_gb','status'
    ])
    for r in summary_rows:
        w.writerow([
            r['case'], r['seed'], r['stage'], r['last_step'],
            'NA' if r['step_target_score'] is None else f"{r['step_target_score']:.6f}",
            'NA' if r['avg_score_tail5'] is None else f"{r['avg_score_tail5']:.6f}",
            'NA' if r['avg_throughput_tail5'] is None else f"{r['avg_throughput_tail5']:.6f}",
            'NA' if r['step_target_mem_alloc'] is None else f"{r['step_target_mem_alloc']:.6f}",
            'NA' if r['avg_mem_alloc_tail5'] is None else f"{r['avg_mem_alloc_tail5']:.6f}",
            'NA' if r['step_target_mem_reserved'] is None else f"{r['step_target_mem_reserved']:.6f}",
            'NA' if r['avg_mem_reserved_tail5'] is None else f"{r['avg_mem_reserved_tail5']:.6f}",
            'NA' if r['peak_mem_reserved'] is None else f"{r['peak_mem_reserved']:.6f}",
            r['status'],
        ])

by_seed = {}
for r in summary_rows:
    by_seed.setdefault(r['seed'], {})[r['stage']] = r

compare_rows = []
delta_vals = []
all_reached = True
for seed in sorted(by_seed.keys()):
    s1 = by_seed[seed].get(1)
    s2 = by_seed[seed].get(2)
    if s1 is None or s2 is None:
        continue
    if s1['last_step'] < steps or s2['last_step'] < steps:
        all_reached = False
    d_step = None
    d_avg = None
    if s1['step_target_score'] is not None and s2['step_target_score'] is not None:
        d_step = s2['step_target_score'] - s1['step_target_score']
    if s1['avg_score_tail5'] is not None and s2['avg_score_tail5'] is not None:
        d_avg = s2['avg_score_tail5'] - s1['avg_score_tail5']
        delta_vals.append(d_avg)
    compare_rows.append({
        'seed': seed,
        'stage1_last_step': s1['last_step'],
        'stage2_last_step': s2['last_step'],
        'stage1_step_target_score': s1['step_target_score'],
        'stage2_step_target_score': s2['step_target_score'],
        'delta_step_target_score_s2_minus_s1': d_step,
        'stage1_avg_score_tail5': s1['avg_score_tail5'],
        'stage2_avg_score_tail5': s2['avg_score_tail5'],
        'delta_avg_score_s2_minus_s1': d_avg,
        'stage1_avg_throughput_tail5': s1['avg_throughput_tail5'],
        'stage2_avg_throughput_tail5': s2['avg_throughput_tail5'],
        'delta_avg_throughput_s2_minus_s1': None if s1['avg_throughput_tail5'] is None or s2['avg_throughput_tail5'] is None else s2['avg_throughput_tail5'] - s1['avg_throughput_tail5'],
        'stage1_avg_mem_reserved_tail5': s1['avg_mem_reserved_tail5'],
        'stage2_avg_mem_reserved_tail5': s2['avg_mem_reserved_tail5'],
        'delta_avg_mem_reserved_tail5_s2_minus_s1': None if s1['avg_mem_reserved_tail5'] is None or s2['avg_mem_reserved_tail5'] is None else s2['avg_mem_reserved_tail5'] - s1['avg_mem_reserved_tail5'],
    })

with open(compare_file, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow([
        'seed','stage1_last_step','stage2_last_step',
        f'stage1_step{steps}_score',f'stage2_step{steps}_score','delta_step_target_score_s2_minus_s1',
        'stage1_avg_score_tail5','stage2_avg_score_tail5','delta_avg_score_s2_minus_s1',
        'stage1_avg_throughput_tail5','stage2_avg_throughput_tail5','delta_avg_throughput_s2_minus_s1',
        'stage1_avg_mem_reserved_tail5','stage2_avg_mem_reserved_tail5','delta_avg_mem_reserved_tail5_s2_minus_s1',
    ])
    for r in compare_rows:
        w.writerow([
            r['seed'], r['stage1_last_step'], r['stage2_last_step'],
            'NA' if r['stage1_step_target_score'] is None else f"{r['stage1_step_target_score']:.6f}",
            'NA' if r['stage2_step_target_score'] is None else f"{r['stage2_step_target_score']:.6f}",
            'NA' if r['delta_step_target_score_s2_minus_s1'] is None else f"{r['delta_step_target_score_s2_minus_s1']:.6f}",
            'NA' if r['stage1_avg_score_tail5'] is None else f"{r['stage1_avg_score_tail5']:.6f}",
            'NA' if r['stage2_avg_score_tail5'] is None else f"{r['stage2_avg_score_tail5']:.6f}",
            'NA' if r['delta_avg_score_s2_minus_s1'] is None else f"{r['delta_avg_score_s2_minus_s1']:.6f}",
            'NA' if r['stage1_avg_throughput_tail5'] is None else f"{r['stage1_avg_throughput_tail5']:.6f}",
            'NA' if r['stage2_avg_throughput_tail5'] is None else f"{r['stage2_avg_throughput_tail5']:.6f}",
            'NA' if r['delta_avg_throughput_s2_minus_s1'] is None else f"{r['delta_avg_throughput_s2_minus_s1']:.6f}",
            'NA' if r['stage1_avg_mem_reserved_tail5'] is None else f"{r['stage1_avg_mem_reserved_tail5']:.6f}",
            'NA' if r['stage2_avg_mem_reserved_tail5'] is None else f"{r['stage2_avg_mem_reserved_tail5']:.6f}",
            'NA' if r['delta_avg_mem_reserved_tail5_s2_minus_s1'] is None else f"{r['delta_avg_mem_reserved_tail5_s2_minus_s1']:.6f}",
        ])

mean_delta = statistics.mean(delta_vals) if delta_vals else float('nan')
max_abs_delta = max([abs(v) for v in delta_vals], default=float('nan'))
gate_pass = bool(all_reached and delta_vals and abs(mean_delta) <= score_thresh)

with open(overall_file, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['metric','value'])
    w.writerow(['num_cases', len(summary_rows)])
    w.writerow(['num_paired_seeds', len(compare_rows)])
    w.writerow([f'all_reached_step{steps}', str(all_reached).lower()])
    w.writerow(['score_delta_threshold_abs', f"{score_thresh:.6f}"])
    w.writerow(['mean_delta_avg_score_s2_minus_s1', f"{mean_delta:.6f}" if delta_vals else 'NA'])
    w.writerow(['max_abs_delta_avg_score_s2_minus_s1', f"{max_abs_delta:.6f}" if delta_vals else 'NA'])
    w.writerow(['gate_pass', str(gate_pass).lower()])

with open(gate_file, 'w', encoding='utf-8') as f:
    f.write('PASS\n' if gate_pass else 'FAIL\n')
PY

if [[ $failed -ne 0 ]]; then
  echo "WARNING: one or more runs failed" | tee -a "$RUN_ROOT/driver.log"
fi

echo "RUN_ROOT=$RUN_ROOT"
echo "SUMMARY=$summary_file"
echo "COMPARE=$compare_file"
echo "OVERALL=$overall_file"
echo "GATE=$gate_file"
