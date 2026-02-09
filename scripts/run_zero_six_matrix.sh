#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ubuntu/verl
STAMP=$(date +%Y%m%d_%H%M%S)
STEPS=${STEPS:-60}
SEED=${SEED:-7777}
RUN_TAG=${RUN_TAG:-zero_six_${STEPS}_seed${SEED}}
RUN_ROOT="$BASE/outputs/${RUN_TAG}_${STAMP}"
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ZERO2_STEP_EACH_MICRO=${ZERO2_STEP_EACH_MICRO:-0}
VERL_PATCH=${VERL_DS_ZERO2_FP32_ACCUM_PATCH:-1}
MAX_RETRIES=${MAX_RETRIES:-2}

mkdir -p "$RUN_ROOT" /home/ubuntu/ray_tmp /home/ubuntu/ds_offload

COMMON_ARGS=(
  actor@actor_rollout_ref.actor=ds_actor
  critic=ds_critic
  actor_rollout_ref.actor.strategy=deepspeed
  trainer.use_legacy_worker_impl=enable
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
  data.seed="$SEED"
  actor_rollout_ref.actor.data_loader_seed="$SEED"
  critic.data_loader_seed="$SEED"
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

cat > "$RUN_ROOT/run_config.env" <<CFG
RUN_ROOT=$RUN_ROOT
RUN_TAG=$RUN_TAG
STEPS=$STEPS
SEED=$SEED
ROLLOUT_MODE=$ROLLOUT_MODE
VERL_DS_ZERO2_FP32_ACCUM_PATCH=$VERL_PATCH
VERL_DS_ZERO2_STEP_EACH_MICRO=$ZERO2_STEP_EACH_MICRO
CFG

run_case_once() {
  local case_name=$1
  local gpus=$2
  local zero_stage=$3
  local offload=$4
  local attempt=$5
  local enable_param_offload=0

  local log_dir="$RUN_ROOT/$case_name"
  local log_file="$log_dir/train.log"
  local short_id
  short_id=$(printf '%s' "$case_name" | md5sum | cut -c1-8)
  local ray_tmp="/tmp/ray_${short_id}"
  local offload_dir="/home/ubuntu/ds_offload/$case_name"
  mkdir -p "$log_dir" "$ray_tmp" "$offload_dir"
  if [[ "$attempt" -eq 1 ]]; then
    : >"$log_file"
  fi

  echo "[start] case=$case_name attempt=$attempt stage=$zero_stage offload=$offload gpus=$gpus $(date -Is)" | tee -a "$log_file"

  local extra=(
    trainer.experiment_name="$case_name"
    trainer.default_local_dir="$log_dir/checkpoints"
    +ray_kwargs.ray_init._temp_dir="$ray_tmp"
    actor_rollout_ref.actor.deepspeed.zero_stage="$zero_stage"
    critic.deepspeed_config.zero_stage="$zero_stage"
  )
  if [[ "$offload" == "cpu" ]]; then
    enable_param_offload=1
    extra+=(
      actor_rollout_ref.actor.deepspeed.offload=cpu
      critic.deepspeed_config.offload=cpu
      actor_rollout_ref.actor.deepspeed.offload_dir="$offload_dir"
      critic.deepspeed_config.offload_dir="$offload_dir"
    )
  else
    extra+=(
      actor_rollout_ref.actor.deepspeed.offload=none
      critic.deepspeed_config.offload=none
    )
  fi

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
    VERL_ENABLE_PARAM_OFFLOAD="$enable_param_offload" \
    VERL_DS_ZERO2_FP32_ACCUM_PATCH="$VERL_PATCH" \
    VERL_DS_ZERO2_STEP_EACH_MICRO="$ZERO2_STEP_EACH_MICRO" \
    NCCL_DEBUG=WARN \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    stdbuf -oL -eL python3 -m verl.trainer.main_ppo -cn ppo_trainer \
    "${COMMON_ARGS[@]}" "${extra[@]}" >>"$log_file" 2>&1
  local rc=$?
  echo "[done] case=$case_name attempt=$attempt rc=$rc $(date -Is)" | tee -a "$log_file"
  return $rc
}

run_case_with_retry() {
  local case_name=$1
  local gpus=$2
  local zero_stage=$3
  local offload=$4
  local rc=1

  for attempt in $(seq 1 "$MAX_RETRIES"); do
    if run_case_once "$case_name" "$gpus" "$zero_stage" "$offload" "$attempt"; then
      rc=0
      break
    fi
    local log_file="$RUN_ROOT/$case_name/train.log"
    local step_line
    step_line=$(rg "step:[0-9]+ -" "$log_file" | tail -n1 || true)
    echo "[retry] case=$case_name attempt=$attempt last_step='${step_line:-none}'" | tee -a "$log_file"
    sleep 5
  done
  return $rc
}

run_wave() {
  local jobs=("$@")
  local pids=()
  for job in "${jobs[@]}"; do
    IFS=":" read -r case_name gpus zero_stage offload <<<"$job"
    run_case_with_retry "$case_name" "$gpus" "$zero_stage" "$offload" &
    pids+=("$!")
    sleep 2
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
}

# Keep GPU pressure balanced:
# - wave1 runs ZeRO-1/2 + offload variants together (4 x 2GPU jobs)
# - wave2 runs ZeRO-3 variants after wave1 completes
# Wave 1: 4-way parallel on 8 GPUs
run_wave \
  "zero1_no_offload:0,1:1:none" \
  "zero2_no_offload:2,3:2:none" \
  "zero1_cpu_offload:4,5:1:cpu" \
  "zero2_cpu_offload:6,7:2:cpu"

# Wave 2: 2-way parallel
run_wave \
  "zero3_no_offload:0,1:3:none" \
  "zero3_cpu_offload:2,3:3:cpu"

summary_file="$RUN_ROOT/summary.tsv"
python3 - <<'PY' "$RUN_ROOT" "$summary_file" "$STEPS"
import os, re, sys
run_root, out_file, target_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
cases = [
    'zero1_no_offload','zero2_no_offload','zero3_no_offload',
    'zero1_cpu_offload','zero2_cpu_offload','zero3_cpu_offload',
]
start_tail = max(1, target_steps - 4)

with open(out_file, 'w', encoding='utf-8') as f:
    f.write(
        'case\tstatus\tlast_step\t'
        'step_target_score_mean\tavg_score_tail5\t'
        'step_target_throughput\tavg_throughput_tail5\t'
        'step_target_max_memory_allocated_gb\tavg_max_memory_allocated_tail5_gb\t'
        'step_target_max_memory_reserved_gb\tavg_max_memory_reserved_tail5_gb\t'
        'peak_max_memory_reserved_gb\t'
        'skip_zero_grad_warn_count\tlast_error\n'
    )
    for case in cases:
        log = os.path.join(run_root, case, 'train.log')
        txt = ''
        if os.path.exists(log):
            txt = open(log, 'rb').read().decode('utf-8', 'ignore')
        rows = []
        for line in txt.splitlines():
            m = re.search(r'step:(\d+) -', line)
            if not m:
                continue
            step = int(m.group(1))
            sm = re.search(r'critic/score/mean:([-+0-9.eE]+)', line)
            tm = re.search(r'perf/throughput:([-+0-9.eE]+)', line)
            ma = re.search(r'perf/max_memory_allocated_gb:([-+0-9.eE]+)', line)
            mr = re.search(r'perf/max_memory_reserved_gb:([-+0-9.eE]+)', line)
            score = float(sm.group(1)) if sm else None
            thru = float(tm.group(1)) if tm else None
            mem_alloc = float(ma.group(1)) if ma else None
            mem_reserved = float(mr.group(1)) if mr else None
            rows.append((step, score, thru, mem_alloc, mem_reserved, line))

        last_step = 0
        step_target_score = 'NA'
        step_target_thru = 'NA'
        step_target_mem_alloc = 'NA'
        step_target_mem_reserved = 'NA'
        if rows:
            last_step = max(r[0] for r in rows)
            for step, score, thru, mem_alloc, mem_reserved, _ in rows:
                if step == target_steps:
                    if score is not None:
                        step_target_score = str(score)
                    if thru is not None:
                        step_target_thru = str(thru)
                    if mem_alloc is not None:
                        step_target_mem_alloc = str(mem_alloc)
                    if mem_reserved is not None:
                        step_target_mem_reserved = str(mem_reserved)

        tail = [r for r in rows if start_tail <= r[0] <= target_steps]
        scores = [r[1] for r in tail if r[1] is not None]
        thrus = [r[2] for r in tail if r[2] is not None]
        mem_allocs = [r[3] for r in tail if r[3] is not None]
        mem_reserved = [r[4] for r in tail if r[4] is not None]
        avg_score = f"{(sum(scores)/len(scores)):.6f}" if scores else 'NA'
        avg_thru = f"{(sum(thrus)/len(thrus)):.6f}" if thrus else 'NA'
        avg_mem_alloc = f"{(sum(mem_allocs)/len(mem_allocs)):.6f}" if mem_allocs else 'NA'
        avg_mem_reserved = f"{(sum(mem_reserved)/len(mem_reserved)):.6f}" if mem_reserved else 'NA'
        peak_mem_reserved_candidates = [r[4] for r in rows if r[4] is not None]
        peak_mem_reserved = f"{max(peak_mem_reserved_candidates):.6f}" if peak_mem_reserved_candidates else 'NA'

        status = 'PASS' if last_step >= target_steps else 'FAIL'
        warn_skip = txt.count('Skip zero_grad because DeepSpeed engine.module is missing')

        err_lines = []
        for line in txt.splitlines():
            if re.search(r'Traceback|RuntimeError|TypeError|ActorDiedError|ray.exceptions', line):
                err_lines.append(line.strip())
        last_error = err_lines[-1] if err_lines else 'NA'
        last_error = last_error.replace('\t', ' ')

        f.write(
            f"{case}\t{status}\t{last_step}\t{step_target_score}\t{avg_score}\t"
            f"{step_target_thru}\t{avg_thru}\t"
            f"{step_target_mem_alloc}\t{avg_mem_alloc}\t"
            f"{step_target_mem_reserved}\t{avg_mem_reserved}\t"
            f"{peak_mem_reserved}\t"
            f"{warn_skip}\t{last_error}\n"
        )
PY

echo "RUN_ROOT=$RUN_ROOT"
echo "SUMMARY=$summary_file"
