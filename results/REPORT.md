# DeepSpeed PPO/GRPO Benchmark Report

## Run Context

- Date: `2026-02-12`
- Repo: `verl` (DeepSpeed backend improvements + GRPO reliability fix)
- GPUs: `8 x A100 40GB` (run used `CUDA_VISIBLE_DEVICES=6,7`)
- Dataset: `../data/gsm8k_ppo/{train,test}.parquet`
- Model: `Qwen/Qwen3-0.6B`

## GRPO Reliability Fix

### Root cause (previous hang)

- ZeRO-2 path could diverge collective ordering across ranks:
  - rank0 entered a scalar allreduce (`numel=1`) from grad-norm collection;
  - rank1 was still in a large gradient allreduce (`numel=440467456`);
  - NCCL watchdog then timed out.
- Trigger condition was amplified by rank-local inactive micro behavior.

### Code changes

- `verl/workers/deepspeed_workers.py`
  - Disabled ZeRO-2 inactive-micro workaround in multi-rank DP mode.
  - Removed explicit `deepspeed_engine.get_global_grad_norm()` calls in train loop.
  - Use cached DeepSpeed grad norm field (no extra collective) for metrics.

## Latest Main Validation Run

### GRPO 60-step (`outputs/grpo60_fix3_0212_180313`)

- Status: `completed`
- Steps: `60/60`
- No NCCL timeout or runtime exception found in Ray logs.

Key metrics:

- `critic/score/mean`
  - step1: `0.1422`
  - max: `0.4094`
  - step60: `0.2688`
- `perf/throughput`
  - mean: `231.82`
  - max: `263.13`
  - step60: `246.09`
- `perf/max_memory_reserved_gb`
  - max: `13.2148`
  - step60: `13.2148`
- `timing_s/step`
  - mean: `46.90`
  - step60: `43.67`
- `actor/skipped_inactive_micro`
  - min/max across 60 steps: `0.0 / 0.0`

## Other Completed Runs

- PPO smoke (`outputs/ppo_smoke_light_0212_171740`): completed 20/20.
- GRPO control (`outputs/grpo_ctrl_0212_171959`): completed 20/20.

## Artifacts

- Full train log:
  - `outputs/grpo60_fix3_0212_180313/train.log`
- Curves/metrics (run folder):
  - `outputs/grpo60_fix3_0212_180313/grpo60_fix3_0212_180313_curves.png`
  - `outputs/grpo60_fix3_0212_180313/grpo60_fix3_0212_180313_metrics.tsv`
- Curves/metrics (home):
  - `/home/ubuntu/grpo60_fix3_0212_180313_curves.png`
  - `/home/ubuntu/grpo60_fix3_0212_180313_metrics.tsv`
