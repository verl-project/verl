# DeepSpeed Integration Guide

This document summarizes how to run verl with the DeepSpeed backend and how to reproduce the core benchmark/demo flows used by the current PR.

## 1. What Is Supported

- DeepSpeed actor/critic worker path.
- ZeRO stage `1/2/3`.
- CPU offload benchmark variants for ZeRO stages.
- ZeRO2 behavior switches:
  - `VERL_DS_ZERO2_FP32_ACCUM_PATCH` (accumulation precision patch).
  - `VERL_DS_ZERO2_STEP_EACH_MICRO` (micro-step mode for ZeRO2).

## 2. Prerequisites

- Multi-GPU node (examples below assume 8x A100).
- Model/data already cached or reachable.
- Entry point:
  - `python3 -m verl.trainer.main_ppo -cn ppo_trainer`

## 3. Ready-to-Run Scripts

### 3.1 Six-case ZeRO benchmark

Runs six cases:

- `zero1_no_offload`
- `zero2_no_offload`
- `zero3_no_offload`
- `zero1_cpu_offload`
- `zero2_cpu_offload`
- `zero3_cpu_offload`

Command:

```bash
SEED=7777 STEPS=60 RUN_TAG=zero_six_60_seed7777_fix \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 ZERO2_STEP_EACH_MICRO=0 \
bash scripts/bench/run_ds_zero_six_case_bench.sh
```

Outputs:

- `outputs/<RUN_TAG>_<timestamp>/summary.tsv`
- `outputs/<RUN_TAG>_<timestamp>/<case>/train.log`

### 3.2 ZeRO1/2 crossover gate benchmark

Compares stage1 vs stage2 across seeds with retry and paired summaries.

Command (`step_each_micro=1`):

```bash
SEEDS="7777" STEPS=60 RUN_TAG=ds_stage12_seed60_gate_stepmicro1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 VERL_DS_ZERO2_STEP_EACH_MICRO=1 \
bash scripts/bench/run_ds_zero12_seed_gate_bench.sh
```

Command (`step_each_micro=0`):

```bash
SEEDS="7777" STEPS=60 RUN_TAG=ds_stage12_seed60_gate_stepmicro0 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
bash scripts/bench/run_ds_zero12_seed_gate_bench.sh
```

Outputs:

- `outputs/<RUN_TAG>_<timestamp>/summary.tsv`
- `outputs/<RUN_TAG>_<timestamp>/compare.tsv`
- `outputs/<RUN_TAG>_<timestamp>/overall.tsv`

## 4. Minimal DeepSpeed Demo (single run)

Use this when you only need one stage/case quick-check instead of full sweeps:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  actor@actor_rollout_ref.actor=ds_actor \
  critic=ds_critic \
  actor_rollout_ref.actor.strategy=deepspeed \
  actor_rollout_ref.actor.deepspeed.zero_stage=2 \
  critic.deepspeed_config.zero_stage=2 \
  actor_rollout_ref.actor.deepspeed.offload=none \
  critic.deepspeed_config.offload=none \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  critic.model.path=Qwen/Qwen3-0.6B \
  trainer.total_training_steps=20 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger=[console]
```

## 5. Notes for PR/Regression

- Prefer `summary.tsv` and tail-step averages (`avg_score_tail5`, `avg_throughput_tail5`) for quick comparisons.
- For ZeRO2 parity checks, always report both:
  - score delta (`z2 - z1`)
  - throughput and memory deltas
