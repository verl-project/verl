# DeepSpeed Worker Integration (Current PR Scope)

This document summarizes the current DeepSpeed worker capabilities, feature boundaries, and reproducible launch commands.

## 1. Feature Support Matrix

| Feature | Status | Notes |
| --- | --- | --- |
| Data Parallel (DP) | Supported | DP layout is used for actor/ref/critic/reward workers. |
| ZeRO-1 | Supported | Offload is disabled in this worker path. |
| ZeRO-2 | Supported | Optimizer offload supported; parameter offload disabled. |
| ZeRO-3 | Supported | Optimizer offload supported; parameter offload is env-gated. |
| CPU/NVMe offload | Supported (stage-aware) | `offload=cpu/nvme/auto` is normalized by ZeRO stage. |
| Checkpoint save/load | Supported | Native DS checkpoint manager (`DeepSpeedCheckpointManager`). |
| Activation checkpointing | Supported | Controlled by model/deepspeed config flags. |
| Ulysses / SP | Not supported | Forced to `ulysses_sequence_parallel_size=1`. |

Key env knobs:

- `VERL_ENABLE_PARAM_OFFLOAD=1`: allow ZeRO-3 param offload.
- `VERL_DS_ZERO2_FP32_ACCUM_PATCH=1`: enable ZeRO-2 accumulation patch.
- `VERL_DS_ZERO2_STEP_EACH_MICRO=0|1`: ZeRO-2 micro-step behavior switch.

## 2. Startup Command Examples

### 2.1 Single-run DeepSpeed PPO (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
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
  data.train_files=/home/ubuntu/data/gsm8k_ppo/train.parquet \
  data.val_files=/home/ubuntu/data/gsm8k_ppo/test.parquet \
  trainer.total_training_steps=20 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger=[console]
```

### 2.2 Six-case ZeRO benchmark

```bash
SEED=7777 STEPS=60 RUN_TAG=zero_six_60_seed7777 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 ZERO2_STEP_EACH_MICRO=0 \
bash scripts/bench/run_ds_zero_six_case_bench.sh
```

Cases:

- `zero1_no_offload`
- `zero2_no_offload`
- `zero3_no_offload`
- `zero1_cpu_offload`
- `zero2_cpu_offload`
- `zero3_cpu_offload`

Outputs:

- `outputs/<RUN_TAG>_<timestamp>/summary.tsv`
- `outputs/<RUN_TAG>_<timestamp>/<case>/train.log`

### 2.3 Plot six-case curves (memory/throughput/score/loss)

```bash
python3 scripts/bench/plot_ds_zero_six_curves.py \
  --run-root outputs/zero_six_60_seed7777_20260212_115956
```

Outputs:

- `outputs/zero_six_60_seed7777_20260212_115956/zero_six_curves.png`
- `outputs/zero_six_60_seed7777_20260212_115956/zero_six_curves_metrics.tsv`

The `loss` curve uses `critic/vf_loss`.

## 3. This Round Benchmark Artifacts

Reference run root:

- `outputs/zero_six_60_seed7777_20260212_115956`

Summary files:

- `outputs/zero_six_60_seed7777_20260212_115956/summary.tsv`
- `outputs/zero_six_60_seed7777_20260212_115956/summary_complete.tsv`

All six cases reached `step=60` in the merged summary.

