# 1x H20 (96GB) measured results — Qwen2.5-0.5B-Instruct GRPO on GSM8K

Two full benchmark runs on 2026-11-18 (10 training steps each,
`TOTAL_TRAINING_STEPS=10`, val_before_train=False):
`logs/3_2_h20/run_20261118_133419` and `run_20261118_145114`.
Numbers below regenerated and cross-checked via
`REPARSE_RUN_DIR=... bash scripts/run_h20_benchmark.sh` (rc=0).

| Tier | train_batch_size | rollout.n | peak GPU mem (nvidia-smi, run1/run2) | actor torch alloc | sec/step (tqdm) | val score after 10 steps* | actor MFU |
|------|------------------|-----------|--------------------------------------|-------------------|-----------------|---------------------------|-----------|
| MIN | 32 | 4 | 38.2 / 38.2 GB | 9.3 GB | 27.4 | 0.0751 | 0.10 |
| RECOMMENDED | 64 | 8 | 74.5 / 74.4 GB | 58.2 GB | 27.3 | 0.2502 | 0.79 |

\* `val-aux/openai/gsm8k/reward/mean@1` (legacy name `val/test_score/openai/gsm8k`),
greedy decoding. The two tiers differ in batch size and rollout.n, so scores are
not directly comparable — they confirm each tier trains and validates correctly.
Run 100+ steps with identical sampling for a fair comparison.

Notes:

- Both tiers completed 10/10 steps and saved `global_step_10` checkpoints
  (`checkpoints/verl_tuning_h20/...`), exit code 0, in both runs.
- Peak GPU memory is consistent across the two runs (±0.1 GB).
- RECOMMENDED processes 4x more trajectories per step (64x8=512 vs 32x4=128) at
  the same wall time per step => ~4x higher training throughput at the cost of
  +36 GB peak GPU memory.
- MIN tier enables optimizer CPU offload: host RAM usage reached ~387 GB
  (`actor/perf/cpu_memory_used_gb`). Budget host memory before using the MIN
  tier on smaller machines.
- Wall time per run ~6.7 min including ~2.3 min engine startup; the 10 training
  steps themselves took ~4.5 min in both tiers (tqdm prints two summary bars per
  run: 29.5/27.4 s/it for MIN and 29.6/27.3 s/it for REC).
- Full logs incl. per-step metrics: `logs/3_2_h20/run_*/train_*.log`.

Ready-to-paste rows for `docs/perf/device_tuning.rst` (already inserted in the
0.5B table):

```rst
* - MIN
  - Qwen2.5-0.5B-Instruct
  - GRPO
  - 1*H20(96G)
  - 32
  - fsdp
  - vllm
  - ``examples/tuning/0.5b/run_qwen2-0.5b_grpo_1_h20_fsdp.sh`` with ``TIER=min`` (peak 38.2GB, ~27s/step @10 steps)
  - `(add yours)`
* - RECOMMENDED
  - Qwen2.5-0.5B-Instruct
  - GRPO
  - 1*H20(96G)
  - 64
  - fsdp
  - vllm
  - ``examples/tuning/0.5b/run_qwen2-0.5b_grpo_1_h20_fsdp.sh`` with ``TIER=recommended`` (peak 74.5GB, ~27s/step @10 steps)
  - `(add yours)`
```

