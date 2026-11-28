# 1x H20 (96GB) measured results — Qwen2.5-0.5B-Instruct GRPO on GSM8K

Formal 100-step comparison: `logs/3_2_h20/run_20261119_045736` (2026-11-19).
Both tiers use **identical sampling** (`train_batch_size=64`, `rollout.n=8`,
`ppo_mini_batch_size=32`, lr 1e-6, KL coef 0.001, prompt/response 512/512,
`test_freq=10`) — only resource knobs differ, so validation scores are directly
comparable. 10-step smoke runs + reproducibility check: `h20_results_10step.md`.

| | MIN | RECOMMENDED |
|---|---|---|
| Resource knobs | optimizer CPU offload + gradient checkpointing + micro bsz 2 + vLLM share 0.35 | dynamic token batching (32K/GPU) + vLLM share 0.7 + chunked prefill |
| Peak GPU memory (nvidia-smi) | **38.9 GB** | **74.6 GB** |
| Actor torch alloc | 9.3 GB | 59.5 GB |
| sec/step (tqdm, 512 traj/step) | 80.1 | 21.0 |
| Total wall (100 steps + 10 validations) | 2 h 16 m | 37 m |
| GSM8K val score @ step 100 | **0.5474** | **0.5390** |

Validation score trajectory (`val-aux/openai/gsm8k/reward/mean@1`, greedy; legacy
metric name `val/test_score/openai/gsm8k`):

| step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|------|------|------|------|------|------|------|------|------|------|------|
| MIN | 0.255 | 0.455 | 0.481 | 0.498 | 0.511 | 0.529 | 0.542 | 0.543 | 0.541 | **0.547** |
| RECOMMENDED | 0.262 | 0.464 | 0.497 | 0.509 | 0.514 | 0.525 | 0.534 | 0.546 | 0.558 | **0.539** |

Conclusions:

- Identical sampling yields statistically equivalent learning curves (final gap
  0.8 pp, within run-to-run noise; REC peaked at 0.558 at step 90). The tiers are
  interchangeable in training quality.
- MIN saves **35.7 GB (48%) of GPU memory** at the cost of **3.8x wall time**
  (80.1 vs 21.0 s/step; 2 h 16 m vs 37 m total) — the classic
  memory-throughput tradeoff on a single H20.
- MIN's optimizer offload pushed host RAM usage to ~387 GB (measured in the
  10-step smoke run). Budget host memory before using MIN on smaller machines.
- Both runs completed 100/100 steps with exit code 0 and saved checkpoints at
  steps 50/100 (`checkpoints/verl_tuning_h20/*_20261119_045736/`).

Reproduce: `scripts/run_h20_benchmark.sh` (see its header for
launch/monitor/stop commands; `TOTAL_TRAINING_STEPS`, `TRAIN_BATCH_SIZE`,
`ROLLOUT_N`, `PPO_MINI_BATCH_SIZE`, `TEST_FREQ`, `SAVE_FREQ` are env knobs).

