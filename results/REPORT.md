# DeepSpeed PPO/GRPO Benchmark (placeholder)

- **Model**: Qwen/Qwen2.5-0.5B-Instruct  
- **Dataset**: GSM8K (train/test parquet)  
- **Hardware**: 8x A100  
- **Runs**: ZeRO-1/2/3, offload={none,cpu}, PPO and GRPO.  

## Metrics to capture
- Throughput (tokens/s) or step time.
- Max GPU memory per rank, offload device usage.
- Reward/KL/loss curves for first few hundred steps.
- Validation accuracy on GSM8K after short training.
- Notes on activation checkpointing and offload behaviour.

## Populate
- Fill `results/summary.json` with structured metrics for each run.
- Add charts/tables here once measurements are collected.
