# GRPO RL Training on Google Cloud TPU (v6e)

This directory contains examples and scripts for running **GRPO (Group Relative Policy Optimization) RL Training** on Google Cloud TPU v6e instances using `verl`.

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`)
- **Rollout Engine**: vLLM (`actor_rollout_ref.rollout.name=vllm`)
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0 for Trainer/Actor, Slice 1 for Rollout)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have a running Ray cluster on TPU v6e nodes with `verl` installed across all head and worker nodes.

Environment variables required:
- `MODEL_PATH`: Path to HuggingFace model checkpoint (e.g. `/data/jialei/assets/hf/Qwen3-0.6B`)
- `TRAIN_FILE` & `TEST_FILE`: Parquet dataset files (e.g. GSM8K dataset)
- `WANDB_API_KEY` (Optional): For experiment tracking on Weights & Biases

---

### 2. Submit a GRPO Training Job

You can submit the training job to your Ray cluster using the Ray CLI (`ray job submit`):

```bash
# Set active Ray cluster address (e.g. localhost:23333 if port-forwarded)
export RAY_ADDRESS="http://localhost:23333"

# Submit GRPO RL training job
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "VLLM_USE_V1": "0",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
      "WANDB_API_KEY": "'"${WANDB_API_KEY}"'"
    }
  }' \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

---

## 📊 Monitoring Progress

### Check Job Status
```bash
ray job status --address http://localhost:23333 <JOB_ID>
```

### Stream Live Logs
```bash
ray job logs --follow --address http://localhost:23333 <JOB_ID>
```
