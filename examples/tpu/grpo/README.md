# GRPO RL Training on Google Cloud TPU with `verl` + `verl-hardware-plugin`

This directory contains the reference script (`run_qwen3_0_6b_torchtitan.sh`) for running **GRPO (Group Relative Policy Optimization) RL training** on Google Cloud TPU v6e using `verl` core together with [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

The training setup uses:
- **Actor Engine**: TorchTitan FSDP2 (`model_engine=torchtitan`, registered by `verl-hardware-plugin`)
- **Rollout Engine**: vLLM on TPU (`actor_rollout_ref.rollout.name=vllm`)
- **Trainer Mode**: V1 disaggregated async overlap (`trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`)
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0: 8-chip TorchTitan FSDP2 trainer; Slice 1: 8-chip TP=8 vLLM rollout server)

---

## Step-by-Step Instructions

### Step 1. Clone `verl` and `verl-hardware-plugin`

On the machine from which you submit Ray jobs, clone both repositories (no Docker image rebuild or `pip install` on the cluster is required):

```bash
# 1. Clone verl 
git clone https://github.com/verl-project/verl


# 2. Clone verl-hardware-plugin alongside verl
git clone https://github.com/verl-project/verl-hardware-plugin.git ../verl-hardware-plugin
export PLUGIN_REPO="$(cd ../verl-hardware-plugin && pwd)"
```

---

### Step 2. Connect to the GKE Ray TPU Cluster

Ensure your KubeRay cluster on TPU v6e (2 slices of `v6e-8` = 4 hosts $\times$ 4 chips = 16 TPU chips total, running image `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261001-tsync990787257`) is `Running`, and port-forward the Ray head dashboard service:

```bash
# Verify 1 head pod + 4 TPU worker pods are Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster

# Port-forward the Ray Dashboard / Job Submission API to localhost:23333
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
export RAY_ADDRESS="http://localhost:23333"
```

Ensure the model and dataset paths referenced by `examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh` are mounted inside the Ray pods (or override `MODEL_PATH`, `TRAIN_FILE`, and `TEST_FILE` via environment variables):
- `MODEL_PATH`: `/data/jialei/assets/hf/Qwen3-0.6B`
- `TRAIN_FILE`: `/data/jialei/data/gsm8k/train.parquet`
- `TEST_FILE`: `/data/jialei/data/gsm8k/test.parquet`

---

### Step 3. Submit the GRPO Training Job

From the root of the `verl` repository, submit the job with `py_modules` pointing to `${PLUGIN_REPO}/verl_hardware_plugin` and `"VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"`. Ray packages both `verl` and `verl_hardware_plugin` from your local checkout and distributes them to the head pod and all TPU worker pods automatically:

```bash
export RAY_ADDRESS="http://localhost:23333"
export PLUGIN_REPO="/path/to/verl-hardware-plugin"

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"VERL_LOGGING_LEVEL\": \"INFO\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

To run a quick 5-step smoke test (`train_batch_size=4`, `max_response_length=512`, `total_training_steps=5`), pass `SMOKE_TEST=1`:

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"SMOKE_TEST\": \"1\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash -c "SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh"
```

---

### Step 4. Monitor Progress

```bash
# Check job status
ray job status --address "${RAY_ADDRESS}" <JOB_ID>

# Stream live logs
ray job logs --follow --address "${RAY_ADDRESS}" <JOB_ID>
```
