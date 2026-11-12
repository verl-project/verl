# SFT (Supervised Fine-Tuning) on Google Cloud TPU (v6e)

This directory contains examples and scripts for running **Multi-Turn / Instruction SFT (Supervised Fine-Tuning)** on Google Cloud TPU v6e clusters using `verl`, `verl-hardware-plugin`, and the **TorchTitan** engine (`engine=torchtitan`).

Both `verl` SFT entrypoints are supported on TPU:
- **Multi-Controller (`torchrun` SPMD, default)**: `torchrun ... -m verl.trainer.sft_trainer` (`SFT_TRAINER_MODE=torchrun`, for single-host TPU VMs or GKE JobSet)
- **Single-Controller (Ray)**: `python3 -m verl.trainer.sft_trainer_ray` (`SFT_TRAINER_MODE=ray`, for KubeRay `ray job submit`)

The training setup uses:
- **Training Engine**: TorchTitan (`engine=torchtitan`, PyTorch FSDP2 on `torch_tpu`)
- **Sequence Packing & Bucketing**: `model.use_remove_padding=True` with `data.pad_mode=no_padding`, `engine.pad_to_length=True`, and `engine.pad_to_length_bucket=256` to pad packed sequences to static 256-token buckets and prevent XLA HLO recompilation
- **Parallelism**: Pure FSDP2 (`engine.tensor_parallel_size=1`, `engine.data_parallel_shard_size=8`) across 1 TPU v6e-8 slice (2 physical hosts $\times$ 4 TPU chips/host = 8 TPU chips)

---

## 🚀 Quick Start on GKE TPU Cluster

### 1. Prepare Model Checkpoint & Dataset

Ensure the model checkpoint and preprocessed GSM8K-SFT parquet files are available under `/data` (or override `RAY_DATA_HOME` / `MODEL_PATH` / `TRAIN_FILE` / `TEST_FILE`):
- `MODEL_PATH`: `/data/assets/hf/Qwen3-0.6B`
- `TRAIN_FILE`: `/data/data/gsm8k_sft/train.parquet`
- `TEST_FILE`: `/data/data/gsm8k_sft/test.parquet`

```bash
# Preprocess GSM8K multi-turn SFT parquet files inside the cluster
python3 examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir /data/data/gsm8k_sft
```

---

### 2. Verify Cluster Readiness & Set Up Port Forwarding

Verify that the Ray head pod and TPU worker pods are `Running` and port-forward the Ray Dashboard service (`8265`) to local port `23333`:

```bash
# Check pod status
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster

# Port-forward Ray head dashboard service to localhost:23333
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
```

---

### 3. Submit an SFT Training Job

#### Option A: Ray Single-Controller (`verl.trainer.sft_trainer_ray`, `SFT_TRAINER_MODE=ray`)

Submit the SFT job from the root of the `verl` repository using `ray job submit` (with `verl-hardware-plugin` checked out at `../verl-hardware-plugin`, see [`examples/tpu/grpo/README.md`](../grpo/README.md#1-get-the-code)):

```bash
export RAY_ADDRESS="http://localhost:23333"
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv", "logs", "*.log", "*.pt", "*.bin"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}'

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- env SFT_TRAINER_MODE=ray bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

For a quick 8-step smoke test with validation at Step 4 and Step 8, pass `SMOKE_TEST=1`:

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- env SFT_TRAINER_MODE=ray SMOKE_TEST=1 bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

#### Option B: Multi-Controller SPMD via `torchrun` (`verl.trainer.sft_trainer`, Default)

To run directly with `torchrun` (the script's default `SFT_TRAINER_MODE=torchrun`, for example on a single-host TPU v6e-4 VM or inside a multi-host JobSet container):

The `torchrun` path needs the TorchTPU slice environment: `TORCH_TPU_SLICEBUILDER_ADDRESSES` and `TORCH_TPU_TOPOLOGY`. GKE TPU pods (including the KubeRay pods above) get them from the TPU webhook. On a plain TPU VM, export them yourself; the script exits early if either is unset. For example, on a single-host TPU v6e-4 (1 host x 4 TPU chips, `2x2` topology):

```bash
export TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:8471,localhost:8472,localhost:8473,localhost:8474
export TORCH_TPU_TOPOLOGY=2,2,1
NNODES_TRAINER=1 N_CHIPS_TRAINER=4 \
  bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
```

---

## ⚙️ Key Configuration & Tuning Notes

1. **TPU Slice Topology (`NNODES_TRAINER` & `N_CHIPS_TRAINER`)**:
   - On a **TPU v6e-8** slice (`2x4` topology = 2 VM hosts $\times$ 4 TPU chips/host), `NNODES_TRAINER` **must** match the full slice host count (`NNODES_TRAINER=2`, `N_CHIPS_TRAINER=4`) so `libtpu`'s multi-host slice builder initializes all 8 chips in the slice together.
   - On a single-host **TPU v6e-4** slice (`2x2` topology = 1 VM host $\times$ 4 TPU chips), override via:
     ```bash
     NNODES_TRAINER=1 N_CHIPS_TRAINER=4 bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
     ```
2. **Pure FSDP by Default (`engine.tensor_parallel_size=1`)**:
   - Pure FSDP (`engine.tensor_parallel_size=1`, `engine.data_parallel_shard_size=<total_chips>`) is the default for best training throughput. Tensor parallelism (`TENSOR_PARALLEL_SIZE > 1`, where `DATA_PARALLEL_SHARD_SIZE` defaults to `TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE`) is also supported when needed to reduce per-chip HBM usage. If you set `DATA_PARALLEL_SHARD_SIZE` explicitly, keep `DATA_PARALLEL_SHARD_SIZE * TENSOR_PARALLEL_SIZE` equal to the number of trainer chips.
3. **Sequence Bucketing (`engine.pad_to_length=True`, `engine.pad_to_length_bucket=256`)**:
   - Packed 1D sequences are padded to multiples of `engine.pad_to_length_bucket` (default `256`, with `VERL_TPU_SEQ_BUCKET_SIZE` fallback) and aligned across data-parallel ranks so XLA compiles a bounded set of static bucket shapes (`256, 512, ..., 2048`) and reuses them with 100% cache hit rate.
4. **Qwen3 Chat Template (`data.ignore_input_ids_mismatch=True`)**:
   - Qwen3's chat template injects `<think>\n\n</think>\n\n` only on the final assistant turn and strips `<think>` blocks from earlier assistant turns, so full-conversation tokenization legitimately differs from per-turn concatenation in `MultiTurnSFTDataset`.

---

## 📊 Monitoring Progress

```bash
# Check job status
ray job status --address http://localhost:23333 <JOB_ID>

# Stream live training and validation logs
ray job logs --follow --address http://localhost:23333 <JOB_ID>
```
