# veRL uv example

🍳 A minimal example for running [verl](https://github.com/volcengine/verl) RL training using [uv](https://github.com/astral-sh/uv) for dependency management on Ray cluster.

> ⚡ **Why uv + Ray to use veRL?**
>
> - 🚀 **Blazing-fast sync at any scale** — uv is 10–100× faster than pip, propagating the exact same environment to every Ray worker instantly, with no per-node installs required
> - 🔒 **Fully locked environment** — lockfile-based installs eliminate version mismatches across nodes
> - 🐳 **No Docker rebuilds** — iterate on dependencies locally and ship to the cluster immediately, without touching a Dockerfile

## Prerequisites

- **CUDA environment** — `nvidia/cuda:12.9.1-devel-ubuntu22.04` works well (same base as verl's [stable image](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm))
- **git** and **curl**:
  ```bash
  apt-get update && apt-get install -y git curl
  ```
- **uv** ([docs](https://docs.astral.sh/uv/getting-started/installation/)):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Setup

**1. Install dependencies**

```bash
uv sync --extra gpu
source .venv/bin/activate  # activate the virtualenv
```

Alternatively, skip activation and prefix commands with `uv run --extra gpu --frozen`.

**2. Download the GSM8K dataset**

```bash
cd verl && python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

## Running

### Option 1: Direct or Local (on Single Node)

```bash
bash example.sh
# or without activating the virtualenv:
uv run --extra gpu --frozen bash example.sh
```

### Option 2: Ray Cluster

`ray_runtime.yaml` is required — it tells Ray workers to use the same uv environment as the driver via `py_executable: "uv -v run --frozen --extra gpu"`, which mirrors the `uv run --extra gpu --frozen` command used in Option 1.

```bash
# Start a local Ray cluster (skip if one is already running)
ray start --head

# Submit the job
ray job submit \
  --address http://127.0.0.1:8265 \
  --runtime-env ray_runtime.yaml \
  -- bash example.sh

# Tear down when done
ray stop
```
