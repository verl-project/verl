# llm-router Inference Quick Start

## What is llm-router

llm-router provides intelligent routing for Agentic RL based on KV cache status and load awareness. It serves as a pluggable load balancer for verl's multi-replica (data-parallel) vLLM inference, replacing the default `global_sticky_inflight` router.

Core components:
- `KVCAwareBalancer` — routing framework, manages component lifecycle and routing decisions
- `Collector` — Transport + Decoder composition, collects metrics for decisions (vLLM KV events and Prometheus metrics)
- `Strategy` — scoring strategy, computes combined scores based on KV cache hit rate and load
- `Store` — singleton storage, caches collected metrics and KV block states

---

## Setup and Usage Guide

### 1. Clone the project

```bash
git clone https://github.com/verl-project/uni-agent.git
```

### 2. Create Docker container

```bash
# Optional env vars (with defaults): CONTAINER_NAME=hgq-swe  IMAGE_NAME=verlai/verl:vllm018.dev1  SHM_SIZE=10g
# DATA_DIR: host data disk path (models/datasets/wheels), replace with your actual path
CONTAINER_NAME=swe-xxx IMAGE_NAME=verlai/verl:vllm018.dev1 SHM_SIZE=10g \
docker run -d \
  --name ${CONTAINER_NAME:-hgq-swe} \
  --gpus all \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --shm-size=${SHM_SIZE:-10g} \
  -v <DATA_DIR>:<DATA_DIR> \
  -v /tmp:/tmp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  --entrypoint sleep \
  ${IMAGE_NAME:-verlai/verl:vllm018.dev1} \
  infinity

# Verify GPU visibility
docker exec ${CONTAINER_NAME:-hgq-swe} nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

> `<DATA_DIR>` is the host data disk path (models, datasets, swe-rex wheels). Replace with your actual path (e.g. `/data1`). This path is mounted into the container and must match the swe-rex wheels mount source in `agent_config_localdocker.yaml` (see §4.5).

### 3. Enter container and install dependencies

```bash
docker exec -it swe-xxx bash
cd /path/to/uni-agent

# Init git submodule + install verl and dependencies
git submodule update --init --recursive
pip install --no-deps -e verl
pip install swe-rex loguru pydantic pydantic_settings boto3
pip install --no-cache-dir swebench
```

### 4. Prepare dataset

```bash
# Default (modal) — works with local docker sandbox
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py --local-save-dir examples/kvc_aware_router

# Specify deployment backend
DEPLOYMENT=vefaas python examples/data_preprocess/swe_bench_verified.py --local-save-dir examples/kvc_aware_router
```

| DEPLOYMENT | Generated image name | Use case |
|------------|---------------------|----------|
| `modal` | `swebench/sweb.eval.x86_64.*` | local docker / modal |
| `vefaas` | Alibaba Cloud veFaaS image | veFaaS only |
| `local` | Not implemented yet | — |

Output: `examples/kvc_aware_router/swe_bench_verified_<deployment>.parquet`

### 4.5 Pre-download swe-rex wheels + pull SWE-bench images (first-time setup)

**Problem 1: Concurrent pip throttle** — Each sandbox runs `pip install swe-rex`. With 500 concurrent sandboxes, pip will throttle/timeout.

**Solution**: Pre-download swe-rex + all dependency wheels to a local directory for offline install:

```bash
# <WHEELS_DIR> should be under <DATA_DIR> (mounted in §2), e.g. /data1/swe_wheels
pip download swe-rex -d <WHEELS_DIR>
```

**Problem 2: Find image list + pull SWE-bench images**

Image names are stored in the parquet `extra_info` field (format: `swebench/sweb.eval.x86_64.<instance>`). Parse with pyarrow:

```python
import json, pyarrow.parquet as pq
t = pq.read_table("examples/kvc_aware_router/swe_bench_verified_modal.parquet")
imgs = set()
for e in t.column("extra_info").to_pylist():
    e = json.loads(e) if isinstance(e, str) else e
    def find(o):
        if isinstance(o, dict):
            for v in o.values(): find(v)
        elif isinstance(o, str) and "sweb.eval" in o:
            imgs.add(o)
    find(e)
print(imgs)
```

Pull from Docker Hub (or a CN mirror like Volcano Engine CR if Docker Hub is slow):

```bash
docker pull swebench/sweb.eval.x86_64.<instance>:latest
```

### 5. Run inference

`examples/kvc_aware_router/run_infer.sh` is a thin wrapper around `parallel_infer.py`. **Defaults (defined in parallel_infer.py) = 2-GPU single-replica smoke test (1 sample)**. The first 3 args are positional; the rest are forwarded to `parallel_infer.py` via `$@`.

```bash
# Smoke test (defaults: 2 GPUs, TP=1, 1 sample)
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B

# Full 8-GPU data-parallel
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B \
    --num-workers 8 --n-gpus-per-node 8 --tensor-parallel-size 2 \
    --max-num-seqs 64 --max-samples -1 --prompt-length 31744

# With MooncakeStoreConnector (cross-replica KV sharing; requires mooncake_master)
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B --enable-mooncake

# The KVCAware router is always used (hardcoded in parallel_infer.py).
# Ascend (vllm-ascend) backend — selects MooncakeConnectorStoreV1 + mem util 0.8:
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B \
    --device ascend --enable-mooncake
```

Positional args:

| Arg | Position | Default | Description |
|-----|----------|---------|-------------|
| `MODEL_PATH` | 1st | **required** | Model path |
| `DATA_PATH` | 2nd | Same dir `swe_bench_verified_modal.parquet` | Dataset path |
| `AGENT_CONFIG` | 3rd | Same dir `agent_config_localdocker.yaml` | Agent config path |

Key CLI args (forwarded to `parallel_infer.py`, see `--help` for all):

| Arg | Default | Description |
|-----|---------|-------------|
| `--num-workers` | `1` | Agent rollout workers (use 8 for full runs) |
| `--n-gpus-per-node` | `2` | GPUs per node (use 8 for full runs) |
| `--tensor-parallel-size` | `1` | Tensor parallel (use 2 for full runs) |
| `--max-num-seqs` | `256` | Concurrent sequences per engine (use 64 on 24GB cards) |
| `--max-samples` | `-1` | Number of samples (-1 = full dataset) |
| `--prompt-length` | `4096` | Prompt length (use 31744 for full runs) |
| `--response-length` | `8192` | Response length |
| `--max-model-len` | unset | Model context length (unset = engine default) |
| `--n` | `1` | Rollouts per prompt (use 4 for full runs) |
| `--enable-mooncake` | off | Attach mooncake connector (cross-replica KV sharing) |
| `--mooncake-config-path` | `mooncake_config.json` | Mooncake config path (with `--enable-mooncake`) |
| `--device` | `gpu` | Backend: `gpu` → MooncakeStoreConnector, mem 0.9; `ascend` → MooncakeConnectorStoreV1, mem 0.8 |

> Set `CUDA_VISIBLE_DEVICES` before calling run_infer.sh (e.g. `CUDA_VISIBLE_DEVICES=6,7 bash run_infer.sh ...`). Concurrency is configured in the agent_config YAML.

> **1-token degradation**: verl's `max_tokens = min(response_length, prompt_length + response_length - prompt)`. Once a multi-turn prompt exceeds this sum, `max_tokens` collapses to 1. Keep the sum below the model's native context (e.g. Qwen3-8B 40960 → use 31744+8192=39936).

### 6. Known issues: transformers / numpy errors

On older Docker/kernel/OS versions (Docker 20.10, kernel 4.15, Ubuntu 18.04), the same image may trigger runtime exceptions. On newer environments (Docker 26.x, kernel 5.4, Ubuntu 20.04) these work out of the box.

| Symptom | Root cause | Fix (run once in container) |
|---------|-----------|-----------------------------|
| `import transformers` fails with `Backend should be defined in BACKENDS_MAPPING` | transformers 5.3.0 backend check fails on old envs | `pip install "transformers==4.57.6"` |
| vllm worker fails, `RecursionError` (numpy `issubdtype` ↔ `__repr__` recursion) | numpy 2.x overlay triggers dtype repr loop on old envs | `pip uninstall -y numpy && pip install "numpy==1.26.4"` |
