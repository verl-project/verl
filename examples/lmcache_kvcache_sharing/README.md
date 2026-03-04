## LMCache KV Cache Sharing Examples (Qwen + vLLM)

This folder contains runnable examples that integrate **LMCache** KV cache sharing with **Qwen** models served via **vLLM**, based on the LMCache “Share KV cache across multiple LLMs” guide ([docs](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html)).

The examples cover:

- **Centralized KV cache sharing** using a single LMCache server (`remote_backend_*` scripts).
- **Peer‑to‑peer (P2P) KV cache sharing** using LMCache’s P2P backend (`p2p_backend_*` scripts).
- **GPU vs NPU** variants for Qwen2.5‑7B‑Instruct.

All scripts fine‑tune Qwen on **GSM8K** with PPO/GRPO using `verl.trainer.main_ppo` and vLLM as the rollout engine.

---

### Prerequisites and required packages

- **Hardware**
  - At least **2 accelerators**:
    - For GPU flows, this matches the LMCache docs requirement of “your server should have at least 2 GPUs” for across‑instance sharing ([docs](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html#prerequisites)).
    - For NPU flows, use 2+ NPUs with a compatible Ascend software stack.

- **Core Python packages**
  - Install LMCache (required for both centralized and P2P examples):

    ```bash
    pip install -U lmcache
    ```

  - Install vLLM (these examples embed vLLM inside Verl instead of using the `vllm-openai` Docker image from the docs):

    ```bash
    pip install -U vllm
    ```

  - Install Verl and its training dependencies by following this repo’s installation instructions (ensures `python -m verl.trainer.main_ppo` works).

- **Additional packages for NPU / Ascend examples**
  - An NPU‑enabled PyTorch / Ascend stack (e.g., `torch-npu` or vendor‑provided wheel).
  - Ascend integration for LMCache and vLLM, e.g.:

    ```bash
    pip install -U lmcache-ascend
    ```

    so that the Ascend connector used in `kv_transfer_config`:

    ```bash
    "kv_connector": "LMCacheAscendConnectorV1Dynamic",
    "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    ```

    is importable.

- **Optional tooling (mirroring the docs)**
  - For containerized runs like in the LMCache quickstart, you can use the official `vllm/vllm-openai` image and then `pip install -U lmcache` inside the container ([docs](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html#run-the-p2p-sharing-workflow)). In this repo’s examples we instead call vLLM from Python via Verl.

---

### Files in this folder

- **Training scripts**
  - `run_qwen_remote_backend.sh`  
    Centralized LMCache server + vLLM on **GPU**.
  - `run_qwen_remote_backend_npu.sh`  
    Centralized LMCache server + vLLM on **NPU** (Ascend), using the Ascend‑specific LMCache connector.
  - `run_qwen_p2p_backend.sh`  
    P2P LMCache backend + vLLM on **GPU**, sharing KV cache across vLLM instances.
  - `run_qwen_p2p_backend_npu.sh`  
    P2P LMCache backend + vLLM on **NPU**.

- **LMCache config files**
  - `config/remote_backend_config.yaml`  
    Minimal config for **centralized** KV cache sharing, roughly mirroring `lmcache_config.yaml` in the LMCache docs:

    ```yaml
    chunk_size: 256
    local_cpu: true
    remote_url: "lm://localhost:65432"
    remote_serde: "naive"
    ```

  - `config/p2p_backend_config.yaml`  
    P2P + controller setup inspired by `p2p_example*.yaml` in the LMCache docs, but scaled to multiple ports for many workers:

    ```yaml
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    enable_async_loading: true

    # P2P configurations
    enable_p2p: true
    p2p_host: "localhost"
    p2p_init_ports: [9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967]
    p2p_lookup_ports: [9968, 9969, 9970, 9971, 9972, 9973, 9974, 9975]
    transfer_channel: "hccl"

    # Controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: [9950, 9951, 9952, 9953, 9954, 9955, 9956, 9957]

    extra_config:
      lookup_backoff_time: 0.001
    ```

---

### Centralized KV cache sharing (GPU)

This corresponds to the **“Centralized KV cache sharing”** section in the LMCache docs ([link](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html#centralized-kv-cache-sharing)).

1. **Start the LMCache centralized server** (matching `remote_backend_config.yaml`):

   ```bash
   lmcache_server localhost 65432
   ```

2. **Run the training script with centralized backend on GPU**:

   ```bash
   cd examples/lmcache_kvcache_sharing

   # LMCache config for centralized mode
   export LMCache_CONFIG_FILE=config/remote_backend_config.yaml

   bash run_qwen_remote_backend.sh
   ```

   Key pieces in `run_qwen_remote_backend.sh`:

   - `LMCache_CONFIG_FILE=remote_backend_config.yaml`
   - `actor_rollout_ref.rollout.kv_transfer_config='{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'`

---

### Centralized KV cache sharing (NPU / Ascend)

`run_qwen_remote_backend_npu.sh` is the **NPU** variant of the centralized example.

Run:

```bash
cd examples/lmcache_kvcache_sharing

export LMCache_CONFIG_FILE=config/remote_backend_config.yaml

bash run_qwen_remote_backend_npu.sh
```

Important differences from the GPU script:

- Uses an **Ascend‑specific connector** in `kv_transfer_config`:

  ```bash
  actor_rollout_ref.rollout.kv_transfer_config='{
    "kv_connector": "LMCacheAscendConnectorV1Dynamic",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
  }'
  ```

- Sets the Verl trainer device to NPU:

  ```bash
  trainer.device=npu
  ```

Your environment must have:

- NPU‑enabled PyTorch / Ascend stack
- `lmcache_ascend` package installed and importable

---

### P2P KV cache sharing (GPU)

This example implements the **“P2P KV cache sharing”** flow from the LMCache docs ([link](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html#p2p-kv-cache-sharing)), but wired into Verl PPO instead of raw vLLM CLI.

High‑level LMCache flow (from the docs):

- Start an LMCache **controller** with pull/reply ports (e.g. 8300/8400).
- For each vLLM instance:
  - Point `LMCACHE_CONFIG_FILE` (here `LMCache_CONFIG_FILE`) to a P2P config file (`p2p_backend_config.yaml`).
  - Launch vLLM with `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.
- First request populates KV cache, subsequent requests on other instances hit LMCache and save prefill time.

In this repo, `run_qwen_p2p_backend.sh` assumes:

- An LMCache **controller** is already running and reachable at:
  - `controller_pull_url: "localhost:8300"`
  - `controller_reply_url: "localhost:8400"`
- Multiple worker ports and P2P ports are available as configured in `p2p_backend_config.yaml`.

To run a simple single‑node training example (one Verl process using the P2P backend settings):

```bash
cd examples/lmcache_kvcache_sharing

# 1) Start LMCache controller (example; adjust paths/ports as needed)
lmcache_controller --host localhost --port 9000 \
  --monitor-ports '{"pull": 8300, "reply": 8400}'

# 2) Point Verl / vLLM to the P2P config
export LMCache_CONFIG_FILE=config/p2p_backend_config.yaml

# 3) Run training with P2P backend on GPU
bash run_qwen_p2p_backend.sh
```

Internally, the script sets:

- `actor_rollout_ref.rollout.kv_transfer_config='{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'`
- P2P + controller parameters via `p2p_backend_config.yaml` (ports, instance id, etc.).

---

### P2P KV cache sharing (NPU)

`run_qwen_p2p_backend_npu.sh` mirrors the GPU P2P script but targets **NPU**:

```bash
cd examples/lmcache_kvcache_sharing

export LMCache_CONFIG_FILE=config/p2p_backend_config.yaml

bash run_qwen_p2p_backend_npu.sh
```

It uses:

- The same `config/p2p_backend_config.yaml` for LMCache P2P + controller.
- NPU device selection via:

  ```bash
  trainer.device='npu'
  ```

You still need an LMCache controller running and the specified P2P/worker ports open, similar to the GPU P2P case.

---

### Notes and troubleshooting

- **Ports**: the provided configs assume localhost and a specific set of ports. If ports are already in use, adjust:
  - `remote_backend_config.yaml`: `remote_url` (for centralized server)
  - `p2p_backend_config.yaml`: `p2p_init_ports`, `p2p_lookup_ports`, `lmcache_worker_ports`, `controller_pull_url`, `controller_reply_url`

For more details on LMCache configuration, refer to the official documentation:  
[Example: Share KV cache across multiple LLMs](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html).

