# Rollout KV Cache Offload via Mooncake Store

Last updated: 11/25/2026.

Offloading prefix KV blocks from vLLM rollout engines to a shared Host-DRAM
[Mooncake Store](https://github.com/kvcache-ai/Mooncake) deduplicates system
prompts, `rollout.n` group samples, and multi-turn tool histories across
requests and rollout replicas over GPUDirect RDMA—avoiding long-context
re-prefills at high concurrency and improving decode throughput.

> **Hardware prerequisite (GPUDirect RDMA):** Mooncake Store relies on zero-copy
> **GPUDirect RDMA (GDR)** (`nvidia_peermem` + RoCEv2/InfiniBand HCAs) to move KV
> blocks directly between GPU HBM and Store DRAM. **Enabling Mooncake Store is
> not recommended without GDR**: falling back to host bounce-buffer copies or
> TCP incurs severe CPU/PCIe contention and transfer latency that typically
> outweigh prefill savings.
>
> **Scope note:** All examples below target **Host DRAM** (`enable_offload: false`, no SSD tier).

Tuning `MooncakeStoreConnector` for Agentic RL focuses on two goals:

1. **Maximize NUMA-local GPUDirect RDMA bandwidth** (`GPU -> local HCA -> local DRAM`).
2. **Minimize `ibv_reg_mr` registration latency**, especially in colocated RL
   where `sleep()` / `wake_up()` re-registers GPU KV caches every step.

Mooncake Store supports two deployment modes:

- **`embedded` (Quickstart — Easiest Setup):** Each vLLM worker contributes
  `global_segment_size > 0` of in-process Host RAM; only `mooncake_master` is
  needed.
- **`standalone-store` (Recommended for Large-Scale Production):** External
  NUMA-pinned `mooncake_client` daemons own the Host DRAM pool
  (`global_segment_size: 0` in vLLM), isolating KV lifetime from vLLM
  `sleep()`/restarts and maximizing dual-socket RDMA bandwidth; see
  [Section 4](#4-advanced-large-scale-production-with-standalone-store).

---

## 1. Quickstart (`embedded` Mode + YAML Preset)

### Step 1: Start `mooncake_master`

```bash
mooncake_master \
  --rpc_address=10.0.0.10 \
  --rpc_port=50051 \
  --allocation_strategy=random \
  --enable_metric_reporting=true \
  --metrics_port=9003 &
```

### Step 2: Write `mooncake_config.json`

Write `mooncake_config.json` on a path accessible to all rollout workers:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "10.0.0.10:50051",
  "global_segment_size": "16GB",
  "local_buffer_size": "512MB",
  "protocol": "rdma",
  "device_name": "",
  "device_name_per_gpu": [
    "mlx5_bond_3", "mlx5_bond_3",
    "mlx5_bond_1", "mlx5_bond_1",
    "mlx5_bond_4", "mlx5_bond_4",
    "mlx5_bond_2", "mlx5_bond_2"
  ],
  "enable_offload": false,
  "tenant_id": "default"
}
```

### Step 3: Enable in `verl` via YAML (`+kv_offload=mooncake`)

`verl` ships a ready-to-use Hydra preset at
`verl/trainer/config/kv_offload/mooncake.yaml` with recommended Agentic RL
defaults pre-filled:

```yaml
# @package _global_
actor_rollout_ref:
  rollout:
    external_kv_cache_invalidation: salt
    engine_kwargs:
      vllm:
        kv_transfer_config:
          kv_connector: MooncakeStoreConnector
          kv_role: kv_both
          kv_load_failure_policy: recompute
          kv_connector_extra_config:
            save_decode_cache: true
            load_async: true
            lookup_async: true
            lookup_prefetch_limit: 64
            lookup_prefetch_ttl_s: 30.0
            lookup_timeout_s: 5.0

ray_kwargs:
  ray_init:
    runtime_env:
      env_vars:
        PYTHONHASHSEED: "0"
        MC_ENABLE_DEST_DEVICE_AFFINITY: "1"
```

Enable it directly with `+kv_offload=mooncake`, or extend it in a small project
YAML overlay so `MOONCAKE_CONFIG_PATH` and cluster `MC_*` variables stay in YAML
rather than cluttering launch scripts:

```yaml
# configs/kv_offload/my_mooncake.yaml
# @package _global_
defaults:
  - mooncake
  - _self_

ray_kwargs:
  ray_init:
    runtime_env:
      env_vars:
        MOONCAKE_CONFIG_PATH: "/path/to/mooncake_config.json"
        # Add optional cluster/host overrides here only when needed (see Section 3.3):
        # MC_MTU: "4096"
        # MC_USE_IPV6: "1"
        # MC_STORE_USE_HUGEPAGE: "1"
        # MC_STORE_HUGEPAGE_SIZE: "2MB"
```

Launch `verl` referencing the preset (or your overlay):

```bash
# Built-in preset:
MOONCAKE_CONFIG_PATH="/path/to/mooncake_config.json" \
  python3 -m verl.trainer.main_ppo +kv_offload=mooncake ...

# Custom YAML overlay:
python3 -m verl.trainer.main_ppo \
  --config-dir=/path/to/configs +kv_offload=my_mooncake ...
```

Under the hood, `verl` automatically:

- Assigns each `vLLMHttpServer` a deterministic `lookup_rpc_port` when omitted
  in YAML, preventing ZMQ IPC collisions across multi-GPU replicas on one host.
- Forwards driver `PYTHONHASHSEED`, `MOONCAKE_*`, `MC_*`, and `VLLM_MOONCAKE_*`
  environment variables into Ray `runtime_env["env_vars"]`.
- Sets `MC_TCP_BIND_ADDRESS` on `vLLMHttpServer` from the node's Ray IP when
  `MC_USE_IPV6="1"`.

---

## 2. RL Correctness & Capacity Essentials

- **Host DRAM & `global_segment_size` Sizing (Avoid Host OOM in `colocate_async`):**
  - *Hit-rate target:* When Host RAM is ample (e.g., `separate_async` rollout-only
    nodes), aim for **`>= 1.5x` aggregate GPU KV cache** so prior-step salted
    blocks age out via LRU without evicting active multi-turn episodes.
  - *Colocated RL memory cap (`colocate_async`):* Store segments—whether
    `n_gpus_per_node * global_segment_size` in `embedded` mode or
    `sum(mooncake_client --global_segment_size)` on the same node in
    `standalone-store` mode—stay resident in Host RAM during training steps.
    Because trainer **`param_offload` and `optimizer_offload`** (FP32 master
    weights, Adam states, plus `sleep()` weight backups and Ray/OS buffers)
    consume substantial Host RAM, **always cap total per-node Store size below
    remaining free Host RAM at peak training step** to prevent Host OOM.
- **Salted Policy Versions (`external_kv_cache_invalidation: salt`):** Enabled by
  `+kv_offload=mooncake`. `verl` tags requests with `cache_salt="<run_id>:<global_steps>"`
  and calls `sleep(reset_connector=False)` (requires vLLM > 0.33.0 with
  [vllm#57810](https://github.com/vllm-project/vllm/pull/57810)), so new weights
  never match stale KV while sibling or standalone replicas continue using the
  store undisturbed.
- **Sleep-Mode & Hybrid Model Rules:** Never set
  `kv_connector_extra_config.custom_mem_pool` (`NVLINK` / `BAREX`) when
  `free_cache_engine: true` (`CuMemAllocator` incompatibility). Hybrid
  Attention + Mamba/GDN models (e.g., Qwen3.5) require vLLM >= 0.34
  (`mamba_cache_mode="align"`, auto-enabled).

---

## 3. Configuration Reference

### 3.1 Rollout YAML (`kv_transfer_config` — Pre-Set in `+kv_offload=mooncake`)

- **`save_decode_cache: true`** *(vLLM default: `false`)*: Writes newly decoded
  full KV blocks back to Store so turn $t+1$ hits assistant responses from turn
  $t$ instead of re-prefilling.
- **`load_async: true` & `kv_load_failure_policy: "recompute"`**: Overlaps RDMA
  `GET` with GPU compute and falls back to local prefill if blocks were evicted.
- **`lookup_async: true` + `lookup_prefetch_limit: 64` + `lookup_prefetch_ttl_s: 30.0` + `lookup_timeout_s: 5.0`**:
  **Tune as a bundle.** `lookup_async: false` repeatedly re-queries master while
  requests wait for GPU blocks, whereas `lookup_async: true` with stock defaults
  (`limit=0, ttl=2s, timeout=1s`) times out under bursts and causes false misses.

### 3.2 `mooncake_config.json` (Topology & Buffers)

- **`device_name_per_gpu`** *(mutually exclusive with `device_name`; **preferred**)*:
  Maps each physical host GPU ID (`0..7`; provide all 8 entries even for GPU
  subsets) to 1 PCIe-local **data** HCA, excluding management NICs (`mlx5_0` /
  `mlx5_bond_0`). Avoids multi-NIC `ibv_reg_mr` fan-out on every `wake_up()`
  with negligible multi-GPU throughput difference.
- **`local_buffer_size: "512MB"`** *(vLLM default: `"4GB"`)*: Zero-copy GPUDirect
  RDMA bypasses this host bounce buffer when `enable_offload: false`; shrinking
  it saves ~28 GB host RAM per 8-GPU node and cuts registration overhead.
- **`global_segment_size`** *(most critical capacity knob)*:
  - In `embedded` mode, sets **per-worker** Host DRAM contribution (`> 0`, e.g.,
    `"16GB"` -> `128 GB` total across 8 workers/node); in `standalone-store`
    mode, must be `0` in `mooncake_config.json` while actual node pool capacity
    is set via `mooncake_client --global_segment_size`.
- **`protocol: "rdma"`**: Required for production and `free_cache_engine: true`.

### 3.3 Worker Environment Variables (`ray_kwargs.ray_init.runtime_env.env_vars`)

All rollout worker environment variables can be declared in YAML under
`ray_kwargs.ray_init.runtime_env.env_vars`:

- **`PYTHONHASHSEED: "0"`** *(in `mooncake.yaml`)*: Mandatory for consistent
  prefix block hashes across worker processes (`0%` cross-replica hit rate if unset).
- **`MC_ENABLE_DEST_DEVICE_AFFINITY: "1"`** *(in `mooncake.yaml`)*: Routes RDMA
  transfers to the HCA local to the target memory region, avoiding cross-socket
  and ToR hairpins.
- **`MC_ENABLE_PARALLEL_REG_MR: "1"`** *(multi-NIC processes only)*: Parallelizes
  `ibv_reg_mr` across multiple HCAs (~1.4x–2.2x faster on 2–4 NIC processes).
  **Keep omitted/unset on 1-NIC `device_name_per_gpu` vLLM workers** (especially
  in `colocate_async`), where extra lock contention slows concurrent 8-GPU
  `wake_up()` by ~2x.
- **`MC_STORE_USE_HUGEPAGE: "1"` & `MC_STORE_HUGEPAGE_SIZE: "2MB"`**: Add in YAML
  **only** when host `/proc/meminfo` reports `HugePages_Total > 0` (512x fewer
  page-table entries). **Keep omitted otherwise**—Mooncake treats *any* set
  value (even `"0"`) as mandatory HugeTLB without 4KB fallback, and already uses
  THP `MADV_HUGEPAGE` by default.
- **Optional Fabric Alignment** *(add only when required by your cluster)*:
  - `MC_MTU`: Match switch port MTU (`"1024"` / `"2048"` / `"4096"`).
  - `MC_USE_IPV6: "1"`: Enables IPv6 TCP handshakes (`verl` auto-sets
    `MC_TCP_BIND_ADDRESS` on vLLM servers; set `MC_TCP_BIND_ADDRESS` explicitly
    on external `mooncake_client` daemons).
  - `MC_GID_INDEX`: Leave unset by default (Mooncake `AutoGid` negotiates active
    `RoCE v2` GIDs once management NICs are excluded); set only if data HCAs
    carry multiple routable `RoCE v2` GIDs.
  - `MC_NUM_QP_PER_EP: "4"`: Optional in `standalone-store` mode on active-active
    (`802.3ad` / `balance-xor`) dual-port `mlx5_bond_*` NICs to avoid per-QP LAG
    hash polarization (~15%–19% higher single-link throughput).

---

## 4. Advanced: Large-Scale Production with `standalone-store`

For large-scale multi-node or colocated RL production runs, prefer
**`standalone-store` with 2 NUMA-Split `mooncake_client` daemons per dual-socket
host**:

- **Better fault tolerance (decoupled lifecycle):** Cached KV pools live in
  independent `mooncake_client` daemons and survive vLLM `sleep()` and worker
  crashes/restarts (still sharing node Host RAM when colocated on GPU hosts, so
  size `--global_segment_size` within trainer offload headroom).
- **Full dual-socket RDMA bandwidth (~2x):** A single cross-NUMA `mooncake_client`
  allocates on one socket and relegates the second socket's HCAs to backup;
  splitting into 2 NUMA-local daemons activates all HCAs across both sockets and
  cuts pool registration time by >2x.
- **Per-process MR parallelism:** Enables `MC_ENABLE_PARALLEL_REG_MR=1` on 2-NIC
  `mooncake_client` daemons while keeping it unset on 1-NIC vLLM workers.

### Step 1: Launch `mooncake_master` and 2 NUMA-Split `mooncake_client` Daemons

```bash
MASTER_HOST="10.0.0.10"
HOST_ID="$(hostname -s)"

mooncake_master \
  --rpc_address="${MASTER_HOST}" --rpc_port=50051 \
  --allocation_strategy=random \
  --enable_metric_reporting=true --metrics_port=9003 &

# Socket 0 daemon (NUMA 0 DRAM + Socket 0 data HCAs only; exclude mgmt NICs)
MOONCAKE_HOST_ID="${HOST_ID}" \
MOONCAKE_LOCAL_HOSTNAME="${MASTER_HOST}:12300" \
MOONCAKE_CLIENT_HTTP_PORT=9300 \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_ENABLE_PARALLEL_REG_MR=1 \
numactl --cpunodebind=0 --membind=0 mooncake_client \
  --host="${MASTER_HOST}" --port=50152 \
  --enable_http_server=true --http_port=9300 \
  --metadata_server=P2PHANDSHAKE \
  --master_server_address="${MASTER_HOST}:50051" \
  --protocol=rdma --device_names="mlx5_bond_3,mlx5_bond_1" \
  --global_segment_size="192GB" --local_buffer_size="0" &

# Socket 1 daemon (NUMA 1 DRAM + Socket 1 data HCAs only)
MOONCAKE_HOST_ID="${HOST_ID}" \
MOONCAKE_LOCAL_HOSTNAME="${MASTER_HOST}:12301" \
MOONCAKE_CLIENT_HTTP_PORT=9301 \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_ENABLE_PARALLEL_REG_MR=1 \
numactl --cpunodebind=1 --membind=1 mooncake_client \
  --host="${MASTER_HOST}" --port=50153 \
  --enable_http_server=true --http_port=9301 \
  --metadata_server=P2PHANDSHAKE \
  --master_server_address="${MASTER_HOST}:50051" \
  --protocol=rdma --device_names="mlx5_bond_4,mlx5_bond_2" \
  --global_segment_size="192GB" --local_buffer_size="0" &
```

> **`--allocation_strategy` selection:** **`random`** (default) in multi-node
> Agentic RL to avoid single-node capacity skew from long-tail multi-turn
> sessions and GRPO `rollout.n` read hotspots. Switch to **`local_first`** (with
> a single node-level `MOONCAKE_HOST_ID` shared by local daemons and workers)
> only when cross-node spine bandwidth is oversubscribed, or **`free_ratio_first`**
> across heterogeneous RAM nodes.

### Step 2: Switch `mooncake_config.json` to `standalone-store`

Only two fields change from the Quickstart JSON (`verl` YAML stays identical):

```json
{
  "mode": "standalone-store",
  "global_segment_size": 0
}
```

---

## 5. References

- [vLLM MooncakeStoreConnector Usage Guide](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
- [Mooncake Store Deployment & Tuning Guide](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)
- [Mooncake Store Architecture & Design](https://kvcache-ai.github.io/Mooncake/design/store/mooncake-store.html)
- [vLLM PR #57810: Preserve external KV connector cache across `sleep(reset_connector=False)`](https://github.com/vllm-project/vllm/pull/57810)
