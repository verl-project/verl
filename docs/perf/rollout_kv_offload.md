# Rollout KV Cache Offload via Mooncake-Store

Last updated: 11/11/2026.

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas. This also helps
long-tail load balancing: when work migrates to idle rollout replicas, shared
prefix KV reduces the re-prefill cost.

`actor_rollout_ref.rollout.cache_pool` is off by default. When it is enabled,
verl starts one `mooncake_master` for the Ray job, writes the Mooncake JSON,
and attaches a Store connector. A hand-written
`engine_kwargs.vllm.kv_transfer_config` still works while `cache_pool` stays
off.

## Setup Mooncake + vLLM

Install the Mooncake package that matches the accelerator:
`mooncake-transfer-engine` on GPU, `mooncake-transfer-engine-npu` on NPU.
vLLM's guide covers the connector and the JSON fields:
**<https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/>**

`cache_pool.master.auto_start` defaults to true. verl then launches
`mooncake_master` on the driver node and writes the bind address into
`cache_pool.master.address`. Port `null` or `0` picks a free port and retries
when that port is already in use. To keep a master you already started, set
`master.auto_start=false` and set either `master.address` or
`store.config_path`. `auto_start=true` together with `store.config_path`
raises.

## Enable `cache_pool`

Requirements:

- `rollout.name=vllm`
- `rollout.enable_prefix_caching=true`
- Leave `engine_kwargs.vllm.kv_transfer_config` unset. Setting both raises.

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    enable_prefix_caching: true
    cache_pool:
      enabled: true
```

Hydra:

```bash
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.enable_prefix_caching=true \
actor_rollout_ref.rollout.cache_pool.enabled=true
```

Non-PD rollout attaches one Store connector with `kv_role=kv_both`:

- GPU uses `MooncakeStoreConnector`. `store.protocol` defaults to `rdma`
  (`tcp` is also allowed). `connector.store_tp_size` defaults to
  `tensor_model_parallel_size`.
- NPU uses `AscendStoreConnector`. `store.protocol` defaults to `ascend`.
  `store.global_segment_size` must be 1GB-aligned, and `store.device_name`
  stays empty. A non-null `store.ssd_offload_path` turns on SSD offload.

Each server receives `PYTHONHASHSEED` from `cache_pool.python_hash_seed`
(default `0`), plus `MOONCAKE_CONFIG_PATH` and `VERL_RAY_JOB_ID`. Reward and
teacher replicas reject `cache_pool.enabled=true`.

`backend` accepts `mooncake`, `memcache`, and `yuanrong`. Only `mooncake` runs
today, and `store.mode` must stay `embedded`. On GPU, `store.enable_offload=true`
and a non-null `store.ssd_offload_path` raise. NPU ignores `enable_offload`.

Field comments for `master`, `store`, and `connector` live in
`verl/trainer/config/rollout/rollout.yaml`.

## Prefill-decode rollout

With `rollout.disaggregation.enabled=true`, verl builds a `MultiConnector`:
the P2P connector first, then the Store connector.

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    enable_prefix_caching: true
    disaggregation:
      enabled: true
      transfer_backend: mooncake
    cache_pool:
      enabled: true
```

`disaggregation.transfer_backend` must be `mooncake` or `nixl`. On NPU, `nixl`
is remapped to `mooncake` and the P2P connector is `MooncakeConnectorV2`. GPU
keeps `NixlConnector` or `MooncakeConnector`. The Store child is
`MooncakeStoreConnector` on GPU and `AscendStoreConnector` on NPU.

Ascend `MooncakeConnectorV2` reserves `tp + 1` consecutive handshake ports so
the scheduler control port stays clear of the worker ports. GPU Mooncake uses
one bootstrap port.

On GPU PD, `connector.store_tp_size` defaults to the LCM of the prefill and
decode tensor-parallel sizes. `connector.enable_store_tp_lcm=true` and an
explicit `store_tp_size` together raise. `save_decode_cache`,
`consumer_is_to_put`, and `consumer_is_to_load` require PD.
`connector.use_layerwise` is the NPU `memcache` prefill field.

Before an Ascend colocated sleep, every server disconnects its Mooncake peers
and then the servers sleep together. Abort runs every prefill server to
completion before any decode server aborts. An aborted prefill output is
returned so the client can retry after resume.

## Manual `kv_transfer_config`

Leave `cache_pool.enabled` at false and pass the connector through
`engine_kwargs`. verl forwards `engine_kwargs.vllm.*` to `vllm serve`.

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        kv_transfer_config: |-
          {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
              "mooncake_config_path": "/path/to/mooncake_config.json"
            }
          }
```

Or as a Hydra CLI override:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.mooncake_config_path=/path/to/mooncake_config.json
```

## RL correctness: hard reset on every weight update

verl clears both local and Mooncake KV caches at every weight update boundary
to avoid reusing KV from the previous policy. The same
`reset_prefix_cache(reset_connector=True)` path runs after `cache_pool`
attaches its Store connector.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.
