# Rollout KV Cache Offload via Mooncake-Store

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas.

## Setup Mooncake + vLLM

Follow vLLM's official guide for installing the Mooncake client, starting a
master, and writing the JSON config:
**<https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/>**

The verl side only consumes whatever that doc produces — no extra steps.

## Enable in verl

verl forwards `engine_kwargs.vllm.*` straight to `vllm serve` as CLI flags.
To attach the Mooncake connector, set `kv_transfer_config`:

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

Set `MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json` on every rollout
actor (verl propagates via Ray `runtime_env`). For `DP>1` or multiple rollout
replicas, also set `PYTHONHASHSEED=0` — vLLM's block-hash seed is randomized
per process and cross-engine hits drop to zero without it.

## RL correctness: hard reset on every weight update

Model weights change between rollout steps, so any KV block written to the
external store under the previous policy must be evicted before the next
rollout starts — otherwise stale KV silently corrupts inference. verl
handles this automatically via `engine.reset_prefix_cache(reset_connector=True)`
in `vllm_async_server.py`'s `wake_up` / `clear_kv_cache` / `abort_all_requests`
paths. The flag cascades through vLLM into `MooncakeStoreConnector.reset_cache()`,
which clears the master via the `RemoveAll` RPC.

**Required vLLM build**: must include
[vllm-project/vllm#42694](https://github.com/vllm-project/vllm/pull/42694)
(`MooncakeStoreConnector.reset_cache` + the `EngineCore._reset_caches` default
that threads `reset_connector=True` through `pause_generation`). Without it,
`reset_connector=True` clears only the local prefix cache and leaves the
Mooncake master populated with stale KV — silent correctness loss.
