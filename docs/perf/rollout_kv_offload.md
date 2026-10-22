# Rollout KV Cache Offload via Mooncake-Store

Last updated: 10/21/2026.

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas. This also helps
long-tail load balancing: when work migrates to idle rollout replicas, shared
prefix KV reduces the re-prefill cost.

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

## RL correctness: hard reset on every weight update

verl clears both local and Mooncake KV caches at every weight update boundary
to avoid reusing KV from the previous policy.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.

## Salting weight versions instead of a reset

Rollout engines increasingly keep prefix KV beyond GPU memory (KV connectors,
HiCache storage), where the hard reset is either missing or too coarse. In vLLM
0.29 (and still in 0.31), `LMCacheMPConnector` and `HF3FSKVConnector` have no
`reset_cache()`, so the reset is a silent no-op and they keep serving KV from the
previous policy. On SGLang, verl's reset leaves HiCache storage in place. Where
the reset does work, each replica's pause, including fully-async rebalance
aborts that change no weights, wipes a store that other replicas may still be
reading or writing. Set

```yaml
actor_rollout_ref:
  rollout:
    external_kv_cache_invalidation: salt  # default: reset
```

to salt weight versions instead of resetting the external store. Each request
carries the engine's native `cache_salt`, set to `<run id>:<global_steps>`. The
engine mixes it into every prefix block hash, so a store keyed by those hashes
never matches entries written under earlier weights (or by an earlier run),
whether or not it can be reset. Those entries are never read again, so
LRU-style eviction reclaims them ahead of entries still in use. GPU-local caches
are still cleared on every weight update, as with `reset`; only the external
store is kept. The exception is vLLM's own sleep (`free_cache_engine`), which
always resets the connector. Usually that happens at a weight update, when the
stored entries are already unreachable, but `separate_async` switching also
sleeps hybrid replicas without one, which wipes a store that standalone replicas
share. The salt keeps both cases correct; keeping the store across sleep needs a
vLLM option to skip the connector reset
([vllm#57810](https://github.com/vllm-project/vllm/pull/57810)).

Things to know before enabling it:

- **Weight versions come from `global_steps`.** Every weight update must pass a
  new `global_steps`, which all verl trainers do; the checkpoint-engine update
  path raises if it is missing.
- **Nothing is reused across runs.** The run id is drawn per driver process, so a
  resumed job starts with a cold store.
- **Only salt-keyed stores.** On vLLM the KV connector must key its store by
  vLLM's block hashes: `MooncakeStoreConnector`, `OffloadingConnector` (what
  vLLM's `kv_offloading_size` sets up), `LMCacheMPConnector` or
  `HF3FSKVConnector`. Others, such as `LMCacheConnectorV1`, which looks blocks
  up by token ids, are rejected at startup. PD disaggregation is not supported yet.
- **Old entries are evicted, not deleted.** The store needs an eviction policy
  (vLLM's CPU offloading, for example, evicts LRU) and room for at least one
  step's worth of blocks.
- **Backends.** vLLM, SGLang and TRT-LLM all accept the salt. On SGLang it also
  covers HiCache storage (verified with the `file` storage backend), which
  `reset` does not clear. On TRT-LLM the mode only adds the salt; verl
  configures no external store for it.
- The salt is computed when a request passes the server's pause gate, so a
  request held across a weight update is salted with the weights it runs on.
