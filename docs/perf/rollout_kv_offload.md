# Rollout KV Cache Offload via Mooncake-Store

Last updated: 2026-05-16.

This document covers how to offload prefix KV blocks from the vLLM rollout
engine to a shared **Mooncake** distributed store, so that long shared
prefixes (system prompt + task description + earlier turns in agentic
workloads) get deduplicated across requests and across rollout replicas
within a single weight generation.

## When to use this

Enable when **all** of these hold:

- Rollout workload has long, reusable prefixes (multi-turn / agentic / shared
  system prompt with many ``actor_rollout_ref.rollout.n`` samples per prompt).
- The rollout-side prefix-cache hit rate on a single engine is already
  saturated, but cross-engine prefix sharing (e.g., DP > 1, multi-replica
  fully-async) is leaving hits on the table.
- You can run a Mooncake master process colocated with (or reachable from)
  every rollout host, and you have enough RDMA / TCP bandwidth between the
  rollout workers and the master.

Do **not** enable for short prompts or small ``rollout.n`` workloads where
within-engine prefix cache is already enough — the round-trip to Mooncake
will net out negative.

## RL-correctness contract: hard reset on every weight update

The unique constraint of RL training is that the model weights change
between rollout steps. Any KV block written to the external store before a
weight update is **computed against the previous policy** — serving it to a
post-update request would silently corrupt inference.

verl handles this correctly. The three prefix-cache-reset call sites in
``vllm_async_server.py`` propagate the connector reset:

- ``wake_up`` and ``clear_kv_cache`` call
  ``engine.reset_prefix_cache(reset_connector=True)`` explicitly.
- ``abort_all_requests`` calls
  ``engine.pause_generation(clear_cache=True)``; in vLLM ≥ the paired
  cascade patch, ``EngineCore._reset_caches`` defaults
  ``reset_connector=True`` so the connector cascade fires automatically
  whenever ``pause_generation`` clears caches.

That flag cascades into vLLM as:

    Scheduler.reset_prefix_cache(reset_connector=True)
      -> Scheduler.reset_connector_cache()
        -> MooncakeStoreConnector.reset_cache()        (SCHEDULER role)
          -> MooncakeStoreScheduler.reset_store()
            -> LookupKeyClient.reset()                  (ZMQ admin frame)
              -> LookupKeyServer (worker rank 0) typed dispatch
                -> store.remove_all(force=True)         (Mooncake master)

The net effect is that after each ``update_weights`` round, the Mooncake
master is empty before any new rollout request starts — matching the
existing in-engine prefix-cache behavior. The contract is symmetric: if
the in-engine guard (``BlockPool.reset_prefix_cache``) fails to clear (e.g.
an in-flight sequence still holds blocks), the external store is *also*
left untouched, so internal and external caches never desynchronize.

When **no** KV connector is attached, ``reset_connector=True`` is a no-op
success in upstream vLLM (the scheduler treats "nothing to reset" as
trivially OK). Passing it unconditionally is therefore safe for every
rollout — there is no verl-side feature flag to remember.

## Pre-requisites

1. **vLLM**: build that includes the ``MooncakeStoreConnector.reset_cache``
   cascade (vllm-project/vllm#42694) and the ``EngineCore._reset_caches``
   default that threads ``reset_connector=True`` from
   ``pause_generation(clear_cache=True)``. Without the cascade,
   ``reset_connector=True`` clears only the local prefix cache and leaves
   the Mooncake master populated with stale KV — silent correctness loss.
   Do **not** enable the external store without that vLLM build.
2. **Mooncake**: install the Python binding (``pip install
   mooncake-transfer-engine``) and a master binary that exposes the
   ``RemoveAll`` RPC. On aarch64 GB200 build from upstream with
   ``-DUSE_CUDA=ON -DWITH_NVIDIA_PEERMEM=OFF -DUSE_MNNVL=ON``.
3. **Mooncake master process**: launched out-of-band, typically per-run
   (see ``scripts/mooncake/start_mooncake_master.sh`` in the
   mooncake-integration project). Single-tenant per run is recommended
   because ``RemoveAll`` is master-wide; multi-tenant sharing would let
   one experiment wipe another's cache.

## Configuration

verl forwards any key under ``actor_rollout_ref.rollout.engine_kwargs.vllm``
to ``vllm serve`` as a CLI flag. To attach the Mooncake connector, set
``kv_transfer_config`` directly — the JSON shape is vLLM's own
``KVTransferConfig`` schema (see ``vllm/config/__init__.py`` ->
``KVTransferConfig``):

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

Equivalently as inline JSON (single line):

```yaml
actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config: '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"mooncake_config_path":"/path/to/mooncake_config.json"}}'
```

verl serializes ``engine_kwargs.vllm.kv_transfer_config`` into the
``--kv-transfer-config`` argument of ``vllm serve``; vLLM's own arg parser
decodes the JSON into ``KVTransferConfig`` and constructs the connector.
There is no verl-side schema layer — any field vLLM accepts is accepted
here, and any future vLLM-side KV connector (NIXL, P2pNcclConnector, future
ones) can be wired the same way without a verl change.

If the Mooncake master is unreachable at engine launch, vLLM crashes the
serve subprocess. That's the intended fail-loud behavior; an RL run that
silently disables a configured KV store would hide hours of stale-cache
corruption. If you want a "soft" mode, wrap the launch with a healthcheck
of the master before starting verl.

## Required environment variables

These are intentionally *not* set automatically — they're cluster-level
choices.

- ``MOONCAKE_CONFIG_PATH``: required by the Mooncake client to locate
  ``master_server_address`` etc. Set on every rollout actor; verl
  propagates it via ``ray`` runtime_env. You can also pass it inline via
  ``kv_connector_extra_config.mooncake_config_path`` as shown above.
- ``PYTHONHASHSEED=0``: **required if** you have ``DP > 1`` or multiple
  rollout replicas — vLLM's block-hash seed is randomized per process and
  cross-engine prefix-cache hits will silently drop to zero without a
  fixed seed. Single-engine rollouts can leave this unset.

## Operational notes

- **Cluster hygiene**: a per-run master is the recommended deployment.
  Reuse across runs invites cross-experiment cache pollution (and a
  ``reset_connector=True`` from one run will wipe the other's keys).
- **Reset cost**: every weight update triggers ``RemoveAll`` on the
  master, which iterates all metadata shards. On a ~600 GB store this is
  sub-second. Frequent fully-async weight syncs (sync every 1-2 rollout
  steps) will see the per-update hit rate stay low; this is expected and
  matches the in-engine prefix-cache behavior under the same conditions.
- **No version tagging**: keys are pure content-addressed
  (``{model_name}@tp_rank:N@...@{block_hash}``); no weight-generation
  field. The hard-reset model relies on the master being cleaned per
  weight update, not on key versioning.
- **Failure modes**:
  - Master unreachable at launch: vLLM serve fails to start; verl
    surfaces the underlying connector error. Fail-loud.
  - Master goes down mid-run: the rollout engine continues using its
    in-engine prefix cache; ``reset_store`` returns False and the rest of
    ``reset_prefix_cache`` still works. Re-attaching to a new master
    mid-run is **not** supported in this version.
  - Cross-rank ``block_hash`` divergence (forgot ``PYTHONHASHSEED``):
    silent zero hit rate. Inspect
    ``vllm_external_prefix_cache_hits`` metric to detect.

## Comparison vs. SGLang's HiCacheStorage flow

| Aspect | SGLang ``/flush_cache`` | verl + vLLM ``MooncakeStoreConnector`` |
|---|---|---|
| Reset trigger | Per-RPC opt-in ``flush_cache=True`` flag on each ``update_weights_*`` call — forget once -> silent stale-cache corruption | Automatic data-plane cascade through ``reset_prefix_cache(reset_connector=True)`` whenever verl drives the existing hard-reset path; no flag |
| Multi-rank coordination | Each scheduler instance calls ``remove_all`` (idempotent, but N redundant RPCs) | Scheduler-side ZMQ admin RPC to rank-0 worker only -> one ``RemoveAll`` per reset |
| Guard / cache consistency | ``is_fully_idle()`` silently false on failure | Cascades only when in-engine guard passes; internal and external caches stay in lockstep |

## Reference

- Paired vLLM upstream PR (must be in your vLLM build):
  [vllm-project/vllm#42694](https://github.com/vllm-project/vllm/pull/42694) —
  implements ``MooncakeStoreConnector.reset_cache()``, the typed-tag ZMQ
  protocol, and the ``EngineCore._reset_caches`` default that threads
  ``reset_connector=True`` through ``pause_generation``.
- vLLM scheduler hook the cascade rides on:
  ``vllm/v1/core/sched/scheduler.py`` (``reset_prefix_cache`` and
  ``reset_connector_cache``).
- Sister rollout-correctness recipes: ``advance/rollout_corr.md``,
  ``advance/rollout_corr_math.md``.
