# Rollout KV Cache Offload via Mooncake-Store

Last updated: 2026-05-11.

This document covers how to offload prefix KV blocks from the vLLM rollout
engine to a shared **Mooncake** distributed store, so that long shared
prefixes (system prompt + task description + earlier turns in agentic
workloads) get deduplicated across requests and across rollout replicas
within a single weight generation.

## When to use this

Enable when **all** of these hold:

- Rollout workload has long, reusable prefixes (multi-turn / agentic / shared
  system prompt with many `actor_rollout_ref.rollout.n` samples per prompt).
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

verl handles this correctly: the existing ``update_weights`` flow in
``verl.checkpoint_engine.base.CheckpointEngineManager`` already does an
``abort_all_requests`` -> drain -> sleep -> NCCL weight sync -> ``wake_up``
sequence. When this feature is enabled, every prefix-cache-reset call site
in ``vllm_async_server.py`` (``wake_up``, ``clear_kv_cache``, the
``abort_all_requests`` fallback path) additionally passes
``reset_connector=True`` to ``engine.reset_prefix_cache(...)``.

That flag cascades into vLLM as:

    Scheduler.reset_prefix_cache(reset_connector=True)
      -> Scheduler.reset_connector_cache()
        -> MooncakeStoreConnector.reset_cache()        (SCHEDULER role)
          -> MooncakeStoreScheduler.reset_store()
            -> LookupKeyClient.reset()                  (ZMQ admin frame)
              -> LookupKeyServer (worker rank 0) recognizes RESET_MAGIC
                -> store.remove_all(force=True)         (Mooncake master)

The net effect is that after each ``update_weights`` round, the Mooncake
master is empty before any new rollout request starts — matching the
existing in-engine prefix-cache behavior. The contract is symmetric: if
the in-engine guard (``BlockPool.reset_prefix_cache``) fails to clear (e.g.
an in-flight sequence still holds blocks), the external store is *also*
left untouched, so internal and external caches never desynchronize.

## Pre-requisites

1. **Mooncake**: install the Python binding (``pip install
   mooncake-transfer-engine``) and a master binary that exposes the
   ``RemoveAll`` RPC. On aarch64 GB200 this currently means building from
   ``ivanium/Mooncake`` ``yifan/dev`` with ``-DUSE_CUDA=ON
   -DWITH_NVIDIA_PEERMEM=OFF -DUSE_MNNVL=ON``.
2. **vLLM**: 0.20.1+ with the paired ``MooncakeStoreConnector.reset_cache``
   patch (the cascade hook). Without it the ``reset_connector=True`` flag
   is a no-op and silent stale-cache corruption is possible — do not enable
   ``kv_store.enable`` without the paired patch.
3. **Mooncake master process**: launched out-of-band (typically per-run, see
   ``scripts/mooncake/start_mooncake_master.sh``). Single-tenant per run is
   recommended because ``RemoveAll`` is master-wide; multi-tenant sharing
   would let one experiment wipe another's cache.

## Configuration

Under ``actor_rollout_ref.rollout``:

```yaml
actor_rollout_ref:
  rollout:
    kv_store:
      enable: true                                   # default false
      kv_connector: MooncakeStoreConnector           # forwarded to vLLM
      kv_role: kv_both                               # both put + get
      config_path: /path/to/mooncake_config.json     # passed via env
      extra_config: {}                               # additional kv_connector_extra_config
      on_failure: fallback                           # fallback | crash
```

Field reference:

- ``enable``: Master switch. When false (default), no ``kv_transfer_config``
  is attached to the vLLM engine and verl behaves exactly as before.
- ``kv_connector``: KVConnector class name forwarded to vLLM's
  ``--kv-transfer-config``. Override only for testing alternate backends.
- ``kv_role``: ``kv_both`` lets the rollout engine both write blocks to and
  read blocks from the store. Other values are exposed for the same reason
  as ``kv_connector`` but the rollout path expects ``kv_both``.
- ``config_path``: Path to the Mooncake client JSON config
  (``master_server_address``, ``global_segment_size``, ``protocol``, ...).
  Falls back to the ``MOONCAKE_CONFIG_PATH`` environment variable when
  unset — most installations should set the env once at the cluster level
  and leave this field empty.
- ``extra_config``: Dict merged into vLLM's ``kv_connector_extra_config``.
  Reserve for connector-specific knobs that don't have a first-class field.
- ``on_failure``: Behavior when the Mooncake master is unreachable at
  rollout-engine launch.
  ``fallback`` (default) drops the connector and continues training with
  external offload disabled — a soft dependency suited to long RL runs
  where pausing training for an infra hiccup is more expensive than losing
  cross-engine prefix hits. Set to ``crash`` to fail fast (e.g., in CI).

## Required environment variables

These are intentionally *not* set automatically — they're cluster-level
choices.

- ``MOONCAKE_CONFIG_PATH``: required by the Mooncake client to locate
  ``master_server_address`` etc. Set on every rollout actor; verl
  propagates it via ``ray`` runtime_env.
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
  - Master unreachable at launch: see ``on_failure``.
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
| Failure surface | Client must read return value and re-issue | Soft-dependency (``on_failure=fallback``) by default; explicit False propagates up to the scheduler |

## Reference

- Paired vLLM upstream patch: implements
  ``MooncakeStoreConnector.reset_cache()`` and the
  ``RESET_MAGIC`` ZMQ discriminator (see ``aoshen02/vllm:feat/mooncake-clear-hook``
  -> ``ivanium/vllm`` -> upstream).
- vLLM scheduler hook the cascade rides on:
  ``vllm/v1/core/sched/scheduler.py:1871`` (``reset_prefix_cache``) and
  ``1917`` (``reset_connector_cache``) — already shipped in 0.20.1, the
  patch only provides the connector-specific ``reset_cache`` body.
- Sister rollout-correctness recipes: ``advance/rollout_corr.md``,
  ``advance/rollout_corr_math.md``.
