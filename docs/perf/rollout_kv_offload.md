# Rollout KV Cache Offload via Mooncake-Store

Last updated: 11/09/2026.

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

This connector-based prefix cache is separate from the opt-in
[colocated partial KV CPU preservation](../advance/v1_async_trainer.md#experimental-colocated-partial-kv-offload).
That experimental path intentionally retains stale in-flight KV across updates,
backs up complete native allocation pools, and rejects external KV connectors.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.
