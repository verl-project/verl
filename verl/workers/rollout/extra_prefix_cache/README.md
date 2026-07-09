# Extra Prefix Cache

Extra Prefix Cache (EPC) is a rollout-side helper for reusing a stable prompt
prefix through an external KV cache. It is intended for agent workloads where
many rollout requests share the same system/tool prefix while the user/task
content changes.

## What It Does

EPC prepares two requests for a shared prefix:

- A warmup request that writes only the stable prefix into the external KV cache.
- A rollout request that is allowed to read that prefix but is not allowed to
  write its dynamic prompt/response tokens back into the prefix cache.

The controller receives prefix metadata from the caller, computes a cache salt,
aligns the stored token range to the backend chunk size, deduplicates warmup by
cache salt, and tags backend request ids with per-request read/write policy.

The important layer here is the verl-side policy and prefix protocol. A generic
external KV cache stores and retrieves blocks after the rollout backend asks it
to do so; EPC decides which part of a request is a stable prefix, which request
is allowed to populate that prefix, which rollout requests may read it, and when
the namespace should move forward after model updates. This keeps agent-specific
knowledge, such as the system/tool prefix boundary, outside the KV server.

Model-weight invalidation is handled by the runtime cache epoch. When
`advance_epoch_on_weight_update=True`, the epoch can advance after weight
updates so requests use a new cache namespace instead of reusing old KV from a
previous model version.

## Prefix Metadata

The default provider is `explicit`. It expects request metadata containing:

- `stable_prefix_token_len`, also accepted as `system_prefix_len`.
- `stable_prefix_fingerprint`, also accepted as `system_prefix_fingerprint`.

In the uni-agent integration, this metadata is produced by the agent model code
and passed to verl as `extra_prefix_cache_metadata`. The stable prefix is the
part of the tokenized prompt that remains shared when the task-specific user
content changes.

The optional `heuristic` provider still requires `stable_prefix_token_len`, but
computes the fingerprint from `prompt_ids[:stable_prefix_token_len]` together
with tokenizer/template fingerprints. If no usable metadata is supplied, EPC
skips the request instead of guessing a prefix boundary.

## How To Enable

EPC is disabled by default. Enable it under the rollout config:

```bash
+actor_rollout_ref.rollout.extra_prefix_cache.enable=True
+actor_rollout_ref.rollout.extra_prefix_cache.backend=lmcache
+actor_rollout_ref.rollout.extra_prefix_cache.scope=system_prefix
+actor_rollout_ref.rollout.extra_prefix_cache.read_policy=rollout
+actor_rollout_ref.rollout.extra_prefix_cache.write_policy=warmup_only
+actor_rollout_ref.rollout.extra_prefix_cache.namespace=uni-agent-system-prefix
+actor_rollout_ref.rollout.extra_prefix_cache.model_cache_epoch=epoch0
+actor_rollout_ref.rollout.extra_prefix_cache.advance_epoch_on_weight_update=False
+actor_rollout_ref.rollout.extra_prefix_cache.chunk_size=64
```

For the verified vLLM + LMCache path, also configure vLLM KV transfer to use
the EPC-aware LMCache connector:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager=True
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config="{kv_connector:VerlLMCacheExtraPrefixConnector,kv_connector_module_path:verl.workers.rollout.extra_prefix_cache.lmcache_connector,kv_role:kv_both,kv_connector_extra_config:{lmcache.mp.host:tcp://127.0.0.1,lmcache.mp.port:5555,lmcache.mp.mp_transfer_mode:engine_driven}}"
```

The training examples wire these settings in
`examples/agent_train/v1/single_node_v1_collocate_async_lmcache.sh`.

## Current Adapter

The confirmed working integration is:

- Rollout backend: vLLM.
- External KV server: LMCache MP server.
- Connector:
  `verl.workers.rollout.extra_prefix_cache.lmcache_connector.VerlLMCacheExtraPrefixConnector`.

vLLM carries namespace isolation through `cache_salt`. The connector reads the
EPC read/write policy from the request id prefix and enforces it before LMCache
loads or stores KV blocks. This lets the integration reuse existing vLLM and
LMCache extension points without modifying vLLM or LMCache source code.

The LMCache server remains responsible for block storage and transfer. The EPC
module does not replace that backend; it supplies verl-specific request tagging,
warmup orchestration, prefix scoping, and model-epoch invalidation around it.

## Adapting Other Backends

The EPC controller is backend-lightweight. A new rollout backend or KV server
can reuse the same idea if it can accept:

- A per-request cache namespace or salt.
- A request id or equivalent metadata channel carrying read/write policy.
- A way to warm up/store a prefix-only request.
- A way to prevent rollout requests from storing dynamic suffix tokens.

This should make other rollout backends that support external KV cache, and
other independent KV cache servers, quick to connect. Those integrations are
not yet confirmed and should be validated with functional reuse and epoch
invalidation tests before being treated as supported.
