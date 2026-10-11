# Recover a standalone vLLM replica and interrupted requests

This example runs a real optimizer update, kills a rollout EngineCore during two
native verl client calls, then restores their generation through
`next_weight_sync`. It creates a tiny Qwen2 model locally and needs no dataset or
model download.

Run from the verl repository root on one node with three available CUDA GPUs:

```bash
uv run --extra vllm --extra cupy-cu130 python examples/fault_tolerance/restart_vllm_replica.py
```

One GPU holds a small HF/AdamW trainer. Two GPUs hold a standalone vLLM MP
replica with TP=2; its checkpoint-engine receivers share those two rollout GPUs.
The `cupy-cu130` extra supplies the NCCL transport dependency. Prepare the
environment once to avoid repeating cold dependency installation; vLLM and
CUDA/driver requirements come from verl's dependency configuration.

## What happens

1. Publish version 0, perform one AdamW update, and publish version 1 through
   verl's normal NCCL `update_weights` path.
2. Start two concurrent calls through the native load balancer: an
   `LLMServerClient` call and a `FullyAsyncLLMServerClient` call.
3. Wait until both real vLLM streams have emitted tokens. Verify that neither
   has finished and both are still present in vLLM's live request state, then
   send `SIGKILL` only to this server's owned EngineCore.
4. The surviving server actor returns each last observed cumulative prefix
   with an engine-failure marker. Both original client calls remain pending.
   Remove the failed handle from routing and verify `EngineDeadError` through
   the native health check.
5. Call `restart_replica(replica)`. The replacement remains paused while the
   original checkpoint receivers and resource pool are retained and rebound.
   Run the normal `update_weights(1)` again, compare both MP ranks' weight
   fingerprints, then register the new handle and HTTP address.
6. The ordinary client replays the complete turn. The fully async client
   appends its saved tokens to the prompt and requests only the remaining
   budget. Verify both outputs have 64 tokens and finite log probabilities,
   version 1, and the same tokens as an uninterrupted greedy reference. For
   continuation, verify the saved tokens and their original log probabilities
   are preserved exactly.
7. Finalize all owned NCCL ranks together, stop the server, and release this
   example's actors and placement group.

The application-side recovery calls are:

```python
client = FullyAsyncLLMServerClient(config, load_balancer, engine_recovery_timeout=300)

await load_balancer.remove_server_if_current.remote(old_address, old_handle)
await checkpoint_manager.restart_replica(replica)
await checkpoint_manager.update_weights(global_steps=1)
await load_balancer.add_servers.remote({replica.server_address: replica.server_handle})
```

Client replay is opt-in and has a timeout and retry limit. The recovery owner
must choose a trainer-safe synchronization boundary and publish the replacement
route only after weights are ready. This example's trainer is idle during
recovery, and its next normal synchronization reuses version 1 without another
optimizer update. It adds no CPU weight backup or new weight transport.

Progress is token and log-probability progress returned by the surviving server
actor. The replacement performs a new prefill over prompt plus saved tokens;
old engine KV state is not migrated. If the whole server actor dies, its
unreturned token prefix is unavailable, so the failed segment must be replayed.

## Scope and validation

This demonstrates one hard EngineCore failure, a single-node standalone MP
replica, full named-tensor NCCL updates, and a small synthetic workload.
Colocated restart, PD, unmerged LoRA and multi-node recovery are outside its
scope. It explicitly invokes recovery after its injected fault; it is not a
continuous failure detector or a PPO/model-quality experiment.

Health checks also work with `disable_log_stats=True`; scheduler metric
snapshots are separate from recovery. NCCL uses `rebuild_group=True`, so each
normal synchronization finalizes its transport group through the native
lifecycle. Final cleanup checks which groups remain initialized and dispatches
all their finalizers concurrently before releasing owned resources.

The test-only helpers inspect vLLM's EngineManager, MP weights and active request
state, and wrap its real generation stream to inject the fault. These private
interfaces are confined to the example; generation and replay use the native
production clients, server and checkpoint manager.

The script writes `result.json` even on failure. Successful receipts include
real request progress and active internal IDs at injection, the killed Core's
PID and creation time, old/new actor and Core identities, per-rank pre-fault and
restored weight digests, both interrupted/replayed request inputs and outputs,
version ranges, lifecycle timestamps, and successful owned-resource cleanup.
Use `--output-dir /path/to/new-directory` to retain the tiny model and receipts.
