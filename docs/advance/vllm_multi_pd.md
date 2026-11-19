# Single-node vLLM multi-prefill / multi-decode rollout

Last updated: 11/19/2026.

A PD replica contains `N` independent prefill engines and `M` independent decode engines. Every prefill can hand a request to any decode in its own replica. The complete group must fit on one physical node, with rollout DP=PP=1. The connectors remain GPU NIXL/Mooncake and Ascend MooncakeConnectorV1.

## Configuration and resource layout

One 16-device group, with two P and two D engines using TP4:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    tensor_model_parallel_size: 4
    data_parallel_size: 1
    pipeline_model_parallel_size: 1
    disable_log_stats: false
    disaggregation:
      enabled: true
      prefill_replicas: 2
      decode_replicas: 2
      decode_tensor_model_parallel_size: 4
      transfer_backend: mooncake
```

Supply the model, training configuration and resource pool as usual. The footprint is `N * P_TP + M * D_TP`. A null decode TP inherits prefill TP. Increasing engine counts does not provision additional devices automatically.

| Group-local worker ranks | Engine | TP ranks | Trainer IPC base rank |
| --- | --- | --- | --- |
| 0–3 | P0 | 0–3 | 0 |
| 4–7 | P1 | 0–3 | 4 |
| 8–11 | D0 | 0–3 | 8 |
| 12–15 | D1 | 0–3 | 12 |

`vLLMPDReplica.launch_servers()` slices workers in this order and creates distinct engine IDs and side-channel ports. It checks that the group's workers share a node. Ascend reserves only the P/D per-TP `kv_port` ranges, without an extra bootstrap range or bootstrap environment override; MooncakeConnectorV1 uses those KV handshake endpoints. GPU Mooncake keeps each P's bootstrap endpoint on that P's assigned port. Reserved ports still have a release-to-bind race window; this is not a cross-job port lease.

## Request path

```text
Agent / Gateway -> LLMServerClient -> existing global request router
                                      |                 |
                                     P0                P1
                                      \                 /
                                       -> D0 or D1 -> result
```

1. `RolloutReplica.get_request_server_endpoints()` returns the ordinary single entry; the PD override returns every P address/handle collected during launch. `LLMServerManager` registers these in the existing router. D engines are not direct generation entries. Obtain PD addresses through the endpoint methods; the inherited single `server_address` property is not populated for PD. The legacy `server_handle` property still refers to P0.
2. With the default global router, an existing routing key stays on its P while the affinity entry is valid; new keys choose a least-inflight entry. `full_determinism` or a configured router can change this policy. The inflight count covers the complete P-to-D call, not just prefill computation.
3. Each P receives all D peers in its group. It selects D, computes local prefill, and passes KV transfer metadata to D through `_pd_dispatch()`. The metadata identifies the originating P; the connector transfers the KV data. Results return through P and the client. The disposable one-token P leg ignores EOS without changing D's sampling parameters. An aborted P leg returns directly without attempting decode.
4. Each P selects D engines with its own round-robin cursor. This revision does not load a configurable D selector or maintain D reservation accounting. Multiple P engines can select the same D concurrently; load-aware, session-aware and cache-aware D scheduling are deferred to a separate change. The existing outer request router still selects P entries.

## Weight synchronization and lifecycle

`ServerAdapter` maps trainer ranks to `(replica, role, engine index, TP-local rank)`. Only TP rank 0 controls each engine's weight-update RPC; every rank retains its own weight payload endpoint. Receiving workers compute `VERL_ZMQ_BASE_TRAINER_RANK + local_rank`. Job ID and replica rank are also included in the IPC path, so matching senders and receivers address the same colocated worker without colliding with another group.

One-P configurations preserve existing actor names. Multi-P names include the P engine index; D actor names keep the existing convention. The bucketed weight-transfer protocol is unchanged.

Before `sleep()`, the trainer must quiesce new generation. The PD override waits for every P's drain call. With Mooncake (including Ascend's NIXL fallback), all D sleep calls complete before any P sleep begins; a D failure prevents P sleep. Native NIXL sleeps all engines after the P drains. Worker sleep owns memory unregistration; no extra connector-disconnect RPC is introduced. Resume keeps the existing barrier: resume every engine, then open admission gates.

This is not a complete handoff/transfer barrier: per-engine drain does not prove the absence of every P-to-D dispatch race. The inherited abort and independent `release_kv_cache()` paths have not been redesigned here. Cancellation, cache release, weight updates and transport lifetime still require target-device validation.

## Per-role tuning and metrics

These optional fields inherit the common rollout configuration by default:

```yaml
disaggregation:
  prefill_gpu_memory_utilization: null
  decode_gpu_memory_utilization: null
  prefill_engine_kwargs: {}
  decode_engine_kwargs: {}
```

For example, `prefill_engine_kwargs: {max_num_batched_tokens: 65536}` changes P's vLLM batch-token budget. Each role gets a deep copy of the common engine kwargs; a missing or null `engine_kwargs.vllm` is treated as an empty mapping. Non-null role overrides replace individual keys, including whole nested objects. Memory fractions must be in `(0, 1]`. An explicit `prefill_gpu_memory_utilization` or `decode_gpu_memory_utilization` takes precedence over memory fractions in both common and role engine kwargs, including their dashed spellings. Leaving the explicit role fraction null preserves the existing engine-kwargs precedence. Use the explicit topology fields for TP and engine counts; role overrides reject conflicting parallelism, worker-extension and connector configuration in both underscore and dashed spellings. Existing common `engine_kwargs` must also agree with the planned topology.

For a controlled PD/non-PD comparison, align batch-token budgets and memory fractions before attributing a throughput difference to separation. This feature does not claim a speedup for SWE or other workloads.

Automatic monitoring includes all P/D. Multi-P RL-Insight labels are `0-prefill0`, `0-prefill1`, `0-decode0`, and `0-decode1`, plus `pd_role` and `pd_index`. One-P groups keep `0-prefill`. See [RL-Insight setup](rl_insight.md) for enable flags and network prerequisites. Standalone Prometheus receives addresses; hardware exporters are configured separately.

## Validation scope

CPU contracts execute production Python methods with fake Ray boundaries: 1P3D, 2P2D, 3P1D, asymmetric TP, all-rank sender/receiver mapping across two groups, endpoint discovery, independent ports, role-config isolation and sleep ordering/failure. They do not start vLLM engines, transfer native KV, or establish GPU/Ascend correctness or performance. Complete an on-device smoke run, weight-update/sleep cycle and generation check with a compatible backend stack before relying on a deployment.
