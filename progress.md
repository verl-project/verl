# Multi-Tenant LoRA Training — Implementation Progress

## Status: Core implementation complete, needs testing

## New Files Created

| File | Purpose |
|------|---------|
| `multi_tenant_run.sh` | Shell script for multi-tenant launch (mirrors `dapo_7b_math_vllm_2_2_lora_run.sh`) |
| `multi_tenant_run.sbatch` | SLURM batch script (mirrors `dapo_7b_vllm_lora_2_2_ra32_st0_ts1.sbatch`) |
| `verl/experimental/fully_async_policy/multi_tenant_main.py` | Entry point — parses tenants, creates per-tenant queues, orchestrates components |
| `verl/experimental/fully_async_policy/multi_tenant_rollouter.py` | Extends `FullyAsyncRollouter` — per-tenant dataloaders, interleaved sample feeding, per-tenant queue routing |
| `verl/experimental/fully_async_policy/multi_tenant_trainer.py` | Extends `FullyAsyncTrainer` — polls per-tenant queues, swaps LoRA adapters, per-tenant weight sync to vLLM |

## Existing Files Modified

| File | Change |
|------|--------|
| `verl/experimental/fully_async_policy/detach_utils.py` | Added `TenantConfig` dataclass, `parse_tenants()` function, `tenant_id` field to `RolloutSample` |
| `verl/experimental/separation/engine_workers.py` | Added `get_lora_adapter_weights()` method to `DetachActorWorker` for extracting LoRA state dict |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | (1) Added `TensorLoRARequest` import; (2) Added `add_tenant_lora()` method to `vLLMHttpServer` for loading per-tenant adapters; (3) Modified `generate()` to support `_lora_int_id` in sampling_params for multi-LoRA routing |
| `verl/experimental/agent_loop/agent_loop.py` | Added `_lora_int_id` passthrough from DataProto `meta_info` to `sampling_params` in `AgentLoopWorker.generate_sequences()` |

## Architecture

```
                    ┌─────────────────────┐
                    │  MultiTenantMain    │
                    │  (parse tenants,    │
                    │   create queues)    │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────────┐
     │  Tenant A  │  │   Tenant B   │  │   Tenant N   │
     │  Queue     │  │   Queue      │  │   Queue      │
     └─────┬──────┘  └──────┬───────┘  └──────┬───────┘
           │                │                  │
     ┌─────┴────────────────┴──────────────────┴─────┐
     │              MultiTenantRollouter             │
     │  - Per-tenant dataloaders (round-robin)       │
     │  - Tags samples with tenant_id + lora_int_id  │
     │  - Routes to correct tenant queue             │
     │  - vLLM generates with tenant's LoRA adapter  │
     └───────────────────────────────────────────────┘
           │                │                  │
     ┌─────┴────────────────┴──────────────────┴─────┐
     │              MultiTenantTrainer               │
     │  - Polls all tenant queues                    │
     │  - Trains whichever tenant is ready first     │
     │  - Swaps LoRA adapter via save/restore CPU    │
     │  - Syncs updated adapter to vLLM directly     │
     └───────────────────────────────────────────────┘
```

## Key Design Decisions

### Tenant adapter swapping (training side)
Uses existing `save_model_to_cpu` / `restore_model_from_cpu` from `DetachActorWorker`. Each tenant gets a unique version key (`100000 + tenant_index`). The full FSDP2 sharded state is saved/restored per worker — efficient because it's per-shard, not full-gather.

### LoRA adapter sync to vLLM
Bypasses the NCCL checkpoint engine entirely. Instead:
1. Extracts LoRA adapter weights via `get_lora_adapter_weights()` (new method on `DetachActorWorker`)
2. Sends directly to vLLM server handles via Ray RPC `add_tenant_lora()`
3. vLLM creates `TensorLoRARequest` with per-tenant `lora_int_id`

This is efficient because LoRA adapters are tiny (~50MB for rank-32 on 7B).

### Multi-LoRA generation
`_lora_int_id` is injected into `DataProto.meta_info` by the rollouter, flows through `AgentLoopWorker` into `sampling_params`, and is popped by the vLLM server to create the correct `LoRARequest`.

## TODOs / Known Limitations

- [ ] **Testing**: No tests written yet — needs end-to-end test on cluster
- [ ] **Checkpoint resume**: Multi-tenant checkpoint loading not implemented (only saving)
- [ ] **Validation**: Per-tenant validation not implemented (currently uses same validation as single-tenant)
- [ ] **Per-tenant hyperparameters**: Currently all tenants share the same config
- [ ] **Dynamic tenants**: Tenants are fixed at launch; no join/leave support
- [ ] **vLLM `add_lora` via `collective_rpc`**: The `add_tenant_lora` method uses `collective_rpc("add_lora", ...)` which may need testing with vLLM v1 — the API for dynamic LoRA loading may differ
