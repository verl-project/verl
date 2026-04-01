---
name: debug-distributed
description: Guide for debugging distributed training issues in veRL. Use when encountering hangs, OOM, wrong results, or Ray errors.
---

# Debug Distributed Training

Diagnose and fix distributed training issues in veRL.

## When to Use

This skill is triggered when:

- Training hangs or deadlocks
- OOM (out of memory) errors
- Wrong/NaN loss or reward values
- Ray worker crashes or connection errors
- NCCL / collective communication errors
- HybridEngine resharding failures

---

## Category 1: Hangs / Deadlocks

### Symptoms
- Training stuck with no output, no error
- `ray.get()` never returns

### Checklist

1. **NCCL barrier mismatch** — all ranks must call the same collectives in the same
   order. Add rank-0 logging before each collective to confirm all ranks reach it.

2. **Ray actor deadlock** — check if a remote call is waiting on a result that itself
   waits on another remote call (circular dependency).

3. **Rollout never returns** — vLLM/SGLang engine stuck. Check inference worker logs:
   ```bash
   ray logs actor --id <actor_id>
   ```

4. **Uneven data across ranks** — padding or sequence balancing issue causes one rank
   to have 0 samples. Check `seqlen_balancing.py`.

5. **Enable NCCL debug**:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

---

## Category 2: OOM (Out of Memory)

### Common Causes & Fixes

| Cause | Fix |
|-------|-----|
| Too large `max_response_length` | Reduce `data.max_response_length` |
| vLLM `gpu_memory_utilization` too high | Lower `actor_rollout_ref.rollout.gpu_memory_utilization` (e.g., 0.4) |
| Gradient accumulation accumulates too much | Reduce `actor.ppo_mini_batch_size` |
| HybridEngine holds two copies during resharding | Use `offload_params=True` |
| Sequence packing increases effective batch size | Lower `data.train_batch_size` |

### Quick Debug

```python
# Add to your trainer to track memory
import torch
print(f"[Rank {rank}] GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
      f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
```

---

## Category 3: NaN / Wrong Loss or Reward

### Checklist

1. **Reward always 0** — `data_source` mismatch. Verify:
   ```python
   # Print data_source values in your dataset
   print(batch.non_tensor_batch["data_source"])
   ```

2. **NaN in loss** — often caused by:
   - Log prob of 0-probability tokens (numerical underflow)
   - `response_mask` is all zeros for some samples
   - Advantage normalization dividing by zero std

   ```python
   # Check advantage stats
   adv = data.batch["advantages"]
   mask = data.batch["response_mask"]
   print(f"adv mean={adv[mask.bool()].mean():.4f}, std={adv[mask.bool()].std():.4f}")
   ```

3. **Importance ratio explodes** — old_log_probs and new log_probs misaligned.
   Check that attention_mask and position_ids match between rollout and training.

4. **Reward NaN** — exception in `compute_score` returning `float('nan')`.
   Add explicit exception handling returning `0.0`.

5. **ref_log_prob mismatch** — reference model not loaded correctly. Verify with:
   ```bash
   actor_rollout_ref.ref.log=True  # Enable ref logprob logging
   ```

---

## Category 4: Ray Worker Crashes

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RayActorError` | Worker process died | Check worker logs for root cause |
| `OutOfMemoryError` in Ray | Ray object store full | Increase `object_store_memory` in `runtime_env.yaml` |
| `ActorDiedError` during resharding | HybridEngine crash | Check NCCL version compatibility |
| `ConnectionRefusedError` | Head node not reachable | Check `ray start --head` is running |

### Get Worker Logs

```bash
# List all actors
ray list actors

# Get logs for a specific actor
ray logs actor --id <actor_id>

# Stream logs in real time
ray logs actor --id <actor_id> --follow
```

---

## Category 5: HybridEngine Resharding Issues

veRL's HybridEngine transitions weights between training (FSDP/Megatron) and inference
(vLLM/SGLang) sharding. Failures here are usually:

1. **TP size mismatch** — training TP and rollout TP must be compatible.
   Check `actor_rollout_ref.actor.model_parallel_size` vs
   `actor_rollout_ref.rollout.tensor_model_parallel_size`.

2. **Weight name mismatch after conversion** — add logging in the sharding manager:
   ```python
   # verl/workers/sharding_manager/
   # Add: print(list(state_dict.keys())[:10])
   ```

3. **CUDA device mismatch** — ensure training and rollout workers are on the same GPU
   set when using collocated placement.

---

## General Debug Tips

```python
# 1. Isolate to single GPU first
trainer.n_gpus_per_node=1 trainer.nnodes=1

# 2. Reduce batch size to minimum
data.train_batch_size=4

# 3. Enable verbose Ray logging
import logging
logging.getLogger("ray").setLevel(logging.DEBUG)

# 4. Use veRL's built-in profiler
trainer.profile=True

# 5. Print DataProto shapes at key points
for k, v in data.batch.items():
    print(f"  {k}: {v.shape} {v.dtype}")
```

## Key Log Locations

| Component | Log location |
|-----------|-------------|
| Ray workers | `ray logs actor --id <id>` |
| vLLM engine | Actor log, search `[vLLM]` prefix |
| SGLang engine | Actor log, search `[SGLang]` prefix |
| NCCL | `NCCL_DEBUG=INFO` output |
| Training metrics | WandB / stdout based on `trainer.logger` |

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/debug-distributed/SKILL.md

## How to Update
- When new common error patterns emerge: add to the relevant category
- When Ray API changes: update log commands
================================================================================
-->
