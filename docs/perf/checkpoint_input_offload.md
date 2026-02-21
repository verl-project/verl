# Checkpoint Input CPU Offload

Offload gradient checkpoint saved tensors (checkpoint inputs) to CPU pinned memory during forward, transfer back to GPU during backward. Reduces GPU memory usage with near-zero forward overhead.

## How It Works

PyTorch's gradient checkpointing saves each layer's **input tensors** for backward recomputation. For large models (e.g., Qwen2.5-VL-32B with 64 decoder layers + 32 vision blocks), these checkpoint inputs consume significant GPU memory proportional to batch/sequence size.

This feature exploits PyTorch's `saved_tensors_hooks` "innermost wins" nesting:

1. **Forward (outside `_checkpoint_hook`)**: Our outer hooks are innermost -> `_pack` intercepts checkpoint inputs -> async D2H to CPU pinned memory via a dedicated CUDA stream.
2. **Forward (inside `_checkpoint_hook`)**: Inner hooks become innermost -> our hooks are **inactive** -> intermediates handled normally by `_checkpoint_hook`.
3. **Backward recomputation**: `_recomputation_hook` becomes innermost -> our hooks are **inactive** -> recomputed tensors stay on GPU.
4. **Backward access**: Captured `_unpack` fires -> sync H2D transfer from CPU back to GPU.

Only checkpoint inputs and non-checkpointed layer tensors (embedding, lm_head) are offloaded. Parameters, small tensors, and non-CUDA tensors are skipped.

## Configuration

```bash
# Enable in training script
actor_rollout_ref.actor.fsdp_config.checkpoint_input_offload=true
```

In `fsdp.yaml`:
```yaml
checkpoint_input_offload: false  # default off
```

**Incompatible with**: `use_prefix_grouper` (PrefixGrouper bypasses the offload context manager). A build-time `ValueError` is raised if both are enabled.

## Benchmark Results

**Setup**: Qwen2.5-VL-32B, 8x H100 80GB, FSDP2, SP_SIZE=2, `ppo_max_token_len_per_gpu=45000`

| Metric | Value |
|--------|-------|
| Forward memory delta (with offload) | +0.31 ~ +0.64 GB |
| Tensors offloaded per micro-batch | 115 (pack=115, unpack=115) |
| Data offloaded per step | 24 ~ 44 GB |

Without checkpoint input offload, forward delta would be ~9 GB instead of ~0.5 GB, potentially causing OOM.

## Implementation Details

### Files Modified

| File | Change |
|------|--------|
| `verl/utils/checkpoint_offload.py` | `CheckpointInputOffload` class (pack/unpack hooks, async D2H stream) |
| `verl/workers/fsdp_workers.py` | Build-time initialization + PrefixGrouper conflict check |
| `verl/workers/actor/dp_actor.py` | Wrap `model()` forward call with offload context manager |
| `verl/workers/config/engine.py` | Config field: `checkpoint_input_offload` |
| `verl/trainer/config/engine/fsdp.yaml` | Default config entry |

### CUDA Stream Synchronization

The D2H copy uses a dedicated CUDA stream for async overlap. A critical `wait_stream` call ensures the D2H stream sees completed data from the default compute stream:

```python
# Without this: d2h_stream may read stale GPU data (race condition)
self.d2h_stream.wait_stream(torch.cuda.current_stream(tensor.device))
with torch.cuda.stream(self.d2h_stream):
    cpu_tensor.copy_(tensor, non_blocking=True)
tensor.record_stream(self.d2h_stream)
```

This adds ~3us per tensor (~0.35ms total for 115 tensors) -- negligible vs forward compute.

## Training Script Reference

Key flags:

```bash
actor_rollout_ref.actor.fsdp_config.checkpoint_input_offload=true
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
```
