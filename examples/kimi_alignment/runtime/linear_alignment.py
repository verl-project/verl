"""Opt-in full-K BF16 projection retaining the existing sequence-shard output."""
import torch
import torch.nn.functional as F


def full_k_sequence_projection(op, input_parallel):
    weight = op.layer.weight
    assert input_parallel.ndim == weight.ndim == 2
    assert input_parallel.dtype == weight.dtype == torch.bfloat16
    assert input_parallel.shape[1] == weight.shape[1]
    assert op.layer.bias is None
    # Gather current runtime weights on every call: no stale cache after reload.
    full_input = op.comm_group.all_gather(input_parallel.contiguous(), dim=1)
    full_weight = op.comm_group.all_gather(weight.contiguous(), dim=1)
    pad = (-full_input.shape[0]) % op.tp_size
    if pad:
        full_input = F.pad(full_input, (0, 0, 0, pad))
    rows = full_input.shape[0] // op.tp_size
    local_input = full_input[op.tp_rank * rows:(op.tp_rank + 1) * rows].contiguous()
    return F.linear(local_input, full_weight.contiguous())
