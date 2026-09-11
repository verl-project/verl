# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Multi-GPU tensor-parallel correctness test for ``linear_topk_log_probs``.

Run with:
    torchrun --standalone --nnodes=1 --nproc-per-node=<tp-size> \
        tests/utils/test_special_linear_topk_log_probs_tp.py
"""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from verl.utils.kernel.linear_topk_log_probs import linear_topk_log_probs


def _run_case(device, rank, world_size, case_index, config):
    dtype, num_tokens, hidden_size, local_vocab_size, topk, temperature = config
    torch.manual_seed(9000 + case_index * 100 + rank)

    hidden_seed = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    dist.broadcast(hidden_seed, src=0)
    topk_ids = torch.randint(0, local_vocab_size * world_size, (num_tokens, topk), device=device)
    if rank == 0:
        # Touch both ends of every vocabulary shard. This catches off-by-one
        # errors in global-id ownership for TP2, TP3, and TP4.
        boundary_ids = []
        for shard in range(world_size):
            boundary_ids.extend([shard * local_vocab_size, (shard + 1) * local_vocab_size - 1])
        boundary_ids.extend(value for value in (1023, 1024) if value < local_vocab_size * world_size)
        boundary_ids = list(dict.fromkeys(boundary_ids))
        count = min(topk, len(boundary_ids))
        topk_ids[0, :count] = torch.tensor(boundary_ids[:count], device=device)
        if topk > 1 and num_tokens > 1:
            # Keep row 0 for shard-boundary coverage and use row 1 to verify
            # that duplicate selected ids accumulate their gradients.
            topk_ids[1, -1] = topk_ids[1, 0]
    dist.broadcast(topk_ids, src=0)

    local_weight = torch.randn(local_vocab_size, hidden_size, device=device, dtype=dtype)
    gathered_weights = [torch.empty_like(local_weight) for _ in range(world_size)]
    dist.all_gather(gathered_weights, local_weight)
    full_weight_seed = torch.cat(gathered_weights, dim=0)

    hidden_actual = hidden_seed.detach().clone().requires_grad_()
    weight_actual = local_weight.detach().clone().requires_grad_()
    # FP32 reference makes BF16 error attributable to kernel arithmetic rather
    # than to an additional low-precision reference operation.
    hidden_expected = hidden_seed.detach().float().requires_grad_()
    weight_expected = full_weight_seed.detach().float().requires_grad_()

    real_all_reduce = dist.all_reduce
    forward_collectives = []

    def record_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        forward_collectives.append((op, tuple(tensor.shape)))
        return real_all_reduce(tensor, op=op, group=group, async_op=async_op)

    with patch.object(dist, "all_reduce", side_effect=record_all_reduce):
        actual = linear_topk_log_probs(
            hidden_actual,
            weight_actual,
            topk_ids,
            temperature,
            dist.group.WORLD,
            chunk_size=5,
        )
    assert forward_collectives == [
        (dist.ReduceOp.MAX, (num_tokens,)),
        (dist.ReduceOp.SUM, (num_tokens, topk + 1)),
    ]
    logits = torch.mm(hidden_expected, weight_expected.T) / temperature
    expected = torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=topk_ids)
    if dtype == torch.float32:
        forward_tolerances = dict(atol=8e-4, rtol=8e-4)
        backward_tolerances = dict(atol=2e-3, rtol=2e-3)
    else:
        forward_tolerances = dict(atol=3e-2, rtol=3e-3)
        backward_tolerances = dict(atol=8e-2, rtol=1e-2)
    torch.testing.assert_close(actual, expected, **forward_tolerances)

    upstream = torch.randn_like(actual) if rank == 0 else torch.empty_like(actual)
    dist.broadcast(upstream, src=0)
    actual_hidden_grad, actual_weight_grad = torch.autograd.grad(
        actual,
        (hidden_actual, weight_actual),
        upstream,
    )
    expected_hidden_grad, expected_weight_grad = torch.autograd.grad(
        expected,
        (hidden_expected, weight_expected),
        upstream,
    )

    # The operator follows LinearCrossEntropy's contract: each TP rank returns
    # its local hidden-gradient contribution and the caller performs the sum.
    dist.all_reduce(actual_hidden_grad, op=dist.ReduceOp.SUM)
    expected_local_weight_grad = expected_weight_grad[rank * local_vocab_size : (rank + 1) * local_vocab_size]
    torch.testing.assert_close(actual_hidden_grad.float(), expected_hidden_grad, **backward_tolerances)
    torch.testing.assert_close(actual_weight_grad.float(), expected_local_weight_grad, **backward_tolerances)
    if rank == 0:
        print(
            f"[PASS] TP case {case_index}: dtype={dtype}, tokens={num_tokens}, "
            f"hidden={hidden_size}, local_vocab={local_vocab_size}, topk={topk}, T={temperature}"
        )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise RuntimeError(f"this test requires at least two ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cases = [
        (torch.float32, 1, 31, 1, 1, 0.5),
        (torch.float32, 23, 64, 37, 8, 0.81),
        (torch.float32, 33, 65, 1025, 8, 2.0),
        (torch.bfloat16, 17, 130, 1025, 16, 1.2),
    ]
    try:
        for case_index, config in enumerate(cases):
            _run_case(device, rank, world_size, case_index, config)
        if rank == 0:
            print(f"[PASS] TP{world_size}: all {len(cases)} forward/backward cases match the full-vocabulary reference")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
