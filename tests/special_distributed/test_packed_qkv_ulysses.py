# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed correctness tests for packed Ulysses QKV all-to-all.

Run on 2 or more GPUs:
    torchrun --standalone --nproc-per-node=2 -m pytest -svv \
        tests/special_distributed/test_packed_qkv_ulysses.py
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from verl.utils import ulysses
from verl.utils.ulysses import (
    gather_packed_qkv_seq_scatter_heads,
    gather_qkv_seq_scatter_heads,
    gather_seq_scatter_heads,
    set_ulysses_sequence_parallel_group,
)

pytestmark = pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="run with torchrun")

_FP8_DTYPE_NAMES = ("float8_e4m3fn", "float8_e5m2")
_FP8_DTYPES = [dtype for name in _FP8_DTYPE_NAMES if (dtype := getattr(torch, name, None)) is not None]
_TEST_DTYPES = [torch.float32, torch.bfloat16, *_FP8_DTYPES]
_TEST_DTYPE_IDS = [
    "fp32",
    "bf16",
    *[name.replace("float8_", "fp8_") for name in _FP8_DTYPE_NAMES if getattr(torch, name, None)],
]


@pytest.fixture(scope="module", autouse=True)
def _init_dist():
    assert torch.cuda.is_available(), "CUDA is required"
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert dist.get_world_size() >= 2, "need at least 2 GPUs"
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    try:
        yield
    finally:
        set_ulysses_sequence_parallel_group(None)
        if initialized_here:
            dist.destroy_process_group()


def _make_qkv(dtype: torch.dtype, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(1234 + rank)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    heads = max(8, 2 * world_size)
    heads = ((heads + world_size - 1) // world_size) * world_size
    shape = [2, 8, 8, 16]
    shape[head_dim] = heads

    def random(shape: list[int]) -> torch.Tensor:
        # torch.randn does not support FP8 directly. Creating FP32 samples then
        # casting mirrors how FP8 activation buffers are normally produced.
        return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)

    return random(shape), random(shape), random(shape)


@pytest.mark.parametrize("dtype", _TEST_DTYPES, ids=_TEST_DTYPE_IDS)
@pytest.mark.parametrize(("seq_dim", "head_dim"), [(2, 1), (1, 2)])
def test_packed_qkv_matches_three_all_to_all_in_forward_and_backward(dtype: torch.dtype, seq_dim: int, head_dim: int):
    q, k, v = _make_qkv(dtype, head_dim)
    legacy_inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    packed_inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]

    with patch.object(ulysses, "all_to_all_tensor", wraps=ulysses.all_to_all_tensor) as all_to_all:
        packed_outputs = gather_qkv_seq_scatter_heads(*packed_inputs, seq_dim=seq_dim, head_dim=head_dim)
        assert all_to_all.call_count == 1

    with patch.object(ulysses, "all_to_all_tensor", wraps=ulysses.all_to_all_tensor) as all_to_all:
        legacy_outputs = tuple(gather_seq_scatter_heads(x, seq_dim=seq_dim, head_dim=head_dim) for x in legacy_inputs)
        assert all_to_all.call_count == 3

    for packed, legacy in zip(packed_outputs, legacy_outputs, strict=True):
        torch.testing.assert_close(packed, legacy, atol=0, rtol=0)

    packed_loss = sum(x.float().square().mean() for x in packed_outputs)
    legacy_loss = sum(x.float().square().mean() for x in legacy_outputs)
    packed_loss.backward()
    legacy_loss.backward()

    for packed, legacy in zip(packed_inputs, legacy_inputs, strict=True):
        torch.testing.assert_close(packed.grad, legacy.grad, atol=0, rtol=0)


def test_prepacked_qkv_api_matches_three_all_to_all():
    q, k, v = _make_qkv(torch.float32, head_dim=1)
    packed = torch.cat((q, k, v), dim=0)

    with patch.object(ulysses, "all_to_all_tensor", wraps=ulysses.all_to_all_tensor) as all_to_all:
        outputs = gather_packed_qkv_seq_scatter_heads(
            packed,
            (q.size(0), k.size(0), v.size(0)),
            seq_dim=2,
            head_dim=1,
        )
        assert all_to_all.call_count == 1

    legacy_outputs = tuple(gather_seq_scatter_heads(x, seq_dim=2, head_dim=1) for x in (q, k, v))
    for actual, expected in zip(outputs, legacy_outputs, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
