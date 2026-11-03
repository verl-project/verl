# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Slot layout and transport helpers shared by the weight publisher and puller."""

from typing import Any, Callable

import torch

from .registry import BucketDesc, TensorSpec

# Every tensor starts at a multiple of this, so any dtype view of the slot is aligned.
TENSOR_ALIGNMENT = 256

TransportFactory = Callable[[str, str], Any]
"""(backend, device) -> torch.distributed._transport.Transport"""


def default_transport_factory(backend: str, device: str):
    # torch.distributed._transport ships with torch >= 2.15; import lazily so the rest of verl
    # does not depend on it.
    from torch.distributed._transport import new_transport

    return new_transport(backend, device)


def plan_layout(named_tensors: list[tuple[str, torch.Size, torch.dtype]]) -> tuple[list[TensorSpec], int]:
    """Place tensors back to back in one slot buffer.

    Returns:
        The tensor specs with offsets, and the slot size in bytes.
    """
    specs, offset = [], 0
    for name, shape, dtype in named_tensors:
        offset = (offset + TENSOR_ALIGNMENT - 1) // TENSOR_ALIGNMENT * TENSOR_ALIGNMENT
        specs.append(TensorSpec(name=name, shape=torch.Size(shape), dtype=dtype, offset=offset))
        offset += torch.Size(shape).numel() * dtype.itemsize
    return specs, offset


def bucket_ranges(total_bytes: int, bucket_size: int) -> list[tuple[int, int]]:
    """Tile [0, total_bytes) with (offset, size) ranges of at most bucket_size bytes."""
    return [(offset, min(bucket_size, total_bytes - offset)) for offset in range(0, total_bytes, bucket_size)]


def tensor_view(buffer: torch.Tensor, spec: TensorSpec) -> torch.Tensor:
    """View of a tensor stored in a uint8 slot buffer."""
    nbytes = spec.shape.numel() * spec.dtype.itemsize
    return buffer[spec.offset : spec.offset + nbytes].view(spec.dtype).view(spec.shape)


def same_layout(buckets: list[BucketDesc], ranges: list[tuple[int, int]]) -> bool:
    return [(b.offset, b.size) for b in buckets] == ranges
