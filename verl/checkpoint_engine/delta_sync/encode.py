# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""On-wire schema for delta sync: per-parameter manifest, flush container, checksum.

One layout: a uint8 positions blob (``indices`` encoding -- little-endian
absolute positions, 3 or 4 bytes / nnz) plus a parameter-dtype values tensor,
described by a per-parameter manifest. Values are sent verbatim in the
parameter's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices"]


def absolute_index_width(numel: int) -> int:
    """Return the narrowest supported width for an index into ``numel`` elements."""
    if numel < 0:
        raise ValueError(f"negative tensor size: {numel}")
    if numel <= (1 << 24):
        return 3
    if numel < (1 << 31):
        return 4
    raise ValueError(f"{numel} elements exceeds the int32 absolute-index encoding")


def pack_absolute_indices(indices: torch.Tensor, width: int) -> torch.Tensor:
    """Pack non-negative int32 absolute indices into a contiguous uint8 blob."""
    indices = indices.to(torch.int32).contiguous()
    raw = indices.view(torch.uint8).view(-1, 4)
    if width == 4:
        return raw.view(-1)
    if width == 3:
        return raw[:, :3].contiguous().view(-1)
    raise ValueError(f"unsupported absolute-index width: {width}")


def unpack_absolute_indices(packed: torch.Tensor, width: int) -> torch.Tensor:
    """Inverse of :func:`pack_absolute_indices`; return int32 indices."""
    if width not in (3, 4):
        raise ValueError(f"unsupported absolute-index width: {width}")
    if packed.numel() % width:
        raise ValueError(f"position blob length {packed.numel()} is not divisible by width {width}")
    if width == 4:
        # A 4-byte parameter may follow a 3-byte parameter in the shared blob,
        # leaving this otherwise-contiguous slice at an unaligned byte offset.
        # clone() both re-bases it and keeps the decode valid on CPU and CUDA.
        return packed.clone().view(torch.int32)
    if width == 3:
        raw = torch.zeros((packed.numel() // 3, 4), dtype=torch.uint8, device=packed.device)
        raw[:, :3] = packed.view(-1, 3)
        return raw.view(torch.int32).view(-1)
    raise AssertionError("unreachable")


# ---------- diff ----------------------------------------------------------


@dataclass
class DeltaParam:
    """Per-parameter manifest entry for a single chunk / bucket.

    Offsets are byte offsets into the surrounding ``__positions__`` blob and
    element offsets into the surrounding ``__values__`` tensor.
    """

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Uses ``torch.hash_tensor`` (XOR-reduce over uint64 bitcast); one reduction
    plus one ``.item()`` sync per argument.
    """
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


# ---------- encode --------------------------------------------------------


@dataclass
class DeltaFlush:
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. Despite the name it lives on
      the GPU in the sharded engine (the wire broadcasts from the GPU, so a
      host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        return self.values_gpu.numel()

    @property
    def wire_bytes(self) -> int:
        return self.positions_cpu.numel() + self.values_gpu.numel() * self.values_gpu.element_size()
