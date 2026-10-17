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

One layout: a uint8 positions blob (``indices`` encoding -- int32 absolute
positions, 4 bytes / nnz) plus a parameter-dtype values tensor, described by a
per-parameter manifest. Values are sent verbatim in the parameter's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices"]


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


_FOLD_CHUNK_BYTES = 64 << 20  # transient-memory bound: bytes folded per chunk
# Two's-complement int64 form of 0x9E3779B97F4A7C15 (odd): mixes a word with
# its position so the fold depends on byte order, not just the byte multiset.
_POS_MIX = -0x61C8864680B583EB


def _wire_digest(t: torch.Tensor) -> int:
    """Order-sensitive 64-bit fold of a tensor's raw bytes (bounded chunks).

    Replaces ``torch.hash_tensor`` (PyTorch >= 2.9, CPU/CUDA kernels only -- no
    NPU kernel, and its torch_npu CPU fallback costs a full D2H per call).
    Every little-endian 64-bit word is mixed with its position
    (``w + _POS_MIX * (index + 1)``, wraparound int64 == mod 2**64) before a
    halving XOR fold: a changed word always changes its mixed value, so any
    single-word corruption survives the fold, and swapping two different words
    swaps their mixed values (the odd multiplier keeps position terms distinct),
    which the fold reports unless the XORs coincide. Words are processed in
    ``_FOLD_CHUNK_BYTES`` chunks so transient memory stays bounded on a
    3072 MB flush. The fold never mutates the argument's storage and is a pure
    function of the bytes, so sender and receiver agree on any device.
    """
    if t.numel() == 0:
        return 0
    flat = t.contiguous().reshape(-1)
    b = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    n8 = b.numel() // 8 * 8
    acc = torch.zeros(1, dtype=torch.int64, device=b.device)
    base = 0
    for start in range(0, n8, _FOLD_CHUNK_BYTES):
        chunk = b[start : min(start + _FOLD_CHUNK_BYTES, n8)]
        if chunk.storage_offset() % 8:
            # A contiguous slice can keep a storage offset that the int64
            # view below rejects; re-house such chunks in fresh storage.
            chunk = chunk.clone()
        w = chunk.view(torch.int64)
        idx = torch.arange(base + 1, base + w.numel() + 1, dtype=torch.int64, device=b.device)
        v = w + idx * _POS_MIX
        # Halving XOR fold of the mixed words; an odd middle element carries.
        while v.numel() > 1:
            n = v.numel()
            half = n // 2
            if n % 2 == 0:
                v = v[:half] ^ v[half:]
            else:
                v = torch.cat((v[:half] ^ v[half + 1 :], v[half : half + 1]))
        acc ^= v
        base += w.numel()
    tail = 0
    for i, x in enumerate(b[n8:].tolist()):
        tail ^= x << (8 * i)
    return (int(acc.item()) ^ tail) & ((1 << 64) - 1)


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Order-sensitive 64-bit fold (see ``_wire_digest``) over each argument's
    bytes, computed on device; the same function runs on both ends, so the
    value is stable across dtypes and devices. The values digest is rotated
    one bit so the two inputs cannot cancel each other.
    """
    p = _wire_digest(positions)
    v = _wire_digest(values)
    v_rot = ((v << 1) | (v >> 63)) & ((1 << 64) - 1)
    return (p ^ v_rot) & ((1 << 64) - 1)


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
