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
# Two's-complement int64 forms of murmur3 fmix64's odd constants; an affine
# position mix alone lets structured word pairs cancel in an XOR fold, and no
# avalanche fixes that completely (crafted two-bit differences fold into
# single-bit ones through the xor-shift stages). Hence the second, mod-2**64
# sum fold in ``_wire_digest``: a swap must cancel BOTH folds to collide.
# Every stage is bijective, so single-word corruption always changes the digest.
_AVALANCHE_MIX = -0x00AE502812AA7333  # 0xFF51AFD7ED558CCD
_AVALANCHE_MIX2 = -0x3B314601E57A13AD  # 0xC4CEB9FE1A85EC53
# torch.int64 ``>>`` is an arithmetic (sign-extending) shift; masking each
# shift to its low (64 - s) bits makes it a logical shift, which is also
# what the shift's invertibility argument assumes -- and identical even if a
# backend implements >> as logical.
_SHIFT33_MASK = (1 << 31) - 1
_SHIFT29_MASK = (1 << 35) - 1
_SHIFT32_MASK = (1 << 32) - 1


def _wire_digest(t: torch.Tensor) -> int:
    """Order-sensitive 128-bit fold of a tensor's raw bytes (bounded chunks).

    Replaces ``torch.hash_tensor`` (PyTorch >= 2.9, CPU/CUDA kernels only -- no
    NPU kernel). Each little-endian 64-bit word is mixed with its position and
    avalanched (see the constants), then accumulated by two independent folds
    of the same mixed values: a halving XOR tree and a mod-2**64 sum. Every
    mixing stage is bijective, so single-word corruption is always detected; a
    word swap collides only if the pair's mixed XOR-difference AND arithmetic
    difference are both constant across the swapped positions, and no
    construction is known to achieve both (every single-fold family found so
    far is non-constant in the other fold; all are pinned as regressions), so
    unstructured collisions are ~2**-128. Integer addition is associative, so
    the sum fold is independent of reduction order and device. Words are
    processed in ``_FOLD_CHUNK_BYTES`` chunks to bound transient memory; the
    fold never mutates its argument, so sender and receiver agree on any
    device.
    """
    if t.numel() == 0:
        return 0
    flat = t.contiguous().reshape(-1)
    b = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    n8 = b.numel() // 8 * 8
    acc = torch.zeros(1, dtype=torch.int64, device=b.device)
    acc_sum = torch.zeros(1, dtype=torch.int64, device=b.device)
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
        v = (v ^ ((v >> 33) & _SHIFT33_MASK)) * _AVALANCHE_MIX
        v = (v ^ ((v >> 29) & _SHIFT29_MASK)) * _AVALANCHE_MIX2
        v = v ^ ((v >> 32) & _SHIFT32_MASK)
        # Sum fold: wraparound int64 == mod 2**64, associative, no per-chunk sync.
        acc_sum += v.sum()
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
    s_xor = (int(acc.item()) ^ tail) & ((1 << 64) - 1)
    s_sum = (int(acc_sum.item()) + tail) & ((1 << 64) - 1)
    return (s_sum << 64) | s_xor


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Order-sensitive 128-bit fold (see ``_wire_digest``) over each argument's
    bytes, computed on device; the same function runs on both ends, so the
    value is stable across dtypes and devices. The values digest is rotated
    one bit so the two inputs cannot cancel each other.
    """
    p = _wire_digest(positions)
    v = _wire_digest(values)
    v_rot = ((v << 1) | (v >> 127)) & ((1 << 128) - 1)
    return (p ^ v_rot) & ((1 << 128) - 1)


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
