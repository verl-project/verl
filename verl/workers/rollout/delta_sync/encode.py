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
"""Position+value encoding and bucketing for delta sync.

Three encodings share one on-wire layout (a uint8 positions blob plus a
parameter-dtype values tensor with a per-parameter manifest); decoders
dispatch on metadata.

* ``indices``     -- int32 absolute positions (4 bytes / nnz)
* ``deltas``      -- uint16 gap-deltas with uint32 per-parameter fallback
* ``deltas_zstd`` -- ``deltas`` wrapped in zstd at the file layer (the
                     compression is applied at safetensors write time;
                     this module produces the uncompressed gap stream)

Values are sent verbatim in the parameter's dtype regardless of encoding.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch

from .delta_state import ParamDiff

DeltaEncodingName = Literal["indices", "deltas", "deltas_zstd"]


# ---------- diff ----------------------------------------------------------


def bytewise_diff_mask(current: torch.Tensor, snapshot: torch.Tensor) -> torch.Tensor:
    """Per-element bool mask: True where ``current`` and ``snapshot`` differ.

    Dtype-agnostic via view-as-integer; no arithmetic.
    """
    es = current.element_size()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    return current.view(int_dtype) != snapshot.view(int_dtype)


# ---------- wire format ---------------------------------------------------


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


@dataclass
class EncodedChunk:
    """One HF chunk after position+value encoding, before bucket merging."""

    pos_bytes: bytes
    val_tensor: torch.Tensor
    params: list[DeltaParam]
    nnz: int

    @classmethod
    def empty(cls) -> "EncodedChunk":
        return cls(
            pos_bytes=b"",
            val_tensor=torch.empty(0, dtype=torch.bfloat16),
            params=[],
            nnz=0,
        )


# ---------- checksum ------------------------------------------------------


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Uses ``torch.hash_tensor`` (XOR-reduce over uint64 bitcast); one reduction
    plus one ``.item()`` sync per argument.
    """
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


# ---------- encode --------------------------------------------------------


def _sparse_boundaries(
    diffs: list[ParamDiff],
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[int]]:
    """One concat -> one nonzero -> one searchsorted -> one tolist().

    Collapses per-parameter host syncs to a single one per chunk.
    """
    device = diffs[0].values.device
    sizes = [d.values.numel() for d in diffs]
    cum = list(itertools.accumulate(sizes))
    cum_t = torch.tensor(cum, dtype=torch.int64, device=device)

    big_values = torch.cat([d.values.contiguous().view(-1) for d in diffs], dim=0)
    big_mask = torch.cat([d.mask.contiguous().view(-1) for d in diffs], dim=0)
    big_idx = big_mask.nonzero(as_tuple=False).view(-1)
    big_val = big_values[big_idx]
    bounds = torch.searchsorted(big_idx, cum_t).tolist()
    return big_val, bounds, big_idx, cum


def _encode_indices(diffs: list[ParamDiff]) -> EncodedChunk:
    if not diffs:
        return EncodedChunk.empty()
    big_val, bounds, big_idx, cum = _sparse_boundaries(diffs)
    pos_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    params: list[DeltaParam] = []
    pos_byte_off = val_off = 0
    prev_b = 0
    prev_param_start = 0
    for i, d in enumerate(diffs):
        b = bounds[i]
        nnz = b - prev_b
        if nnz > 0:
            local_idx = (big_idx[prev_b:b] - prev_param_start).to(torch.int32)
            pos_pieces.append(local_idx)
            val_pieces.append(big_val[prev_b:b])
            params.append(
                DeltaParam(
                    name=d.name,
                    dtype=str(d.values.dtype).replace("torch.", ""),
                    shape=list(d.values.shape),
                    pos_start=pos_byte_off,
                    pos_end=pos_byte_off + nnz * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + nnz,
                )
            )
            pos_byte_off += nnz * 4
            val_off += nnz
        prev_b = b
        prev_param_start = cum[i]
    if not params:
        return EncodedChunk.empty()
    positions = torch.cat(pos_pieces, dim=0)
    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(
        pos_bytes=positions.cpu().numpy().tobytes(),
        val_tensor=values,
        params=params,
        nnz=val_off,
    )


def _encode_deltas(diffs: list[ParamDiff]) -> EncodedChunk:
    """Gap-encode sorted positions.

    Store ``idx[k] - idx[k-1] - 1`` with ``idx[-1] := -1`` so the first delta
    equals the first index. Each parameter downcasts to uint16 if the max gap
    fits, else uint32. At ~2% density on bf16 weights the typical max gap is
    ~300, so uint16 normally suffices; the uint32 fallback covers pathological
    inputs without correctness risk. The receiver inverts via
    ``idx = cumsum(delta + 1) - 1``.
    """
    if not diffs:
        return EncodedChunk.empty()
    big_val, bounds, big_idx, cum = _sparse_boundaries(diffs)

    kept: list[tuple[ParamDiff, int]] = []
    per_param_deltas: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    prev_b = 0
    prev_param_start = 0
    for i, d in enumerate(diffs):
        b = bounds[i]
        nnz = b - prev_b
        if nnz > 0:
            local_idx = big_idx[prev_b:b] - prev_param_start  # int64, sorted
            prev = torch.cat(
                [
                    torch.tensor(
                        [-1], dtype=local_idx.dtype, device=local_idx.device
                    ),
                    local_idx[:-1],
                ]
            )
            per_param_deltas.append(local_idx - prev - 1)
            val_pieces.append(big_val[prev_b:b])
            kept.append((d, nnz))
        prev_b = b
        prev_param_start = cum[i]

    if not kept:
        return EncodedChunk.empty()

    max_per_param = (
        torch.stack([d.max() for d in per_param_deltas]).cpu().tolist()
    )
    pos_byte_pieces: list[bytes] = []
    pos_byte_off = val_off = 0
    params: list[DeltaParam] = []
    for (d, nnz), deltas, max_d in zip(
        kept, per_param_deltas, max_per_param, strict=True
    ):
        width = 2 if int(max_d) <= 65535 else 4
        np_dtype = np.uint16 if width == 2 else np.uint32
        b_chunk = deltas.cpu().numpy().astype(np_dtype, copy=False).tobytes()
        pos_byte_pieces.append(b_chunk)
        params.append(
            DeltaParam(
                name=d.name,
                dtype=str(d.values.dtype).replace("torch.", ""),
                shape=list(d.values.shape),
                pos_start=pos_byte_off,
                pos_end=pos_byte_off + len(b_chunk),
                pos_width=width,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_byte_off += len(b_chunk)
        val_off += nnz

    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(
        pos_bytes=b"".join(pos_byte_pieces),
        val_tensor=values,
        params=params,
        nnz=val_off,
    )


def encode_chunk(diffs: list[ParamDiff], encoding: DeltaEncodingName) -> EncodedChunk:
    """Encode one chunk of per-parameter diffs into an :class:`EncodedChunk`."""
    if encoding == "indices":
        return _encode_indices(diffs)
    if encoding in ("deltas", "deltas_zstd"):
        # ``deltas`` and ``deltas_zstd`` share the gap encoder. The zstd wrap is
        # applied at safetensors write time by the disk transport, not here.
        return _encode_deltas(diffs)
    raise ValueError(f"unsupported delta encoding: {encoding!r}")


# ---------- decoder (for tests / receivers that live in Python) ----------


def decode_chunk(
    encoding: DeltaEncodingName,
    pos_bytes: bytes,
    val_tensor: torch.Tensor,
    params: list[DeltaParam],
) -> dict[str, torch.Tensor]:
    """Reverse of :func:`encode_chunk`.

    Returns full-shape parameter tensors with NaN at unchanged positions. The
    real receiver lives in SGLang (sgl-project/sglang#26519); this Python
    decoder exists so we can unit-test bit-identity locally.
    """
    out: dict[str, torch.Tensor] = {}
    pos_np = np.frombuffer(pos_bytes, dtype=np.uint8)
    for p in params:
        param_dtype = getattr(torch, p.dtype)
        numel = 1
        for s in p.shape:
            numel *= s
        flat = torch.full(
            (numel,), float("nan"), dtype=param_dtype, device=val_tensor.device
        )
        vals = val_tensor[p.val_start : p.val_end]
        chunk_bytes = pos_np[p.pos_start : p.pos_end]
        n_elems = len(chunk_bytes) // p.pos_width
        if n_elems == 0:
            out[p.name] = flat.view(*p.shape)
            continue
        if p.pos_width == 4:
            idx = torch.from_numpy(
                chunk_bytes.view(np.int32).copy()
            ).to(dtype=torch.int64, device=val_tensor.device)
            if encoding != "indices":
                idx = (idx + 1).cumsum(dim=0) - 1
        elif p.pos_width == 2:
            idx = torch.from_numpy(
                chunk_bytes.view(np.uint16).copy()
            ).to(dtype=torch.int64, device=val_tensor.device)
            idx = (idx + 1).cumsum(dim=0) - 1
        else:
            raise ValueError(f"unsupported pos_width {p.pos_width}")
        flat.index_copy_(0, idx, vals.to(param_dtype))
        out[p.name] = flat.view(*p.shape)
    return out


# ---------- bucket --------------------------------------------------------


@dataclass
class DeltaBucket:
    """Accumulates encoded chunks for one flush.

    Per-parameter offsets in incoming chunks are rebased into the bucket's
    growing position blob and value tensor on ``add``.
    """

    pos_pieces: list[bytes] = field(default_factory=list)
    val_pieces: list[torch.Tensor] = field(default_factory=list)
    params: list[DeltaParam] = field(default_factory=list)
    pos_total: int = 0
    val_total: int = 0
    byte_size: int = 0

    @property
    def has_updates(self) -> bool:
        return bool(self.pos_pieces)

    def should_flush_before_add(self, chunk: EncodedChunk, byte_limit: int) -> bool:
        chunk_bytes = (
            len(chunk.pos_bytes)
            + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        )
        return self.has_updates and self.byte_size + chunk_bytes > byte_limit

    def add(self, chunk: EncodedChunk) -> None:
        for p in chunk.params:
            self.params.append(
                replace(
                    p,
                    pos_start=p.pos_start + self.pos_total,
                    pos_end=p.pos_end + self.pos_total,
                    val_start=p.val_start + self.val_total,
                    val_end=p.val_end + self.val_total,
                )
            )
        self.pos_pieces.append(chunk.pos_bytes)
        self.val_pieces.append(chunk.val_tensor)
        self.pos_total += len(chunk.pos_bytes)
        self.val_total += chunk.val_tensor.numel()
        self.byte_size += (
            len(chunk.pos_bytes)
            + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        )

    def merged_positions_cpu(self) -> torch.Tensor:
        merged = b"".join(self.pos_pieces)
        if not merged:
            return torch.empty(0, dtype=torch.uint8)
        return torch.from_numpy(np.frombuffer(merged, dtype=np.uint8).copy())

    def merged_values(self) -> torch.Tensor:
        if not self.val_pieces:
            return torch.empty(0, dtype=torch.bfloat16)
        return torch.cat(self.val_pieces, dim=0)

    def clear(self) -> None:
        self.pos_pieces.clear()
        self.val_pieces.clear()
        self.params.clear()
        self.pos_total = 0
        self.val_total = 0
        self.byte_size = 0
