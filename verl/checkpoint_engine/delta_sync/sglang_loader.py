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
"""In-process sparse delta apply for SGLang, loaded via ``--custom-weight-loader``.

SGLang's ``update_weights_from_tensor`` supports pluggable loaders: when the
request's ``load_format`` names an import path registered in
``--custom-weight-loader``, SGLang ``dynamic_import``s it **inside every TP
worker process** and calls ``loader(model, named_tensors)``. That is exactly
the hook a sparse delta needs: the delta payload is decoded and applied *in
place* onto SGLang's live weights (masked overwrite of only the changed
positions), so the receiver never stages a full-model mirror anywhere —
peak memory is one decode chunk, independent of model size — and no SGLang
fork or patch is required.

Wire contract (what the delta checkpoint engine sends per flush):

* ``__delta_spec__`` — uint8 tensor holding a JSON manifest
  ``{"encoding", "params": [DeltaParam-dict...], "checksum"}``.
* ``__positions__`` — uint8 blob of packed positions (per-param slices are
  byte offsets ``pos_start:pos_end``; ``indices`` packs little-endian int32
  absolute positions, ``deltas`` packs uint16/uint32 gaps per ``pos_width``).
* ``__values__``  — the changed values in the flush's (uniform) dtype;
  per-param slices are element offsets ``val_start:val_end``.

Register at server launch (verl config)::

    +actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader='["verl.checkpoint_engine.delta_sync.sglang_loader.apply_delta"]'
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager

import torch

# Cap on the densified tensors handed to one model.load_weights call, matching
# SGLang's own delta-apply chunking default.
CHUNK_BYTES = 512 << 20

# Import path callers pass as both --custom-weight-loader and load_format.
LOADER_FQN = "verl.checkpoint_engine.delta_sync.sglang_loader.apply_delta"


def apply_delta(model, named_tensors) -> None:
    """Decode one sparse delta flush and masked-apply it onto ``model`` in place."""
    from .encode import checksum as _checksum

    tensors = dict(named_tensors)
    spec = json.loads(bytes(tensors["__delta_spec__"].cpu().numpy().tobytes()).decode())
    positions = tensors["__positions__"]
    values = tensors["__values__"]

    got = _checksum(positions, values)
    if got != int(spec["checksum"]):
        raise RuntimeError(
            f"delta checksum mismatch in sglang loader: got {got}, expected {spec['checksum']}; "
            "indicates corruption between sender encode and receiver apply"
        )

    encoding = spec["encoding"]
    with _masked_copy():
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for p in spec["params"]:
            t = _decode_one(encoding, positions, values, p)
            nbytes = t.numel() * t.element_size()
            if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
                model.load_weights(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append((p["name"], t))
            chunk_bytes += nbytes
        if chunk:
            model.load_weights(chunk)


def _decode_one(encoding: str, positions: torch.Tensor, values: torch.Tensor, p: dict) -> torch.Tensor:
    """Densify one param's sparse delta into a full-shape NaN-masked tensor.

    ``indices`` positions are reinterpreted via an int32 view (8 B/element
    transient) rather than a per-byte int64 unpack (32 B/element), so even a
    full-seed flush of a large embedding stays within a few GiB.
    """
    numel = math.prod(p["shape"])
    dtype = getattr(torch, p["dtype"])
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    vals = values[p["val_start"] : p["val_end"]]
    if vals.numel() == 0:
        return flat.view(p["shape"])

    pos_b = positions[p["pos_start"] : p["pos_end"]]
    if encoding == "indices":
        # pos_start is always int32-aligned (each entry packs nnz * 4 bytes).
        idx = pos_b.view(torch.int32).to(torch.int64)
    elif encoding == "deltas":
        w = int(p["pos_width"])
        bv = pos_b.view(-1, w).to(torch.int64)
        if w == 2:
            gaps = bv[:, 0] | (bv[:, 1] << 8)
        else:
            gaps = bv[:, 0] | (bv[:, 1] << 8) | (bv[:, 2] << 16) | (bv[:, 3] << 24)
        idx = (gaps + 1).cumsum(dim=0) - 1
    else:
        raise ValueError(f"unsupported delta encoding: {encoding!r}")

    flat.index_copy_(0, idx, vals.to(dtype))
    return flat.view(p["shape"])


@contextmanager
def _masked_copy():
    """Temporarily make ``Tensor.copy_`` skip NaN positions in the source.

    SGLang's per-model ``load_weights`` ultimately lands on ``param.copy_(loaded)``
    (possibly on a narrowed TP slice). Under this context a NaN-masked source
    overwrites only the changed positions; fully dense sources (e.g. the first
    full-seed flush) take the original fast path untouched.
    """
    orig_copy = torch.Tensor.copy_

    def masked_copy_(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and src.is_floating_point() and self.shape == src.shape:
            mask = ~torch.isnan(src)
            if not bool(mask.all()):
                self[mask] = src[mask].to(self.dtype)
                return self
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy
