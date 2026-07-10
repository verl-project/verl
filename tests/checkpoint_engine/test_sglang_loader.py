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
"""Bit-identity tests for the SGLang custom-weight-loader delta apply.

Drives the real sender machinery (``DeltaState`` + ``iter_delta_flushes``) and
feeds each flush through :func:`sglang_loader.apply_delta` against a stand-in
model whose ``load_weights`` mimics SGLang's ``param.copy_(loaded)`` semantics.
Verifies the masked in-place apply: changed positions land bit-exactly, and
positions outside the delta payload are never touched.
"""

from __future__ import annotations

import json

import pytest
import torch

from verl.checkpoint_engine.delta_sync import DeltaState, iter_delta_flushes
from verl.checkpoint_engine.delta_sync.sglang_loader import apply_delta


class _FakeModel:
    """Holds live params; load_weights lands on param.copy_(loaded), like SGLang."""

    def __init__(self, named: list[tuple[str, torch.Tensor]]):
        self.params = {n: t.clone() for n, t in named}

    def load_weights(self, chunk):
        for name, tensor in chunk:
            self.params[name].copy_(tensor)


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


def _flush_to_named_tensors(flush) -> list[tuple[str, torch.Tensor]]:
    """Package one DeltaFlush exactly like DeltaCheckpointEngine.update_weights_via_server."""
    spec = {
        "encoding": flush.encoding,
        "params": [vars(p) for p in flush.params],
        "checksum": int(flush.checksum),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    return [
        ("__delta_spec__", spec_t),
        ("__positions__", flush.positions_cpu.clone()),
        ("__values__", flush.values_gpu.clone()),
    ]


@pytest.mark.parametrize("encoding", ["indices", "deltas"])
def test_masked_apply_bit_identical(encoding: str):
    state = DeltaState()
    named = _make_named()
    list(iter_delta_flushes(iter(named), state, encoding=encoding, bucket_bytes=1 << 20))  # seed

    model = _FakeModel(named)

    # Mutate a sparse set of positions in each tensor.
    new_named = []
    for name, t in named:
        new = t.clone()
        flat = new.view(-1)
        idx = torch.tensor([1, 17, 200, 511], dtype=torch.int64) % flat.numel()
        flat[idx] = flat[idx] + 0.5
        new_named.append((name, new))

    flushes = list(iter_delta_flushes(iter(new_named), state, encoding=encoding, bucket_bytes=1 << 20))
    assert flushes
    for flush in flushes:
        apply_delta(model, _flush_to_named_tensors(flush))

    for name, expected in new_named:
        got = model.params[name]
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), f"{name} not bit-identical"


def test_untouched_positions_preserved():
    """Positions absent from the delta must keep the model's LIVE values (not the
    trainer snapshot's) -- proves the apply is masked, not a full overwrite."""
    state = DeltaState()
    named = _make_named()
    list(iter_delta_flushes(iter(named), state, encoding="indices", bucket_bytes=1 << 20))

    model = _FakeModel(named)
    # Poison one untouched position in the live model; a full overwrite would revert it.
    sentinel_name = named[0][0]
    model.params[sentinel_name].view(-1)[3] = 42.0

    new_named = []
    for name, t in named:
        new = t.clone()
        new.view(-1)[7] = new.view(-1)[7] + 1.0  # change only position 7
        new_named.append((name, new))

    flushes = list(iter_delta_flushes(iter(new_named), state, encoding="indices", bucket_bytes=1 << 20))
    for flush in flushes:
        apply_delta(model, _flush_to_named_tensors(flush))

    live = model.params[sentinel_name].view(-1)
    assert live[3].item() == 42.0, "masked apply must not touch positions outside the delta"
    assert live[7] == new_named[0][1].view(-1)[7]


def test_checksum_mismatch_raises():
    state = DeltaState()
    named = _make_named()
    list(iter_delta_flushes(iter(named), state, encoding="indices", bucket_bytes=1 << 20))
    new_named = [(n, t + 0.5) for n, t in named]
    flushes = list(iter_delta_flushes(iter(new_named), state, encoding="indices", bucket_bytes=1 << 20))
    named_tensors = _flush_to_named_tensors(flushes[0])
    named_tensors[2][1].view(torch.uint8)[0] ^= 0xFF  # corrupt one value byte
    with pytest.raises(RuntimeError, match="checksum"):
        apply_delta(_FakeModel(named), named_tensors)


def test_dense_flush_applies_full_tensors():
    """Dense (first-sync) flushes carry values only; the loader must apply them verbatim."""
    named = _make_named()
    model = _FakeModel([(n, torch.zeros_like(t)) for n, t in named])  # dummy init

    params, pieces, val_off = [], [], 0
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append({
            "name": name, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
            "pos_start": 0, "pos_end": 0, "pos_width": 4,
            "val_start": val_off, "val_end": val_off + flat.numel(),
        })
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)

    from verl.checkpoint_engine.delta_sync.encode import checksum

    spec = {"encoding": "dense", "params": params,
            "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values))}
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    apply_delta(model, [("__delta_spec__", spec_t), ("__values__", values)])

    for name, expected in named:
        assert torch.equal(model.params[name].view(torch.int16), expected.view(torch.int16)), name
