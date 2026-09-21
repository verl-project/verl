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

Builds sparse ``indices``-encoding flushes exactly as the sharded engines
assemble them (fixed-width within-parameter positions + a value stream +
checksum) and feeds each through :func:`delta_loader.apply_delta`
against a stand-in model whose ``load_weights`` mimics SGLang's
``param.copy_(loaded)`` semantics. Verifies the masked in-place apply: changed
positions land bit-exactly, and positions outside the delta are never touched.
"""

from __future__ import annotations

import json

import torch

from verl.checkpoint_engine.delta_sync import (
    DeltaParam,
    absolute_index_width,
    checksum,
    pack_absolute_indices,
    unpack_absolute_indices,
)
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


class _FakeModel:
    """Holds live params; load_weights lands on param.copy_(loaded), like SGLang."""

    def __init__(self, named: list[tuple[str, torch.Tensor]]):
        self.params = {n: t.clone() for n, t in named}

    def load_weights(self, chunk):
        for name, tensor in chunk:
            self.params[name].copy_(tensor)

    def named_parameters(self):
        return self.params.items()

    def named_buffers(self):
        return iter(())


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


def _sparse_indices_flush(old_named, new_named, pos_width=4):
    """Assemble one indices-encoding flush from a bytewise old/new diff --
    the same layout the sharded engines' ``_assemble_flush`` produces."""
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for (name, old), (_, new) in zip(old_named, new_named, strict=True):
        fo, fn = old.reshape(-1), new.reshape(-1)
        changed = (fo.view(torch.uint8).view(fo.numel(), -1) != fn.view(torch.uint8).view(fn.numel(), -1)).any(dim=-1)
        idx = changed.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        nnz = int(idx.numel())
        idx_pieces.append(pack_absolute_indices(idx, pos_width))
        val_pieces.append(fn[idx])
        params.append(
            DeltaParam(
                name=name,
                dtype=str(new.dtype).replace("torch.", ""),
                shape=list(new.shape),
                pos_start=pos_off,
                pos_end=pos_off + nnz * pos_width,
                pos_width=pos_width,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_off += nnz * pos_width
        val_off += nnz
    positions = torch.cat(idx_pieces).contiguous().view(torch.uint8)
    values = torch.cat(val_pieces)
    return params, positions, values


def _named_tensors(params, positions, values, encoding="indices"):
    spec = {
        "encoding": encoding,
        "params": [vars(p) for p in params],
        "checksum": int(checksum(positions, values)),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    out = [("__delta_spec__", spec_t), ("__values__", values.clone())]
    if positions.numel():
        out.insert(1, ("__positions__", positions.clone()))
    return out


def test_masked_apply_bit_identical():
    named = _make_named()
    model = _FakeModel(named)

    new_named = []
    for name, t in named:
        new = t.clone()
        flat = new.view(-1)
        idx = torch.tensor([1, 17, 200, 511], dtype=torch.int64) % flat.numel()
        flat[idx] = flat[idx] + 0.5
        new_named.append((name, new))

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    for name, expected in new_named:
        got = model.params[name]
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), f"{name} not bit-identical"


def test_untouched_positions_preserved():
    """Positions absent from the delta must keep the model's LIVE values (not the
    trainer snapshot's) -- proves the apply is masked, not a full overwrite."""
    named = _make_named()
    model = _FakeModel(named)
    # Poison one untouched position in the live model; a full overwrite would revert it.
    sentinel_name = named[0][0]
    model.params[sentinel_name].view(-1)[3] = 42.0

    new_named = []
    for name, t in named:
        new = t.clone()
        new.view(-1)[7] = new.view(-1)[7] + 1.0  # change only position 7
        new_named.append((name, new))

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    live = model.params[sentinel_name].view(-1)
    assert live[3].item() == 42.0, "masked apply must not touch positions outside the delta"
    assert live[7] == new_named[0][1].view(-1)[7]


def test_checksum_mismatch_raises():
    import pytest

    named = _make_named()
    new_named = [(n, t + 0.5) for n, t in named]
    named_tensors = _named_tensors(*_sparse_indices_flush(named, new_named))
    named_tensors[2][1].view(torch.uint8)[0] ^= 0xFF  # corrupt one value byte
    with pytest.raises(RuntimeError, match="checksum"):
        apply_delta(_FakeModel(named), named_tensors)


def test_24bit_positions_roundtrip_and_apply_bit_identical():
    assert absolute_index_width(1 << 24) == 3
    assert absolute_index_width((1 << 24) + 1) == 4
    indices = torch.tensor([0, 1, 255, 256, 65535, 65536, (1 << 24) - 1], dtype=torch.int64)
    packed = pack_absolute_indices(indices, 3)
    assert packed.numel() == indices.numel() * 3
    assert torch.equal(unpack_absolute_indices(packed, 3), indices.to(torch.int32))

    named = _make_named()
    new_named = [(name, tensor.clone()) for name, tensor in named]
    new_named[0][1].view(-1)[[0, 5, 17]] += 1
    new_named[1][1].view(-1)[[3, 31]] -= 1
    model = _FakeModel([(name, tensor.clone()) for name, tensor in named])
    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named, pos_width=3)))
    for name, expected in new_named:
        assert torch.equal(model.params[name].view(torch.int16), expected.view(torch.int16)), name


def test_mixed_position_width_rebases_unaligned_32bit_slice():
    packed24 = pack_absolute_indices(torch.tensor([1], dtype=torch.int32), 3)
    expected32 = torch.tensor([17, 1 << 24], dtype=torch.int32)
    packed32 = pack_absolute_indices(expected32, 4)
    mixed = torch.cat((packed24, packed32))
    assert mixed[3:].storage_offset() == 3
    assert torch.equal(unpack_absolute_indices(mixed[3:], 4), expected32)


def test_dense_flush_applies_full_tensors():
    """Dense (first-sync) flushes carry values only; the loader must apply them verbatim."""
    named = _make_named()
    model = _FakeModel([(n, torch.zeros_like(t)) for n, t in named])  # dummy init

    params, pieces, val_off = [], [], 0
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append(
            {
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + flat.numel(),
            }
        )
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)

    spec = {"encoding": "dense", "params": params, "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values))}
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    apply_delta(model, [("__delta_spec__", spec_t), ("__values__", values)])

    for name, expected in named:
        assert torch.equal(model.params[name].view(torch.int16), expected.view(torch.int16)), name


def _dense_verify_flush(named, is_last=True, verify=True):
    params, pieces, val_off = [], [], 0
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append(
            {
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + flat.numel(),
            }
        )
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)
    spec = {
        "encoding": "dense",
        "verify": verify,
        "is_last": is_last,
        "params": params,
        "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values)),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    return [("__delta_spec__", spec_t), ("__values__", values)]


def test_verify_sweep_passes_on_identical_state():
    """A verify flush against a bit-identical model reports zero mismatches."""
    named = _make_named()
    model = _FakeModel([(n, t.clone()) for n, t in named])
    apply_delta(model, _dense_verify_flush(named))  # must not raise


def test_verify_sweep_fails_loud_on_divergence():
    """A single flipped element in the server state must fail the sweep."""
    import pytest

    named = _make_named()
    diverged = [(n, t.clone()) for n, t in named]
    diverged[1][1].view(-1)[3] += 1.0
    model = _FakeModel(diverged)
    with pytest.raises(RuntimeError, match="verification FAILED"):
        apply_delta(model, _dense_verify_flush(named))


def test_verify_sweep_replays_fused_members_atomically():
    """DSv4 creates and drains its fusion cache within one load_weights call."""
    names = (
        "model.layers.0.self_attn.wq_a.weight",
        "model.layers.0.self_attn.wkv.weight",
    )
    named = [(name, torch.randn(8, 8, dtype=torch.bfloat16)) for name in names]

    class _FusionModel(_FakeModel):
        def load_weights(self, chunk):
            chunk_names = {name for name, _tensor in chunk}
            if chunk_names.intersection(names):
                assert set(names).issubset(chunk_names), chunk_names
            super().load_weights(chunk)

    apply_delta(_FusionModel(named), _dense_verify_flush(named))
