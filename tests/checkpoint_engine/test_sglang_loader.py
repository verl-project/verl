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
assemble them (int32 within-parameter flat positions viewed as bytes + a value
stream + checksum) and feeds each through :func:`delta_loader.apply_delta`
against a stand-in model whose ``load_weights`` mimics SGLang's
``param.copy_(loaded)`` semantics. Verifies the masked in-place apply: changed
positions land bit-exactly, and positions outside the delta are never touched.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from verl.checkpoint_engine.delta_sync import DeltaParam, checksum
from verl.workers.rollout.sglang_rollout.delta_loader import _masked_copy, apply_delta


class _FakeModel(torch.nn.Module):
    """Holds live params; load_weights lands on param.copy_(loaded), like SGLang."""

    def __init__(self, named: list[tuple[str, torch.Tensor]]):
        super().__init__()
        self.params = torch.nn.ParameterDict(
            {n.replace(".", "_"): torch.nn.Parameter(t.clone(), requires_grad=False) for n, t in named}
        )
        self._param_names = {name: name.replace(".", "_") for name, _ in named}

    def load_weights(self, chunk):
        for name, tensor in chunk:
            self.get_param(name).copy_(tensor)

    def get_param(self, name):
        return self.params[self._param_names[name]]


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


def _sparse_indices_flush(old_named, new_named):
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
        idx_pieces.append(idx.to(torch.int32))
        val_pieces.append(fn[idx])
        params.append(
            DeltaParam(
                name=name,
                dtype=str(new.dtype).replace("torch.", ""),
                shape=list(new.shape),
                pos_start=pos_off,
                pos_end=pos_off + nnz * 4,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_off += nnz * 4
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
        got = model.get_param(name)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), f"{name} not bit-identical"


def test_untouched_positions_preserved():
    """Positions absent from the delta must keep the model's LIVE values (not the
    trainer snapshot's) -- proves the apply is masked, not a full overwrite."""
    named = _make_named()
    model = _FakeModel(named)
    # Poison one untouched position in the live model; a full overwrite would revert it.
    sentinel_name = named[0][0]
    model.get_param(sentinel_name).view(-1)[3] = 42.0

    new_named = []
    for name, t in named:
        new = t.clone()
        new.view(-1)[7] = new.view(-1)[7] + 1.0  # change only position 7
        new_named.append((name, new))

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    live = model.get_param(sentinel_name).view(-1)
    assert live[3].item() == 42.0, "masked apply must not touch positions outside the delta"
    assert live[7] == new_named[0][1].view(-1)[7]


def test_masked_apply_does_not_change_scratch_copy_semantics():
    named = _make_named()
    model = _FakeModel(named)
    scratch = torch.zeros_like(named[0][1])

    def load_weights(chunk):
        for name, tensor in chunk:
            scratch.copy_(tensor)
            model.get_param(name).copy_(tensor)

    model.load_weights = load_weights
    new_named = [(name, tensor.clone()) for name, tensor in named]
    new_named[0][1].view(-1)[7] += 1.0

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    assert torch.isnan(scratch).any(), "non-model scratch copies must retain ordinary copy_ semantics"
    assert model.get_param(named[0][0]).view(-1)[7] == new_named[0][1].view(-1)[7]


def test_masked_apply_covers_views_into_parameter_storage():
    named = _make_named()
    model = _FakeModel(named)

    def load_weights(chunk):
        for name, tensor in chunk:
            destination = model.get_param(name)
            midpoint = destination.shape[0] // 2
            destination[:midpoint].copy_(tensor[:midpoint])
            destination[midpoint:].copy_(tensor[midpoint:])

    model.load_weights = load_weights
    new_named = [(name, tensor.clone()) for name, tensor in named]
    for _, tensor in new_named:
        tensor.view(-1)[7] += 1.0
        tensor.view(-1)[-7] += 1.0

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    for name, expected in new_named:
        assert torch.equal(model.get_param(name).view(torch.int16), expected.view(torch.int16))


def test_masked_apply_covers_registered_buffer_storage():
    named = _make_named()
    model = _FakeModel(named)
    model.register_buffer("packed_weight", named[0][1].clone())

    def load_weights(chunk):
        for _, tensor in chunk:
            model.packed_weight.copy_(tensor)
            break

    model.load_weights = load_weights
    new_named = [(name, tensor.clone()) for name, tensor in named]
    new_named[0][1].view(-1)[7] += 1.0
    model.packed_weight.view(-1)[3] = 42.0

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    assert model.packed_weight.view(-1)[3].item() == 42.0
    assert model.packed_weight.view(-1)[7] == new_named[0][1].view(-1)[7]


def test_post_load_weights_runs_with_original_copy_semantics():
    named = _make_named()
    model = _FakeModel(named)
    derived = torch.zeros_like(named[0][1])
    post_load_calls = 0

    def post_load_weights():
        nonlocal post_load_calls
        post_load_calls += 1
        derived.copy_(torch.full_like(derived, float("nan")))

    def load_weights(chunk):
        for name, tensor in chunk:
            model.get_param(name).copy_(tensor)
        model.post_load_weights()

    model.load_weights = load_weights
    model.post_load_weights = post_load_weights
    new_named = [(name, tensor.clone()) for name, tensor in named]
    new_named[0][1].view(-1)[7] += 1.0

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    assert post_load_calls == 1
    assert torch.isnan(derived).all(), "post-load derived tensor writes must not use masked copy semantics"


def test_quant_method_post_load_runs_with_original_copy_semantics():
    named = _make_named()
    model = _FakeModel(named)
    derived = torch.zeros_like(named[0][1])
    post_load_calls = 0

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            nonlocal post_load_calls
            post_load_calls += 1
            assert layer is model
            derived.copy_(torch.full_like(derived, float("nan")))

    model.quant_method = QuantMethod()

    def load_weights(chunk):
        for name, tensor in chunk:
            model.get_param(name).copy_(tensor)
        model.quant_method.process_weights_after_loading(model)

    model.load_weights = load_weights
    new_named = [(name, tensor.clone()) for name, tensor in named]
    new_named[0][1].view(-1)[7] += 1.0

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    assert post_load_calls == 1
    assert torch.isnan(derived).all(), "quantization post-load writes must not use masked copy semantics"


def test_masked_copy_restores_global_and_model_hooks_after_error():
    named = _make_named()
    model = _FakeModel(named)

    def post_load_weights():
        pass

    model.post_load_weights = post_load_weights
    original_copy = torch.Tensor.copy_
    original_post_load = model.post_load_weights

    with pytest.raises(RuntimeError, match="load failed"):
        with _masked_copy(model):
            assert torch.Tensor.copy_ is not original_copy
            assert model.post_load_weights is not original_post_load
            raise RuntimeError("load failed")

    assert torch.Tensor.copy_ is original_copy
    assert model.post_load_weights is original_post_load


def test_masked_copy_rolls_back_hooks_when_one_cannot_be_wrapped():
    named = _make_named()
    model = _FakeModel(named)

    def post_load_weights():
        pass

    class ReadOnlyQuantMethod:
        __slots__ = ()

        def process_weights_after_loading(self, layer):
            pass

    model.post_load_weights = post_load_weights
    model.quant_method = ReadOnlyQuantMethod()
    original_copy = torch.Tensor.copy_
    original_post_load = model.post_load_weights

    with pytest.raises(RuntimeError, match="cannot safely isolate post-load hook"):
        with _masked_copy(model):
            pass

    assert torch.Tensor.copy_ is original_copy
    assert model.post_load_weights is original_post_load


def test_masked_copy_fails_closed_when_model_storage_is_unavailable():
    named = _make_named()
    model = _FakeModel(named)

    with (
        patch.object(torch.Tensor, "untyped_storage", side_effect=NotImplementedError("unsupported")),
        pytest.raises(RuntimeError, match="cannot identify storage for model tensor"),
    ):
        with _masked_copy(model):
            pass


def test_checksum_mismatch_raises():
    named = _make_named()
    new_named = [(n, t + 0.5) for n, t in named]
    named_tensors = _named_tensors(*_sparse_indices_flush(named, new_named))
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
        assert torch.equal(model.get_param(name).view(torch.int16), expected.view(torch.int16)), name
