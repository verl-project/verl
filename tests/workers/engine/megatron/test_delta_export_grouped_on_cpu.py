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
"""CPU unit tests for Megatron-Bridge ``is_grouped_export=True`` delta export."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

from verl.workers.engine.spec import ShardSpec


def _load_delta_export():
    try:
        from verl.workers.engine.megatron import delta_export

        return delta_export
    except ImportError:
        path = pathlib.Path(__file__).resolve().parents[4] / "verl/workers/engine/megatron/delta_export.py"
        spec = importlib.util.spec_from_file_location("_delta_export_under_test", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


_de = _load_delta_export()
McoreParamExport = _de.McoreParamExport
_extract_expert_id = _de._extract_expert_id
_get_hf_expected_shape = _de._get_hf_expected_shape
_get_num_moe_experts = _de._get_num_moe_experts
_GroupedExportMeta = _de._GroupedExportMeta
_slot_idx_dtype = _de._slot_idx_dtype
mcore_hf_delta_entry = _de.mcore_hf_delta_entry


class _StubRowSplitProbe:
    """Mimics an ETP-sharded ``FusedGatedExpertMapping`` / ``FusedExpertMapping`` probe.

    Each ETP rank owns a disjoint slice of rows of one expert's 2D matrix and
    returns a full 2D per-expert tensor with NaNs outside this ETP rank's rows.
    """

    def __init__(self, hf_name: str, etp_rank: int, etp_size: int):
        self.hf_name = hf_name
        self.etp_rank = etp_rank
        self.etp_size = etp_size

    def megatron_to_hf(self, buf: torch.Tensor, _module):
        nan = torch.full_like(buf, float("nan"))
        chunks = [nan] * self.etp_size
        chunks[self.etp_rank] = buf
        return {self.hf_name: torch.cat(chunks, dim=0)}


class _StubIdentityProbe:
    def __init__(self, hf_name: str):
        self.hf_name = hf_name

    def megatron_to_hf(self, buf: torch.Tensor, _module):
        return {self.hf_name: buf}


class _StubMultiKeyProbe:
    """Mimics ``StackedExpertGatedMLPMapping`` returning two stacked HF tensors."""

    def __init__(self, gate_name: str, up_name: str):
        self.gate_name = gate_name
        self.up_name = up_name

    def megatron_to_hf(self, buf: torch.Tensor, _module):
        gate, up = torch.chunk(buf, 2, dim=0)
        return {self.gate_name: gate, self.up_name: up}


def _make_record(
    megatron_name: str,
    param: torch.Tensor | None,
    probe,
    grouped_export: _GroupedExportMeta | None = None,
    slots: list | None = None,
) -> McoreParamExport:
    shape = tuple(param.shape) if param is not None else (0,)
    return McoreParamExport(
        megatron_name=megatron_name,
        param=param,
        spec=ShardSpec(full_shape=shape),
        probe=probe,
        module=None,
        slots=slots,
        grouped_export=grouped_export,
    )


def test_grouped_export_ep_etp_disjoint_and_bitwise_exact():
    num_experts = 4
    etp_size = 2
    rows_per_etp = 3
    hidden = 5
    hf_name = "model.layers.1.mlp.experts.gate_up_proj"

    torch.manual_seed(0)
    full_ref = torch.randn(num_experts, rows_per_etp * etp_size, hidden, dtype=torch.bfloat16)

    # Full-delta assembly across all (expert_id, etp_rank) shards.
    assembled = torch.full_like(full_ref, float("nan"))
    covered = torch.zeros(full_ref.numel(), dtype=torch.int32)
    flat = assembled.view(-1)

    for expert_id in range(num_experts):
        for etp_rank in range(etp_size):
            shard = full_ref[expert_id, etp_rank * rows_per_etp : (etp_rank + 1) * rows_per_etp].clone()
            rec = _make_record(
                megatron_name=f"decoder.layers.1.mlp.experts.linear_fc1.weight{expert_id}",
                param=shard,
                probe=_StubRowSplitProbe(hf_name, etp_rank, etp_size),
                grouped_export=_GroupedExportMeta(expert_id=expert_id, num_experts=num_experts),
            )
            slot_cache: dict = {}
            lidx = torch.arange(shard.numel(), dtype=torch.int64)
            slots, dtype_str, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, shard.reshape(-1), slot_cache)
            assert dtype_str == "bfloat16"
            assert slots == [(hf_name, (num_experts, rows_per_etp * etp_size, hidden))]
            assert counts.tolist() == [shard.numel()]
            assert hf_idx.dtype == torch.int32
            pos = hf_idx.to(torch.int64)
            assert (covered[pos] == 0).all(), "overlapping coordinates across (expert_id, etp_rank)"
            flat[pos] = hf_val
            covered[pos] += 1

    assert (covered == 1).all()
    assert torch.equal(assembled.view(torch.int16), full_ref.view(torch.int16))

    # Sparse partial delta: only 3 arbitrary elements changed on expert 2, ETP rank 1.
    shard = full_ref[2, rows_per_etp : 2 * rows_per_etp].clone()
    rec = _make_record(
        megatron_name="decoder.layers.1.mlp.experts.linear_fc1.weight2",
        param=shard,
        probe=_StubRowSplitProbe(hf_name, etp_rank=1, etp_size=etp_size),
        grouped_export=_GroupedExportMeta(expert_id=2, num_experts=num_experts),
        slots=[(hf_name, (num_experts, rows_per_etp * etp_size, hidden))],
    )
    sparse_idx = torch.tensor([0, 4, shard.numel() - 1], dtype=torch.int64)
    sparse_val = shard.reshape(-1)[sparse_idx]
    _slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, sparse_idx, sparse_val, {})
    assert counts.tolist() == [3]
    expected = full_ref.reshape(-1)[hf_idx.to(torch.int64)]
    assert torch.equal(hf_val.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    ("expected_shapes", "should_transpose"),
    [
        (None, True),  # unconditional when HF state shape unavailable
        ({"experts.down_proj": (4, 3, 5)}, True),  # adaptive: HF checkpoint is transposed
        ({"experts.down_proj": (4, 5, 3)}, False),  # adaptive: HF checkpoint already matches stacked
    ],
)
def test_grouped_export_transpose_on_export(expected_shapes, should_transpose):
    num_experts = 4
    hf_name = "experts.down_proj"
    torch.manual_seed(1)
    # Non-square per-expert slices [5, 3] so transpose [3, 5] is unambiguous.
    per_expert = [torch.randn(5, 3, dtype=torch.bfloat16) for _ in range(num_experts)]
    stacked = torch.stack(per_expert, dim=0)
    ref = stacked.transpose(-1, -2).contiguous() if should_transpose else stacked

    assembled = torch.full_like(ref, float("nan"))
    flat = assembled.view(-1)
    for eid, w in enumerate(per_expert):
        rec = _make_record(
            megatron_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{eid}",
            param=w,
            probe=_StubIdentityProbe(hf_name),
            grouped_export=_GroupedExportMeta(
                expert_id=eid,
                num_experts=num_experts,
                transpose_on_export=True,
                expected_shapes=expected_shapes,
            ),
        )
        lidx = torch.arange(w.numel(), dtype=torch.int64)
        slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, w.reshape(-1), {})
        assert slots == [(hf_name, tuple(ref.shape))]
        assert counts.tolist() == [w.numel()]
        flat[hf_idx.to(torch.int64)] = hf_val

    assert torch.equal(assembled.view(torch.int16), ref.view(torch.int16))


def test_grouped_export_1d_bias_and_multi_key():
    num_experts = 3
    hidden = 4
    # 1D per-expert bias with transpose_on_export=True must stay 2D (E, H).
    biases = [torch.arange(hidden, dtype=torch.bfloat16) + eid * 10 for eid in range(num_experts)]
    ref_bias = torch.stack(biases, dim=0)
    assembled = torch.full_like(ref_bias, float("nan"))
    for eid, b in enumerate(biases):
        rec = _make_record(
            megatron_name=f"decoder.layers.0.mlp.experts.linear_fc2.bias{eid}",
            param=b,
            probe=_StubIdentityProbe("model.layers.0.mlp.experts.down_proj_bias"),
            grouped_export=_GroupedExportMeta(expert_id=eid, num_experts=num_experts, transpose_on_export=True),
        )
        lidx = torch.arange(b.numel(), dtype=torch.int64)
        slots, _dt, _counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, b, {})
        assert slots == [("model.layers.0.mlp.experts.down_proj_bias", (num_experts, hidden))]
        assembled.view(-1)[hf_idx.to(torch.int64)] = hf_val
    assert torch.equal(assembled.view(torch.int16), ref_bias.view(torch.int16))

    # Multi-key grouped mapping (e.g. Step3.5 StackedExpertGatedMLPMapping).
    torch.manual_seed(2)
    fc1_list = [torch.randn(6, 4, dtype=torch.bfloat16) for _ in range(num_experts)]
    ref_gate = torch.stack([w[:3] for w in fc1_list], dim=0)
    ref_up = torch.stack([w[3:] for w in fc1_list], dim=0)
    got = {"gate": torch.full_like(ref_gate, float("nan")), "up": torch.full_like(ref_up, float("nan"))}
    for eid, w in enumerate(fc1_list):
        rec = _make_record(
            megatron_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{eid}",
            param=w,
            probe=_StubMultiKeyProbe("gate", "up"),
            grouped_export=_GroupedExportMeta(expert_id=eid, num_experts=num_experts),
        )
        lidx = torch.arange(w.numel(), dtype=torch.int64)
        slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, w.reshape(-1), {})
        assert slots == [("gate", (num_experts, 3, 4)), ("up", (num_experts, 3, 4))]
        off = 0
        for (sname, _sshape), c in zip(slots, counts.tolist(), strict=True):
            got[sname].view(-1)[hf_idx[off : off + c].to(torch.int64)] = hf_val[off : off + c]
            off += c
    assert torch.equal(got["gate"].view(torch.int16), ref_gate.view(torch.int16))
    assert torch.equal(got["up"].view(torch.int16), ref_up.view(torch.int16))


def test_pp_placeholder_and_int64_index_dtype_alignment():
    small_slots = [("experts.gate_up_proj", (8, 128, 64))]
    large_slots = [("experts.gate_up_proj", (256, 4096, 4096))]  # 4B elems >= 2^31

    assert _slot_idx_dtype(small_slots) == torch.int32
    assert _slot_idx_dtype(large_slots) == torch.int64

    empty_idx = torch.empty(0, dtype=torch.int64)
    empty_val = torch.empty(0, dtype=torch.bfloat16)

    for slots, expected_dtype in [(small_slots, torch.int32), (large_slots, torch.int64)]:
        placeholder = _make_record("decoder.layers.1.mlp.experts.linear_fc1.weight0", None, None, slots=slots)
        owner = _make_record(
            "decoder.layers.1.mlp.experts.linear_fc1.weight0",
            torch.zeros(4, 4, dtype=torch.bfloat16),
            _StubIdentityProbe("experts.gate_up_proj"),
            grouped_export=_GroupedExportMeta(expert_id=0, num_experts=8),
            slots=slots,
        )
        for rec in (placeholder, owner):
            got_slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, empty_idx, empty_val, {})
            assert got_slots == slots
            assert counts.tolist() == [0]
            assert hf_idx.dtype == expected_dtype
            assert hf_val.numel() == 0


def test_non_grouped_regression_and_shape_mismatch_guard():
    w = torch.randn(4, 3, dtype=torch.bfloat16)
    rec = _make_record("decoder.layers.0.mlp.linear_fc1.weight", w, _StubIdentityProbe("layers.0.mlp.fc1"))
    lidx = torch.arange(w.numel(), dtype=torch.int64)
    slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, w.reshape(-1), {})
    assert slots == [("layers.0.mlp.fc1", (4, 3))]
    assert counts.tolist() == [12]
    assert hf_idx.tolist() == list(range(12))
    assert torch.equal(hf_val.view(torch.int16), w.reshape(-1).view(torch.int16))

    # Mismatched pre-seeded slot shape on a grouped-export record must fail loud.
    bad_rec = _make_record(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        w,
        _StubIdentityProbe("experts.gate_up_proj"),
        grouped_export=_GroupedExportMeta(expert_id=0, num_experts=4),
        slots=[("experts.gate_up_proj", (4, 3))],  # wrong 2D slot instead of (4, 4, 3)
    )
    with pytest.raises(AssertionError, match="does not match slot shape"):
        mcore_hf_delta_entry(bad_rec, 0, lidx, w.reshape(-1), {})


def test_grouped_export_checkpoint_shape_guard():
    """The checkpoint header is the only witness that num_experts and the slice rank are right."""
    hf_name = "experts.gate_up_proj"
    w = torch.randn(4, 3, dtype=torch.bfloat16)
    lidx = torch.arange(w.numel(), dtype=torch.int64)

    def _run(num_experts, param, checkpoint_shape):
        rec = _make_record(
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            param,
            _StubIdentityProbe(hf_name),
            grouped_export=_GroupedExportMeta(
                expert_id=0,
                num_experts=num_experts,
                expected_shapes={hf_name: checkpoint_shape},
            ),
        )
        return mcore_hf_delta_entry(rec, 0, torch.arange(param.numel(), dtype=torch.int64), param.reshape(-1), {})

    # Agreeing checkpoint shape: 4 experts x [4, 3] slice.
    slots, _dt, counts, _idx, _val = _run(4, w, (4, 4, 3))
    assert slots == [(hf_name, (4, 4, 3))]
    assert counts.tolist() == [w.numel()]

    # num_experts resolved from the wrong config: slot claims 8 layers, checkpoint has 4.
    with pytest.raises(AssertionError, match="does not match the checkpoint tensor shape"):
        _run(8, w, (4, 4, 3))

    # Slice already carries a leading expert dim, so stacking would add a 4th dim.
    with pytest.raises(AssertionError, match="does not match the checkpoint tensor shape"):
        _run(4, torch.randn(4, 4, 3, dtype=torch.bfloat16), (4, 4, 3))

    # An unrelated key in expected_shapes must not be mistaken for this slot's truth.
    rec = _make_record(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        w,
        _StubIdentityProbe(hf_name),
        grouped_export=_GroupedExportMeta(expert_id=0, num_experts=4, expected_shapes={"experts.down_proj": (4, 9, 9)}),
    )
    slots, _dt, _counts, _idx, _val = mcore_hf_delta_entry(rec, 0, lidx, w.reshape(-1), {})
    assert slots == [(hf_name, (4, 4, 3))]


def test_helpers_extract_expert_id_num_experts_and_expected_shape(tmp_path):
    assert _extract_expert_id("decoder.layers.1.mlp.experts.linear_fc1.weight7") == 7
    assert _extract_expert_id("decoder.layers.1.mlp.experts.linear_fc2.bias12") == 12
    assert _extract_expert_id("decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.weight") == 3
    assert _extract_expert_id("model.layers.1.mlp.experts.5.gate_proj.weight") == 5
    with pytest.raises(ValueError, match="No expert number"):
        _extract_expert_id("decoder.layers.0.self_attention.linear_qkv.weight")

    mod_with_cfg = types.SimpleNamespace(config=types.SimpleNamespace(num_moe_experts=16))
    wrapped = [types.SimpleNamespace(module=mod_with_cfg)]
    assert _get_num_moe_experts(None, mod_with_cfg) == 16
    assert _get_num_moe_experts(wrapped, types.SimpleNamespace()) == 16
    with pytest.raises(ValueError, match="num_moe_experts"):
        _get_num_moe_experts([types.SimpleNamespace()], types.SimpleNamespace())

    # Dict-backed hf_pretrained.state
    dict_bridge = types.SimpleNamespace(
        hf_pretrained=types.SimpleNamespace(state={"k1": torch.empty(8, 12, 6, dtype=torch.bfloat16)})
    )
    cache: dict = {}
    assert _get_hf_expected_shape(dict_bridge, "k1", cache) == (8, 12, 6)
    assert _get_hf_expected_shape(dict_bridge, "missing", cache) is None

    # Safetensors header-only lookup via source
    safetensors_torch = pytest.importorskip("safetensors.torch")
    st_path = tmp_path / "model.safetensors"
    safetensors_torch.save_file({"k_st": torch.zeros(4, 6, 8, dtype=torch.bfloat16)}, str(st_path))

    class _NoLoadState:
        def __init__(self, source):
            self.source = source

        def __contains__(self, _k):
            raise AssertionError("header-only safetensors lookup should not fall back to loading tensors")

    st_source = types.SimpleNamespace(path=tmp_path, key_to_filename_map={"k_st": "model.safetensors"})
    st_bridge = types.SimpleNamespace(hf_pretrained=types.SimpleNamespace(state=_NoLoadState(st_source)))
    st_cache: dict = {}
    assert _get_hf_expected_shape(st_bridge, "k_st", st_cache) == (4, 6, 8)

    # Without a key -> filename index, the shards are walked once and every header cached.
    shard_dir = tmp_path / "sharded"
    shard_dir.mkdir()
    safetensors_torch.save_file({"a": torch.zeros(2, 3, dtype=torch.bfloat16)}, str(shard_dir / "p1.safetensors"))
    safetensors_torch.save_file({"b": torch.zeros(5, 7, dtype=torch.bfloat16)}, str(shard_dir / "p2.safetensors"))
    scan_source = types.SimpleNamespace(path=shard_dir, key_to_filename_map=None)
    scan_bridge = types.SimpleNamespace(hf_pretrained=types.SimpleNamespace(state=_NoLoadState(scan_source)))
    scan_cache: dict = {}
    assert _get_hf_expected_shape(scan_bridge, "b", scan_cache) == (5, 7)
    assert scan_cache["a"] == (2, 3), "the single scan should cache sibling keys too"
