# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json

import torch

from verl.checkpoint_engine.delta_sync import checksum
from verl.utils.fp8_sharded import QuantSpec, quantize_hf_stream, quantize_mxfp4_e2m1
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


def test_mxfp4_e2m1_known_codes_and_stream_layout():
    row = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        * 2,
        dtype=torch.bfloat16,
    ).reshape(1, 32)
    packed, scales = quantize_mxfp4_e2m1(row)
    expected = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE] * 2, dtype=torch.uint8)
    assert packed.dtype == torch.int8
    assert torch.equal(packed.view(torch.uint8).reshape(-1), expected)
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.shape == (1, 1)
    assert float(scales.float().item()) == 1.0

    spec = QuantSpec(
        weight_block_size=(128, 128),
        should_quantize=lambda _name: False,
        mxfp4_predicate=lambda name: ".experts." in name,
    )
    out = list(quantize_hf_stream([("model.layers.1.mlp.experts.0.w1.weight", row)], spec))
    assert [name for name, _ in out] == [
        "model.layers.1.mlp.experts.0.w1.weight",
        "model.layers.1.mlp.experts.0.w1.weight_scale_inv",
    ]
    assert out[0][1].shape == (1, 16)
    assert out[1][1].shape == (1, 1)


class Mxfp4FakeMethod:
    def process_weights_after_loading(self, layer):
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            data = getattr(layer, name).data.flip(-1).contiguous()
            if name.endswith("scale_inv"):
                data = data.to(torch.float8_e4m3fn)
            setattr(layer, name, torch.nn.Parameter(data, requires_grad=False))


class Fp8FakeMethod:
    is_fp4_expert = True

    def process_weights_after_loading(self, _layer):
        raise AssertionError("checkpoint-layout FP4 experts must not be refit")


class _FakeMxfp4Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = self._num_local_routed = 2
        self.quant_method = Mxfp4FakeMethod()
        self.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.int8), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.int8), requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self.weight_loader

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return int(expert_id) if 0 <= int(expert_id) < self.num_local_experts else -1

    def weight_loader(self, param, loaded_weight, _weight_name, shard_id, expert_id):
        dst = param.data[int(expert_id)]
        if shard_id == "w1":
            dst[: dst.shape[0] // 2].copy_(loaded_weight)
        elif shard_id == "w3":
            dst[dst.shape[0] // 2 :].copy_(loaded_weight)
        else:
            dst.copy_(loaded_weight)


class _FakeMixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _FakeMxfp4Moe()
        self.post_load_calls = 0

    def load_weights(self, chunk):
        for name, tensor in chunk:
            _, expert, shard, kind = name.split(".")
            attr = {
                ("w1", "weight"): "w13_weight",
                ("w3", "weight"): "w13_weight",
                ("w2", "weight"): "w2_weight",
                ("w1", "scale"): "w13_weight_scale_inv",
                ("w3", "scale"): "w13_weight_scale_inv",
                ("w2", "scale"): "w2_weight_scale_inv",
            }[(shard, kind)]
            param = getattr(self.moe, attr)
            param.weight_loader(param, tensor, name, shard, int(expert))

    def post_load_weights(self):
        self.post_load_calls += 1


def _dense_mixed_flush(entries, *, is_last):
    params, pieces, offset = [], [], 0
    for name, tensor in entries:
        raw = tensor.contiguous().view(torch.uint8).reshape(-1)
        params.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": offset,
                "val_end": offset + raw.numel(),
            }
        )
        pieces.append(raw)
        offset += raw.numel()
    values = torch.cat(pieces)
    empty = torch.empty(0, dtype=torch.uint8)
    spec = {
        "encoding": "dense",
        "is_last": is_last,
        "values_bytes": True,
        "quant_config": {"expert_dtype": "fp4"},
        "params": params,
        "checksum": int(checksum(empty, values)),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    return [("__delta_spec__", spec_t), ("__values__", values)]


def test_mxfp4_dense_refit_spans_flushes_and_preserves_live_parameter_addresses():
    model = _FakeMixedModel()
    pointers = {name: p.data_ptr() for name, p in model.moe.named_parameters()}
    raw = {}
    entries_by_expert = []
    for expert in range(2):
        entries = []
        for shard, shape in (("w1", (2, 2)), ("w3", (2, 2)), ("w2", (4, 2))):
            weight = torch.arange(torch.tensor(shape).prod(), dtype=torch.int8).reshape(shape) + expert * 17
            scale = torch.full((shape[0], 1), float(expert + 1), dtype=torch.float32)
            entries.extend([(f"expert.{expert}.{shard}.weight", weight), (f"expert.{expert}.{shard}.scale", scale)])
            raw[(expert, shard, "weight")] = weight
            raw[(expert, shard, "scale")] = scale
        entries_by_expert.append(entries)

    apply_delta(model, _dense_mixed_flush(entries_by_expert[0], is_last=False))
    assert hasattr(model, "_verl_delta_mxfp4_refit")
    apply_delta(model, _dense_mixed_flush(entries_by_expert[1], is_last=True))
    assert not hasattr(model, "_verl_delta_mxfp4_refit")
    assert model.post_load_calls == 1
    assert {name: p.data_ptr() for name, p in model.moe.named_parameters()} == pointers

    w13 = torch.stack(
        [torch.cat([raw[(e, "w1", "weight")], raw[(e, "w3", "weight")]], dim=0) for e in range(2)]
    )
    w2 = torch.stack([raw[(e, "w2", "weight")] for e in range(2)])
    s13 = torch.stack(
        [torch.cat([raw[(e, "w1", "scale")], raw[(e, "w3", "scale")]], dim=0) for e in range(2)]
    )
    s2 = torch.stack([raw[(e, "w2", "scale")] for e in range(2)])
    assert torch.equal(model.moe.w13_weight, w13.flip(-1))
    assert torch.equal(model.moe.w2_weight, w2.flip(-1))
    assert torch.equal(model.moe.w13_weight_scale_inv.view(torch.uint8), s13.to(torch.float8_e4m3fn).flip(-1).view(torch.uint8))
    assert torch.equal(model.moe.w2_weight_scale_inv.view(torch.uint8), s2.to(torch.float8_e4m3fn).flip(-1).view(torch.uint8))


def test_incomplete_mxfp4_refit_restores_live_parameters():
    model = _FakeMixedModel()
    originals = {name: param for name, param in model.moe.named_parameters()}
    incomplete = [("expert.0.w1.weight", torch.ones(2, 2, dtype=torch.int8))]

    try:
        apply_delta(model, _dense_mixed_flush(incomplete, is_last=True))
    except RuntimeError as exc:
        assert "incomplete values-only MXFP4 refit" in str(exc)
    else:
        raise AssertionError("an incomplete MXFP4 refit must fail closed")

    assert not hasattr(model, "_verl_delta_mxfp4_refit")
    for name, param in model.moe.named_parameters():
        assert param is originals[name]


def test_fp8_method_marked_fp4_loads_checkpoint_layout_directly():
    model = _FakeMixedModel()
    model.moe.quant_method = Fp8FakeMethod()
    pointers = {name: p.data_ptr() for name, p in model.moe.named_parameters()}
    entries = []
    for expert in range(2):
        for shard, shape in (("w1", (2, 2)), ("w3", (2, 2)), ("w2", (4, 2))):
            weight = torch.full(shape, expert + 3, dtype=torch.int8)
            scale = torch.full((shape[0], 1), float(expert + 1), dtype=torch.float32)
            entries.extend([(f"expert.{expert}.{shard}.weight", weight), (f"expert.{expert}.{shard}.scale", scale)])

    apply_delta(model, _dense_mixed_flush(entries, is_last=True))
    assert model.post_load_calls == 1
    assert {name: p.data_ptr() for name, p in model.moe.named_parameters()} == pointers
    assert torch.all(model.moe.w13_weight[0] == 3)
    assert torch.all(model.moe.w13_weight[1] == 4)
