# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""The rollout exporter compresses one layer's experts one projection per call under group strategies.

The batched export must give the same tensors, bit for bit and in the same order, as compressing every weight on its
own, which is what the exporter did before; only the number of compression calls changes.
"""

import importlib.util

import pytest
import torch

import verl.utils.qat.compressed_tensors as ct

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("compressed_tensors") is None, reason="compressed-tensors is not installed"
)

HIDDEN, FFN = 64, 32
PROJS = (("gate_proj", (FFN, HIDDEN)), ("up_proj", (FFN, HIDDEN)), ("down_proj", (HIDDEN, FFN)))


def _quantizer(moe, linear=None):
    config = ct.build_compressed_tensors_config(linear=linear, moe=moe, ignore=[], config_file=None)
    return ct.CompressedTensorsWeightQuantizer(config)


def _expert_weight(generator, shape, i):
    weight = torch.randn(shape, generator=generator) * [1e-3, 0.05, 1.0, 30.0][i % 4]
    if i % 5 == 0:
        weight[0] = 0
    if i % 7 == 0:
        weight[1, :32] = 6.0  # a group at the largest E2M1 magnitude
    return weight.to(torch.bfloat16)


def _weights(layers=2, experts=4, seed=0):
    """Expert-major order, as Megatron-Bridge exports grouped experts, with dense weights between the layers."""
    generator = torch.Generator().manual_seed(seed)
    out = []
    i = 0
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        out.append((f"{prefix}.self_attn.q_proj.weight", torch.randn(HIDDEN, HIDDEN, generator=generator).bfloat16()))
        out.append((f"{prefix}.mlp.gate.weight", torch.randn(experts, HIDDEN, generator=generator).bfloat16()))
        for expert in range(experts):
            for proj, shape in PROJS:
                out.append((f"{prefix}.mlp.experts.{expert}.{proj}.weight", _expert_weight(generator, shape, i)))
                i += 1
    out.append(("lm_head.weight", torch.randn(128, HIDDEN, generator=generator).bfloat16()))
    return out


def _one_by_one(quantizer, weights):
    """The exporter before batching: every 2D weight compressed on its own, in input order."""
    out = []
    for name, tensor in weights:
        scheme = quantizer._scheme_for(name, tensor)
        if scheme is None:
            out.append((name, tensor))
            continue
        for key, value in ct._compress_weight(tensor, scheme).items():
            out.append((name[: -len("weight")] + key, value))
        input_scale = quantizer._input_global_scale(name[: -len(".weight")], scheme)
        if input_scale is not None:
            out.append((name[: -len("weight")] + "input_global_scale", input_scale))
    return out


def _assert_same(actual, expected):
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (name, a), (_, b) in zip(actual, expected, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name


def _count_calls(monkeypatch):
    calls = []
    original = ct._compress_weight

    def counted(weight, scheme, **kwargs):
        calls.append(tuple(weight.shape))
        return original(weight, scheme, **kwargs)

    monkeypatch.setattr(ct, "_compress_weight", counted)
    return calls


@pytest.mark.parametrize(("moe", "linear"), [("MXFP4", None), ("MXFP8", None), ("MXFP8", "MXFP8"), ("MXFP4", "MXFP8")])
def test_batched_export_matches_one_by_one(moe, linear):
    quantizer = _quantizer(moe, linear)
    weights = _weights()
    _assert_same(list(quantizer(iter(weights))), _one_by_one(quantizer, weights))


def test_one_call_per_projection_and_layer(monkeypatch):
    calls = _count_calls(monkeypatch)
    out = list(_quantizer("MXFP4")(iter(_weights(layers=2, experts=4))))
    assert calls == [(4 * FFN, HIDDEN), (4 * FFN, HIDDEN), (4 * HIDDEN, FFN)] * 2
    assert sum(name.endswith(".weight_packed") for name, _ in out) == 2 * 4 * 3


def test_runs_are_capped(monkeypatch):
    quantizer = _quantizer("MXFP4")
    weights = _weights(layers=1, experts=8)
    expected = _one_by_one(quantizer, weights)
    # room for three expert weights: the 24 expert weights split into runs of three
    monkeypatch.setattr(ct, "_EXPERT_BATCH_BYTES", 3 * FFN * HIDDEN * 2)
    calls = _count_calls(monkeypatch)
    _assert_same(list(quantizer(iter(weights))), expected)
    assert len(calls) == 8 * 3


def test_exporter_refilling_one_buffer(monkeypatch):
    """An exporter may yield views of a buffer it refills for the next weights (VeOmni's expert broadcast)."""
    quantizer = _quantizer("MXFP4")
    weights = _weights(layers=1, experts=4)
    expected = _one_by_one(quantizer, weights)
    buffers = {}

    def refilled():
        for name, tensor in weights:
            buffer = buffers.setdefault(tuple(tensor.shape), torch.empty_like(tensor))
            buffer.copy_(tensor)
            yield name, buffer

    _assert_same(list(quantizer(refilled())), expected)


def test_packed_experts_match_one_by_one(monkeypatch):
    quantizer = _quantizer("MXFP4")
    generator = torch.Generator().manual_seed(1)
    gate_up = torch.stack([_expert_weight(generator, (2 * FFN, HIDDEN), i) for i in range(4)])
    down = torch.stack([_expert_weight(generator, (HIDDEN, FFN), i) for i in range(4)])
    packed = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]
    expected = _one_by_one(quantizer, [item for name, t in packed for item in ct._unpack_experts(name, t)])
    calls = _count_calls(monkeypatch)
    _assert_same(list(quantizer(iter(packed))), expected)
    assert len(calls) == 3


def test_nvfp4_experts_stay_one_by_one(monkeypatch):
    # NVFP4 scales against a per-weight global scale, which the rows of other experts would change
    quantizer = _quantizer("NVFP4")
    weights = _weights(layers=1, experts=2)
    expected = _one_by_one(quantizer, weights)
    calls = _count_calls(monkeypatch)
    _assert_same(list(quantizer(iter(weights))), expected)
    assert len(calls) == 2 * 3
