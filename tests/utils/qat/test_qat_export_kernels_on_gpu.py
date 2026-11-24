# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

pytest.importorskip("compressed_tensors")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    CompressedTensorsWeightQuantizer,
    _compress_weight,
    _compress_weight_with_compressed_tensors,
    _scheme_for_preset,
    _unpack_experts,
    build_compressed_tensors_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="TileLang FP8/FP4 export kernels need an SM90 or later GPU",
)


@pytest.mark.parametrize(
    ("preset", "shape"),
    [
        ("FP8_BLOCK", (256, 256)),
    ],
)
def test_export_kernel_matches_compressed_tensors_bitwise(preset, shape):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight[0].zero_()
    weight[1].mul_(1e-8)
    scheme = _scheme_for_preset(preset, ["Linear"])

    actual = _compress_weight(weight, scheme)
    expected = _compress_weight_with_compressed_tensors(weight, scheme)

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))


@pytest.mark.parametrize(
    ("preset", "scale_name", "expected_scale_bytes"),
    [
        ("MXFP8", "weight_scale", (127, 127, 128, 129)),
        ("MXFP4", "weight_scale", (134, 134, 134, 135)),
    ],
)
def test_mx_export_kernel_uses_deepseek_rounding(preset, scale_name, expected_scale_bytes):
    weight = (
        torch.tensor((400.0, 448.0, 500.0, 1000.0), device="cuda", dtype=torch.bfloat16)
        .unsqueeze(-1)
        .expand(-1, 32)
        .contiguous()
    )
    scheme = _scheme_for_preset(preset, ["Linear"])

    actual = _compress_weight(weight, scheme)

    assert torch.equal(
        actual[scale_name].reshape(-1),
        torch.tensor(expected_scale_bytes, device="cuda", dtype=torch.uint8),
    )


def test_nvfp4_export_keeps_compressed_tensors_path(monkeypatch):
    weight = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    scheme = _scheme_for_preset("NVFP4", ["Linear"])
    expected = _compress_weight_with_compressed_tensors(weight, scheme)
    calls = []

    def fallback(*args, **kwargs):
        calls.append(True)
        return expected

    monkeypatch.setattr("verl.utils.qat.compressed_tensors._compress_weight_with_compressed_tensors", fallback)

    actual = _compress_weight(weight, scheme)

    assert calls == [True]
    assert actual is expected


@pytest.mark.parametrize("preset", ["FP8_BLOCK", "MXFP8", "MXFP4"])
@pytest.mark.parametrize(
    ("projection", "shape"),
    [
        ("gate_up_proj", (2, 256, 256)),
        ("down_proj", (2, 256, 256)),
    ],
)
def test_packed_experts_use_one_group_quant_call_and_match_per_expert(monkeypatch, preset, projection, shape):
    config = build_compressed_tensors_config(
        linear=None,
        moe=preset,
        ignore=[],
        config_file=None,
    )
    quantizer = CompressedTensorsWeightQuantizer(config)
    name = f"model.layers.0.mlp.experts.{projection}"
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    scheme = quantizer._module_scheme("model.layers.0.mlp.experts.0.gate_proj")
    real_compress_weight = _compress_weight
    calls = []

    def record_compress_weight(tensor, scheme, **kwargs):
        calls.append(tensor.shape)
        return real_compress_weight(tensor, scheme, **kwargs)

    monkeypatch.setattr("verl.utils.qat.compressed_tensors._compress_weight", record_compress_weight)
    actual = dict(quantizer([(name, weight)]))
    assert calls == [torch.Size(shape)]

    expected = {}
    for weight_name, expert_weight in _unpack_experts(name, weight):
        prefix = weight_name[: -len("weight")]
        for key, value in real_compress_weight(expert_weight, scheme).items():
            expected[prefix + key] = value

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))


@pytest.mark.parametrize("preset", ["FP8_BLOCK", "MXFP8", "MXFP4"])
@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w3.weight",
    ],
)
def test_per_expert_export_keeps_names_and_uses_moe_scheme(preset, name):
    config = build_compressed_tensors_config(
        linear=None,
        moe=preset,
        ignore=[],
        config_file=None,
    )
    quantizer = CompressedTensorsWeightQuantizer(config)
    weight = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    scheme = quantizer._module_scheme(name.removesuffix(".weight"))

    actual = dict(quantizer([(name, weight)]))
    expected = {name[: -len("weight")] + key: value for key, value in _compress_weight(weight, scheme).items()}

    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))


def test_fused_expert_export_matches_non_mlp_moe_container():
    config = build_compressed_tensors_config(
        linear=None,
        moe="MXFP4",
        ignore=[],
        config_file=None,
    )
    quantizer = CompressedTensorsWeightQuantizer(config)
    name = "language_model.model.layers.0.block_sparse_moe.experts.down_proj"
    weight = torch.randn((2, 256, 256), device="cuda", dtype=torch.bfloat16)

    actual = dict(quantizer([(name, weight)]))

    assert set(actual) == {
        f"language_model.model.layers.0.block_sparse_moe.experts.{expert}.down_proj.{suffix}"
        for expert in range(2)
        for suffix in ("weight_packed", "weight_scale")
    }


def test_fp8_packed_gate_up_rejects_scale_block_crossing_projection():
    config = build_compressed_tensors_config(
        linear=None,
        moe="FP8_BLOCK",
        ignore=[],
        config_file=None,
    )
    quantizer = CompressedTensorsWeightQuantizer(config)
    name = "model.layers.0.mlp.experts.gate_up_proj"
    weight = torch.randn((2, 384, 256), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must be divisible by FP8 block rows"):
        dict(quantizer([(name, weight)]))


def test_fp8_export_rejects_unaligned_weight_shape():
    scheme = _scheme_for_preset("FP8_BLOCK", ["Linear"])
    weight = torch.randn((130, 257), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="not divisible by block_shape"):
        _compress_weight(weight, scheme)
