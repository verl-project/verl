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

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from verl.utils.qat import fp8

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FP8_PATH = _REPO_ROOT / "verl/utils/qat/fp8.py"


class _FakeGpuTensor(torch.Tensor):
    """CPU tensor subclass reporting ``is_cuda == True`` to exercise GPU-gated export paths."""

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

    @property
    def is_cuda(self) -> bool:
        return True

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        kwargs = {} if kwargs is None else kwargs
        unwrap = lambda a: a.as_subclass(torch.Tensor) if isinstance(a, cls) else a
        return super().__torch_function__(
            func,
            tuple(torch.Tensor if issubclass(t, cls) else t for t in types),
            torch.utils._pytree.tree_map(unwrap, args),
            torch.utils._pytree.tree_map(unwrap, kwargs),
        )


def _reference(
    x: torch.Tensor,
    block_shape: tuple[int, int],
    *,
    round_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_rows, block_cols = block_shape
    rows, cols = x.shape[-2:]
    pad_rows, pad_cols = -rows % block_rows, -cols % block_cols
    padded = F.pad(x.float(), (0, pad_cols, 0, pad_rows))
    blocks = padded.reshape(
        *x.shape[:-2],
        (rows + pad_rows) // block_rows,
        block_rows,
        (cols + pad_cols) // block_cols,
        block_cols,
    )
    amax = blocks.abs().amax(dim=(-3, -1))
    scales = amax.clamp_min(1e-4) / 448.0
    if round_scale:
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    quantized = (
        (blocks / scales.unsqueeze(-1).unsqueeze(-3))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(*x.shape[:-2], rows + pad_rows, cols + pad_cols)[..., :rows, :cols]
        .contiguous()
    )
    return quantized, scales


def test_fp8_module_imports_without_tilelang(monkeypatch):
    monkeypatch.setitem(sys.modules, "tilelang", None)
    monkeypatch.setitem(sys.modules, "tilelang.language", None)

    spec = importlib.util.spec_from_file_location("verl_fp8_without_tilelang", _FP8_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.tilelang is None
    assert module.pass_configs == {}
    assert not module._use_tilelang()


@pytest.mark.parametrize("use_hip", [False, True])
@pytest.mark.parametrize("block_shape", [(128, 128), (32, 32), (1, 128), (1, 32)])
def test_fp8_quant_pytorch_fallback_matches_reference(monkeypatch, use_hip, block_shape):
    if use_hip:
        monkeypatch.setattr(torch.version, "hip", "7.14.0")
    else:
        monkeypatch.setattr(fp8, "tilelang", None)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn((2, 256, 256), dtype=torch.bfloat16, generator=generator)

    quantized, scales = fp8.fp8_quant(_FakeGpuTensor(x), block_shape)
    expected_quantized, expected_scales = _reference(x, block_shape, round_scale=False)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == expected_scales.shape
    assert scales.dtype == torch.float32
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.parametrize("scale_fmt", ["e8m0", "ue8m0"])
def test_fp8_quant_fallback_e8m0_and_dequant_match_reference(monkeypatch, scale_fmt):
    monkeypatch.setattr(fp8, "tilelang", None)

    edge_x = torch.tensor((400.0, 448.0, 500.0, 1000.0)).unsqueeze(-1).expand(-1, 32).contiguous()
    _, edge_scales = fp8.fp8_quant(
        _FakeGpuTensor(edge_x),
        (1, 32),
        scale_dtype=torch.float8_e8m0fnu,
        scale_fmt=scale_fmt,
    )
    assert torch.equal(edge_scales.float(), torch.tensor(((1.0,), (1.0,), (2.0,), (4.0,))))

    subnormal_x = torch.stack(
        [
            torch.zeros(128),
            torch.full((128,), 1e-8),
            torch.linspace(-3.0, 3.0, 128),
        ]
    ).to(dtype=torch.bfloat16)
    quantized, scales = fp8.fp8_quant(
        _FakeGpuTensor(subnormal_x),
        (1, 128),
        scale_dtype=torch.float8_e8m0fnu,
        scale_fmt=scale_fmt,
    )
    dequantized = fp8.fp8_quant(_FakeGpuTensor(subnormal_x), (1, 128), scale_fmt=scale_fmt, dequant=True)
    expected_quantized, expected_scales = _reference(subnormal_x, (1, 128), round_scale=True)
    expected_dequantized = (expected_quantized.float() * expected_scales).to(subnormal_x.dtype)

    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(scales.float(), expected_scales)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(dequantized, expected_dequantized)


@pytest.mark.parametrize(
    ("block_shape", "scale_fmt"),
    [
        ((128, 128), None),
        ((1, 32), "e8m0"),
    ],
)
def test_fp8_grouped_fake_quant_fallback_matches_per_expert(monkeypatch, block_shape, scale_fmt):
    monkeypatch.setattr(fp8, "tilelang", None)

    generator = torch.Generator(device="cpu").manual_seed(42)
    experts = [torch.randn((256, 256), dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    fake_experts = [_FakeGpuTensor(expert) for expert in experts]

    grouped = fp8.fp8_grouped_fake_quant(fake_experts, block_shape, scale_fmt=scale_fmt)
    expected = torch.stack(
        [fp8.fp8_quant(expert, block_shape, scale_fmt=scale_fmt, dequant=True) for expert in fake_experts],
        dim=0,
    )

    assert torch.equal(grouped, expected)


def test_compress_weight_fp8_block_without_tilelang_matches_compressed_tensors_bitwise(monkeypatch):
    pytest.importorskip("compressed_tensors")
    from verl.utils.qat.compressed_tensors import (
        _compress_weight,
        _compress_weight_with_compressed_tensors,
        _scheme_for_preset,
    )

    monkeypatch.setattr(fp8, "tilelang", None)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    weight = torch.randn((256, 256), dtype=torch.bfloat16, generator=generator)
    weight[0].zero_()
    weight[1].mul_(1e-8)
    scheme = _scheme_for_preset("FP8_BLOCK", ["Linear"])

    actual = _compress_weight(_FakeGpuTensor(weight), scheme)
    expected = _compress_weight_with_compressed_tensors(weight, scheme)

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))


@pytest.mark.parametrize("preset", ["FP8_BLOCK", "MXFP8"])
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_packed_experts_export_without_tilelang_matches_per_expert(monkeypatch, preset, projection):
    pytest.importorskip("compressed_tensors")
    from verl.utils.qat.compressed_tensors import (
        CompressedTensorsWeightQuantizer,
        Fp8ConfigWeightQuantizer,
        _compress_weight,
        _unpack_experts,
        build_compressed_tensors_config,
    )

    monkeypatch.setattr(fp8, "tilelang", None)
    config = build_compressed_tensors_config(linear=None, moe=preset, ignore=[], config_file=None)
    quantizer = CompressedTensorsWeightQuantizer(config)
    name = f"model.layers.0.mlp.experts.{projection}"
    generator = torch.Generator(device="cpu").manual_seed(99)
    weight = torch.randn((2, 256, 256), dtype=torch.bfloat16, generator=generator)
    scheme = quantizer._module_scheme("model.layers.0.mlp.experts.0.gate_proj")
    assert scheme is not None

    actual = dict(quantizer([(name, _FakeGpuTensor(weight))]))
    expected = {}
    for weight_name, expert_weight in _unpack_experts(name, weight):
        prefix = weight_name[: -len("weight")]
        for key, value in _compress_weight(_FakeGpuTensor(expert_weight), scheme).items():
            expected[prefix + key] = value

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))

    if preset == "FP8_BLOCK":
        native_quantizer = Fp8ConfigWeightQuantizer(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        )
        native_actual = dict(native_quantizer([(name, _FakeGpuTensor(weight))]))
        native_expected = {
            (k[: -len("weight_scale")] + "weight_scale_inv" if k.endswith(".weight_scale") else k): v
            for k, v in expected.items()
        }
        assert native_actual.keys() == native_expected.keys()
        for key in native_actual:
            assert torch.equal(native_actual[key].view(torch.uint8), native_expected[key].view(torch.uint8))


def test_fp8_export_without_tilelang_rejects_unaligned_weight_shape(monkeypatch):
    pytest.importorskip("compressed_tensors")
    from verl.utils.qat.compressed_tensors import _compress_weight, _scheme_for_preset

    monkeypatch.setattr(fp8, "tilelang", None)
    scheme = _scheme_for_preset("FP8_BLOCK", ["Linear"])
    weight = torch.randn((130, 257), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="not divisible by block_shape"):
        _compress_weight(_FakeGpuTensor(weight), scheme)
