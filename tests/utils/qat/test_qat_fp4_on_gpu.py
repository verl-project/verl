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

from verl.utils.qat.fp4 import fp4_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="MXFP4 quantization needs an SM90 or later GPU",
)

_FP4_LEVELS = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)


def _encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    magnitudes = values.abs().unsqueeze(-1)
    levels = _FP4_LEVELS.to(values.device)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    even_codes = (torch.arange(8, device=values.device) % 2 == 0).view(*((1,) * values.ndim), 8)
    codes = torch.where(candidates & even_codes, 1, 0).argmax(dim=-1)
    no_even_tie = ~(candidates & even_codes).any(dim=-1)
    codes = torch.where(no_even_tie, distances.argmin(dim=-1), codes).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def _reference(x: torch.Tensor, dequant: bool):
    cols = x.shape[-1]
    blocks = x.float().reshape(-1, cols // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = amax.clamp_min(6.0 * (2.0**-126)) / 6.0
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    codes = _encode_e2m1((blocks / scales).clamp(-6.0, 6.0))
    logical = _FP4_LEVELS.to(x.device)[(codes & 0x7).long()]
    logical = torch.where((codes & 0x8) != 0, -logical, logical)
    if dequant:
        return (logical * scales).reshape_as(x).to(x.dtype)

    flat_codes = codes.reshape(*x.shape[:-1], cols)
    packed = flat_codes[..., 0::2] | (flat_codes[..., 1::2] << 4)
    quantized = packed.contiguous().view(torch.float4_e2m1fn_x2)
    scale_shape = (*x.shape[:-1], cols // 32)
    return quantized, scales.squeeze(-1).reshape(scale_shape).to(torch.float8_e8m0fnu)


def test_fp4_quant_matches_reference_and_fused_dequant():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn((2, 3, 64), device="cuda", dtype=torch.bfloat16, generator=generator)
    x[0].zero_()
    x[1, 0, 7] = 500.0
    ties = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0), device="cuda")
    x[1, 1] = ties.repeat(8).to(torch.bfloat16)

    quantized, scales = fp4_quant(x)
    dequantized = fp4_quant(x, dequant=True)
    expected_quantized, expected_scales = _reference(x, dequant=False)
    expected_dequantized = _reference(x, dequant=True)

    assert quantized.shape == (2, 3, 32)
    assert quantized.dtype == torch.float4_e2m1fn_x2
    assert scales.shape == (2, 3, 2)
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(dequantized, expected_dequantized)
