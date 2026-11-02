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
import torch.nn.functional as F

from verl.utils.qat.fp8 import fp8_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 quantization needs an SM90 or later GPU",
)


def _reference(
    x: torch.Tensor,
    block_shape: tuple[int, int],
    *,
    round_scale: bool,
    eps: float = 1e-12,
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
    if round_scale:
        bits = amax.view(torch.int32)
        scales = ((bits + (1 << 21)) & 0xFF800000).view(torch.float32) * (2.0**-8)
        scales = scales.clamp_min(2.0**-127)
    else:
        scales = amax.clamp_min(eps) / 448.0
        scales = torch.where(amax == 0, torch.full_like(scales, 2.0**-23), scales)
    quantized = (
        (blocks / scales.unsqueeze(-1).unsqueeze(-3))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(*x.shape[:-2], rows + pad_rows, cols + pad_cols)[..., :rows, :cols]
        .contiguous()
    )
    return quantized, scales


@pytest.mark.parametrize("block_shape", [(128, 128), (32, 32), (1, 128), (1, 32)])
def test_fp8_quant_matches_reference(block_shape):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn((2, 256, 256), device="cuda", dtype=torch.bfloat16, generator=generator)

    quantized, scales = fp8_quant(x, block_shape)
    expected_quantized, expected_scales = _reference(x, block_shape, round_scale=False)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == expected_scales.shape
    assert scales.dtype == torch.float32
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


def test_fp8_quant_e8m0_scale_and_fused_dequant_match_reference():
    generator = torch.Generator(device="cuda").manual_seed(5678)
    x = torch.randn((3, 65, 96), device="cuda", dtype=torch.bfloat16, generator=generator)
    block_shape = (1, 32)

    quantized, scales = fp8_quant(
        x,
        block_shape,
        scale_dtype=torch.float8_e8m0fnu,
        scale_fmt="e8m0",
    )
    dequantized = fp8_quant(x, block_shape, scale_fmt="e8m0", dequant=True)
    expected_quantized, expected_scales = _reference(x, block_shape, round_scale=True)
    expected_dequantized = (
        (expected_quantized.float().unflatten(-1, (-1, block_shape[1])) * expected_scales.unsqueeze(-1))
        .flatten(-2)
        .to(x.dtype)
    )

    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(scales.float(), expected_scales)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(dequantized, expected_dequantized)
