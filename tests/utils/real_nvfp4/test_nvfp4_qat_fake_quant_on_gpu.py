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

"""NVFP4 QAT fake quantization equals the dequantized rollout quantization bitwise.

Activations are compared with the per-token quantization vLLM applies to the MoE input, which follows
``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH``; run this file with and without it set to ``1``.
"""

import pytest
import torch

# flashinfer_utils is imported by fused_moe, which must load first.
pytest.importorskip("vllm.model_executor.layers.fused_moe")
flashinfer_utils = pytest.importorskip("vllm.model_executor.layers.quantization.utils.flashinfer_utils")

from verl.utils.qat.nvfp4 import (  # noqa: E402
    fake_quantize_nvfp4,
    fake_quantize_nvfp4_per_token,
    nvfp4_global_scale,
    quantize_nvfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4 quantization kernels need a Blackwell (SM100) GPU",
)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequantize(packed: torch.Tensor, block_scale: torch.Tensor, decode: torch.Tensor) -> torch.Tensor:
    packed = packed.view(torch.uint8)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = torch.tensor(_E2M1, device=packed.device)[codes]
    scale = block_scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * (scale * decode)


def _inputs(shape, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    # Rows with different ranges, as tokens have.
    x.mul_(torch.logspace(-3, 2, shape[0], device="cuda", dtype=torch.float32)[:, None].to(dtype))
    x[0, :16] = 0.0  # a zero block
    if shape[1] >= 32:
        x[0, 16:32] = -0.0
    if shape[0] > 1:
        # Values on the E2M1 rounding ties once scaled by the block maximum.
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
        x[1, :16] = torch.cat([ties, -ties]).to(dtype)
    if shape[0] > 2:
        x[2] = 0.0  # a zero token
    if shape[0] > 3 and shape[1] >= 32:
        x[3, :16].mul_(1e-6)  # a block far below the token's maximum
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 16), (5, 48), (130, 272), (512, 4096)])
def test_per_token_fake_quant_matches_the_rollout_moe_input_quantization(dtype, shape):
    x = _inputs(shape, dtype, sum(shape))
    packed, block_scale, per_token_scale = flashinfer_utils.quantize_nvfp4_per_token_input(x)
    expected = _dequantize(packed, block_scale.reshape(shape[0], -1), per_token_scale.reshape(-1, 1).float())
    assert torch.equal(fake_quantize_nvfp4_per_token(x), expected.to(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 16), (5, 48), (130, 272), (512, 4096)])
def test_weight_fake_quant_matches_the_export(dtype, shape):
    x = _inputs(shape, dtype, 2 * sum(shape))
    global_scale = nvfp4_global_scale(x)
    packed, block_scale = quantize_nvfp4(x, global_scale)
    expected = _dequantize(packed, block_scale, 1.0 / global_scale)
    assert torch.equal(fake_quantize_nvfp4(x, global_scale), expected.to(dtype))
