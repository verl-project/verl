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

"""The trainer-side per-token NVFP4 export equals vLLM's online nvfp4_per_token quantization."""

import pytest
import torch

pytest.importorskip("compressed_tensors")
nvfp4 = pytest.importorskip("vllm.model_executor.layers.quantization.online.nvfp4")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    NVFP4PerTokenWeightQuantizer,
    build_compressed_tensors_config,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="vLLM's NVFP4 quantization kernel needs a Blackwell (SM100) GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_export_matches_vllm_online_quantization_bitwise(dtype):
    generator = torch.Generator(device="cuda").manual_seed(17029)
    num_experts, hidden, intermediate = 4, 256, 128
    w13 = torch.randn((num_experts, 2 * intermediate, hidden), device="cuda", dtype=dtype, generator=generator)
    w2 = torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=dtype, generator=generator)
    # Zero, tiny and large experts, and gate/up halves with different ranges.
    for weight in (w13, w2):
        weight[0].zero_()
        weight[1].mul_(0.01)
        weight[2].mul_(10)
    w13[3, :intermediate].mul_(3)

    prefix = "model.layers.0.mlp.experts"
    stream = []
    for expert in range(num_experts):
        gate, up = w13[expert].split(intermediate)
        stream += [
            (f"{prefix}.{expert}.gate_proj.weight", gate.clone()),
            (f"{prefix}.{expert}.down_proj.weight", w2[expert].clone()),
            (f"{prefix}.{expert}.up_proj.weight", up.clone()),
        ]
    config = build_compressed_tensors_config(
        linear=None, moe="NVFP4", moe_activation="nvfp4_per_token", ignore=[], config_file=None
    )
    out = dict(NVFP4PerTokenWeightQuantizer(config)(stream))

    for stacked, projections in ((w13, ("gate_proj", "up_proj")), (w2, ("down_proj",))):
        packed, block_scale, weight_scale_2 = nvfp4._quantize_moe_weight_to_nvfp4(stacked)
        for expert in range(num_experts):
            names = [f"{prefix}.{expert}.{projection}" for projection in projections]
            got_packed = torch.cat([out[f"{name}.weight_packed"] for name in names])
            got_scale = torch.cat([out[f"{name}.weight_scale"] for name in names])
            assert torch.equal(got_packed, packed[expert])
            assert got_scale.dtype == torch.float8_e4m3fn
            assert torch.equal(got_scale.view(torch.uint8), block_scale[expert].view(torch.uint8))
            for name in names:
                # vLLM's compressed-tensors loader stores the reciprocal of the exported global scale.
                assert torch.equal(1.0 / out[f"{name}.weight_global_scale"], weight_scale_2[expert].reshape(1))
                assert torch.equal(out[f"{name}.input_global_scale"].cpu(), torch.ones(1))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="vLLM's NVFP4 quantization kernel needs a Blackwell (SM100) GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 16), (3, 48), (130, 272), (512, 4096)])
def test_engine_neutral_quantizer_matches_vllm_bitwise(dtype, shape):
    from vllm._custom_ops import scaled_fp4_quant

    from verl.utils.qat.nvfp4 import quantize_nvfp4

    generator = torch.Generator(device="cuda").manual_seed(sum(shape))
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    x[0, :16] = 0.0  # a zero block
    if shape[1] >= 32:
        x[0, 16:32] = -0.0
    if shape[0] > 1:
        # Values on the E2M1 rounding ties once scaled by the block maximum.
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
        x[1, :16] = torch.cat([ties, -ties]).to(dtype)
    if shape[0] > 2:
        x[2].mul_(1e-3)
    for amax in (x.float().abs().amax(), torch.tensor(0.37, device="cuda")):
        global_scale = (6.0 * 448.0 / amax.clamp_min(1e-8)).reshape(1).float()
        packed, scale = quantize_nvfp4(x, global_scale)
        expected_packed, expected_scale = scaled_fp4_quant(x, global_scale, is_sf_swizzled_layout=False)
        assert torch.equal(packed, expected_packed)
        assert torch.equal(scale.view(torch.uint8), expected_scale.view(torch.uint8))
