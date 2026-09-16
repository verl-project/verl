"""Actual vLLM function tests. Run only in the scheduled Blackwell job."""

import pytest
import torch
from vllm.model_executor.layers.quantization.online import nvfp4


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
def test_native_packing_matches_per_expert_gpu_reference(dtype, strided):
    generator = torch.Generator(device="cuda").manual_seed(17029)
    weight = torch.randn((4, 64, 128), device="cuda", dtype=dtype, generator=generator)
    if strided:
        weight = weight.transpose(1, 2)
    # Include zero and heterogeneous expert ranges; avoid a single global-scale fixture.
    weight[0].zero_()
    weight[1].mul_(0.01)
    weight[2].mul_(10)
    original = weight.clone()
    actual = nvfp4._quantize_moe_weight_to_nvfp4(weight)
    scale = (6.0 * 448.0) / weight.abs().amax(dim=(1, 2)).float().clamp_min(1e-8)
    reference = [
        nvfp4.scaled_fp4_quant(w.contiguous(), s, is_sf_swizzled_layout=False)
        for w, s in zip(weight, scale, strict=True)
    ]
    expected = (
        torch.stack([q for q, _ in reference]),
        torch.stack([s for _, s in reference]),
        scale.reciprocal(),
    )
    assert torch.equal(weight, original), (
        "packing must not mutate the incoming BF16 weights"
    )
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    assert torch.isfinite(actual[1].float()).all()
    assert torch.isfinite(actual[2]).all()
