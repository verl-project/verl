# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real TE module smoke tests; not train/vLLM numerical equivalence tests."""

import pytest
import torch

from verl.utils.real_nvfp4.config import validate_real_nvfp4_te_recipe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a scheduled Blackwell GPU")
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("fp4", [False, True])
def test_release_forward_backward_and_weight_update(grouped, fp4):
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    torch.manual_seed(1234)
    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
        nvfp4_4over6="none",
    )
    validate_real_nvfp4_te_recipe(recipe, backward_override="dequantized")
    kwargs = dict(in_features=256, out_features=128, bias=False, params_dtype=torch.bfloat16, device="cuda")
    layer = te.GroupedLinear(num_gemms=2, **kwargs) if grouped else te.Linear(**kwargs)
    params = list(layer.parameters())
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.SGD(params, lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with te.autocast(enabled=fp4, recipe=recipe):
            y = layer(x, [64, 64]) if grouped else layer(x)
        if fp4:
            assert isinstance(layer.fp8_meta["recipe"], NVFP4BlockScaling)
        assert torch.isfinite(y).all()
        y.float().square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for p in params:
            assert p.dtype == torch.bfloat16
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.count_nonzero(p.grad) > 0
        optimizer.step()
    torch.cuda.synchronize()
    assert all(not torch.equal(old, new) for old, new in zip(before, params, strict=True))
