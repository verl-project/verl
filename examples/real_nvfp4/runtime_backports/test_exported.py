# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise the actual patched package, retaining the original as reference."""

import os
from contextlib import contextmanager
from importlib.metadata import version

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.cpp_extensions import gemm
from transformer_engine.pytorch.module import grouped_linear


@contextmanager
def implementation(function):
    old_gemm, old_layer = gemm.general_grouped_gemm, grouped_linear.general_grouped_gemm
    gemm.general_grouped_gemm = grouped_linear.general_grouped_gemm = function
    try:
        yield
    finally:
        gemm.general_grouped_gemm, grouped_linear.general_grouped_gemm = (
            old_gemm,
            old_layer,
        )


def check_differential(original, exported, recipe):
    """Compare the installed implementation with its saved upstream function."""
    for name, dim_in, dim_out in (("fc1", 2048, 1536), ("fc2", 768, 2048)):
        for split_name, splits in (
            ("uniform64", [64] * 32),
            ("uniform256", [256] * 32),
            ("uniform1024", [1024] * 32),
            ("ragged_zero", [0, 16, 64, 176, 256, 512, 768, 256] * 4),
        ):
            torch.manual_seed(20260915)
            layer = te.GroupedLinear(
                num_gemms=32,
                in_features=dim_in,
                out_features=dim_out,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
            x = torch.randn((sum(splits), dim_in), dtype=torch.bfloat16, device="cuda", requires_grad=True)
            dy = torch.randn((sum(splits), dim_out), dtype=torch.bfloat16, device="cuda")
            with implementation(original), torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
                layer(x, splits, is_first_microbatch=True)
            for weight_version in range(2):
                if weight_version:
                    with torch.no_grad():
                        for parameter in layer.parameters():
                            parameter.add_(0.001)
                reference = None
                for function in (original, exported):
                    layer.zero_grad(set_to_none=True)
                    x.grad = None
                    with implementation(function), te.autocast(enabled=True, recipe=recipe):
                        y = layer(x, splits, is_first_microbatch=True)
                    y.backward(dy)
                    tensors = [y.detach(), x.grad, *(p.grad for p in layer.parameters())]
                    assert all(t is not None and torch.isfinite(t).all() for t in tensors)
                    if reference is None:
                        reference = [t.clone() for t in tensors]
                    else:
                        for actual, expected in zip(tensors, reference, strict=True):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                print("DIFFERENTIAL_PASS", name, split_name, weight_version, flush=True)
                del reference, tensors, y
            del layer, x, dy


def main():
    assert version("transformer-engine") == "2.18.0"
    assert gemm.NVFP4_ROWSCALE_GROUPED_BACKPORT == "20260915-v1"
    exported = gemm.general_grouped_gemm
    original = gemm._nvfp4_rowscale_original_grouped_gemm
    assert grouped_linear.general_grouped_gemm is exported
    assert exported.__module__ == gemm.__name__ and original is not exported
    if os.environ.get("EXPORT_GUARD_ONLY") == "1":
        print("EXPORTED_TE_ROWSCALE_GUARD_PASS", flush=True)
        return
    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
        nvfp4_4over6="none",
    )
    assert torch.cuda.device_count() == 1
    counters = gemm._nvfp4_rowscale_backport_counters
    check_differential(original, exported, recipe)
    for label, enabled, bias, magnitude, splits in (
        ("bf16_fallback", False, False, 1.0, [256] * 32),
        ("bias_fallback", True, True, 1.0, [256] * 32),
        ("empty_fallback", True, False, 1.0, [0] * 32),
        ("zero_activations", True, False, 0.0, [256] * 32),
        ("small_activations", True, False, 0.001, [256] * 32),
        ("large_activations", True, False, 1000.0, [256] * 32),
    ):
        torch.manual_seed(918)
        layer = te.GroupedLinear(32, 2048, 1536, bias=bias, params_dtype=torch.bfloat16, device="cuda")
        x = (torch.randn((sum(splits), 2048), device="cuda", dtype=torch.bfloat16) * magnitude).requires_grad_()
        dy = torch.randn((sum(splits), 1536), device="cuda", dtype=torch.bfloat16)
        refs = None
        before = dict(counters)
        for function in (original, exported):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            with implementation(function), te.autocast(enabled=enabled, recipe=recipe):
                y = layer(x, splits, is_first_microbatch=True)
                y.backward(dy)
            tensors = [y.detach(), x.grad, *(p.grad for p in layer.parameters())]
            assert all(t is not None and torch.isfinite(t).all() for t in tensors)
            if refs is None:
                refs = [t.clone() for t in tensors]
            else:
                for actual, expected in zip(tensors, refs, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if label.endswith("fallback"):
            assert counters["batched"] == before["batched"]
        else:
            assert counters["batched"] > before["batched"]
        print("EXPORTED_EDGE_PASS", label, flush=True)
    assert counters["batched"] > 0 and counters["fallback"] > 0
    print("EXPORTED_TE_ROWSCALE_TEST_PASS", flush=True)


if __name__ == "__main__":
    main()
