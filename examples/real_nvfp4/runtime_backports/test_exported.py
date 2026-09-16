"""Exercise the actual patched package, retaining the original as reference."""

import os
from contextlib import contextmanager
from importlib.metadata import version

import candidate
import probe
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
    # Reuse the exact successful 2828300 differential/timing workload. Replace
    # only the tested function with the actual function loaded from the image.
    candidate.ORIGINAL = original
    candidate.batched_row_scaled_gemm = exported
    candidate.COUNTERS = gemm._nvfp4_rowscale_backport_counters
    probe.COUNTERS = candidate.COUNTERS
    with implementation(original):
        probe.main()

    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
        nvfp4_4over6="none",
    )
    for label, enabled, bias, magnitude, splits in (
        ("bf16_fallback", False, False, 1.0, [256] * 32),
        ("bias_fallback", True, True, 1.0, [256] * 32),
        ("empty_fallback", True, False, 1.0, [0] * 32),
        ("zero_activations", True, False, 0.0, [256] * 32),
        ("small_activations", True, False, 0.001, [256] * 32),
        ("large_activations", True, False, 1000.0, [256] * 32),
    ):
        torch.manual_seed(918)
        layer = te.GroupedLinear(
            32, 2048, 1536, bias=bias, params_dtype=torch.bfloat16, device="cuda"
        )
        x = (
            torch.randn((sum(splits), 2048), device="cuda", dtype=torch.bfloat16)
            * magnitude
        ).requires_grad_()
        dy = torch.randn((sum(splits), 1536), device="cuda", dtype=torch.bfloat16)
        refs = None
        before = dict(candidate.COUNTERS)
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
            assert candidate.COUNTERS["batched"] == before["batched"]
        else:
            assert candidate.COUNTERS["batched"] > before["batched"]
        print("EXPORTED_EDGE_PASS", label, flush=True)
    print("EXPORTED_TE_ROWSCALE_TEST_PASS", flush=True)


if __name__ == "__main__":
    main()
