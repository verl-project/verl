"""Single-process scheduled differential and timing probe, no training claims."""

import gc
import json
import os
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from candidate import COUNTERS, installed
from operator_profile import measure, trace
from transformer_engine.common.recipe import NVFP4BlockScaling


def main():
    assert version("transformer-engine") == "2.18.0"
    assert torch.cuda.device_count() == 1
    out = Path(os.environ["EFFICIENCY_OUTPUT"])
    out.mkdir(parents=True, exist_ok=False)
    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
        nvfp4_4over6="none",
    )
    results = []
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
            x = torch.randn(
                (sum(splits), dim_in),
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            dy = torch.randn(
                (sum(splits), dim_out), dtype=torch.bfloat16, device="cuda"
            )
            with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
                layer(x, splits, is_first_microbatch=True)

            def call(
                backward=True, first=False, layer=layer, x=x, dy=dy, splits=splits
            ):
                layer.zero_grad(set_to_none=True)
                x.grad = None
                with (
                    torch.set_grad_enabled(backward),
                    te.autocast(enabled=True, recipe=recipe),
                ):
                    y = layer(x, splits, is_first_microbatch=first)
                if backward:
                    y.backward(dy)
                return y

            # Compare exactly the same inputs, weights and incoming gradient.
            # Strict bitwise equality initially; a failed comparison is not a PASS.
            for weight_version in range(2):
                if weight_version:
                    with torch.no_grad():
                        for parameter in layer.parameters():
                            parameter.add_(0.001)
                ref = call(first=True).detach().clone()
                dx_ref = x.grad.clone()
                dw_ref = [p.grad.clone() for p in layer.parameters()]
                with installed():
                    candidate = call(first=True)
                torch.testing.assert_close(candidate, ref, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, dx_ref, rtol=0, atol=0)
                for p, expected in zip(layer.parameters(), dw_ref, strict=True):
                    torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
                assert torch.isfinite(candidate).all() and torch.isfinite(x.grad).all()
                print("DIFFERENTIAL_PASS", name, split_name, weight_version, flush=True)
                del candidate, ref, dx_ref, dw_ref
            row = {"projection": name, "split": split_name}
            for backward in (False, True):
                mode = "forward_backward" if backward else "forward"
                for optimized in (False, True):
                    arm = "candidate" if optimized else "original"
                    with installed() if optimized else nullcontext():
                        row[f"{mode}_{arm}"] = measure(
                            lambda call=call, backward=backward: call(backward=backward)
                        )
                        if split_name == "uniform256":
                            trace(
                                lambda call=call, backward=backward: call(
                                    backward=backward
                                ),
                                out,
                                f"{name}_{mode}_{arm}",
                            )
            results.append(row)
            print("BATCHED_RESULT", json.dumps(row), flush=True)
            del call, layer, x, dy
            gc.collect()
            torch.cuda.empty_cache()
    assert COUNTERS["batched"] > 0 and COUNTERS["fallback"] > 0
    (out / "results.json").write_text(
        json.dumps({"results": results, "counters": COUNTERS}, indent=2)
    )
    print("GROUPED_ROWSCALE_PASS", len(results), COUNTERS, flush=True)


if __name__ == "__main__":
    main()
