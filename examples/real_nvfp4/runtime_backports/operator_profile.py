"""Scheduled single-GPU mechanism tests, NOT an end-to-end training benchmark.

Real Qwen3 expert dimensions and the production TE recipe. Synthetic uniform
token splits isolate local compute; distributed imbalance/communication require
the separate, unchanged-topology actor traces. Never deploy cached=False across
optimizer updates: it is only a same-weight subsequent-microbatch diagnostic.
"""

import gc
import json
import os
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from torch.profiler import ProfilerActivity, profile, record_function
from transformer_engine.common.recipe import NVFP4BlockScaling

from verl.utils.real_nvfp4.config import validate_real_nvfp4_te_recipe
from verl.utils.real_nvfp4.vllm_runtime import require_vllm_nvfp4_backports


def measure(call, repeats=7):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - start))
    return {"wall_ms_median": statistics.median(values), "wall_ms_samples": values}


def trace(call, out, name):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        call()
        torch.cuda.synchronize()
    rows = [
        {
            "name": event.key,
            "count": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "device_us": event.device_time_total,
            "input_shapes": event.input_shapes,
        }
        for event in prof.key_averages(group_by_input_shape=True)
    ]
    (out / f"{name}.operators.json").write_text(json.dumps(rows, indent=2))
    prof.export_chrome_trace(str(out / f"{name}.trace.json"))
    print(
        "OPERATOR_TRACE",
        name,
        json.dumps(sorted(rows, key=lambda x: x["self_device_us"], reverse=True)[:12]),
        flush=True,
    )


def main():
    out = Path(os.environ["EFFICIENCY_OUTPUT"])
    out.mkdir(parents=True, exist_ok=False)
    assert (
        torch.cuda.device_count() == 1 and torch.cuda.get_device_capability()[0] == 10
    )
    require_vllm_nvfp4_backports()
    config = json.loads(Path(os.environ["MODEL_PATH"], "config.json").read_text())
    assert (
        config["hidden_size"],
        config["moe_intermediate_size"],
        config["num_experts"],
    ) == (2048, 768, 128)
    print(
        "OPERATOR_ENV",
        json.dumps(
            {
                "versions": {
                    k: version(k) for k in ("torch", "transformer-engine", "vllm")
                },
                "gpu": torch.cuda.get_device_name(),
                "grouped_fused_env": os.getenv(
                    "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0"
                ),
                "local_experts": 32,
                "warning": "synthetic splits; not full Megatron or communication",
            }
        ),
        flush=True,
    )
    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
        nvfp4_4over6="none",
    )
    validate_real_nvfp4_te_recipe(recipe, backward_override="dequantized")
    results = []
    for projection, in_features, out_features in (
        ("fc1", 2048, 1536),
        ("fc2", 768, 2048),
    ):
        for tokens in (64, 256, 1024):
            # Same seed and values per precision/cache variant. Fresh modules avoid cross-mode metadata reuse.
            for precision, cache in (
                ("bf16", "none"),
                ("nvfp4", "none"),
                ("nvfp4", "refresh"),
                ("nvfp4", "reuse"),
            ):
                for mode in ("forward", "forward_backward"):
                    torch.manual_seed(20260915)
                    layer = te.GroupedLinear(
                        num_gemms=32,
                        in_features=in_features,
                        out_features=out_features,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        device="cuda",
                    )
                    backward = mode == "forward_backward"
                    x = torch.randn(
                        (32 * tokens, in_features),
                        dtype=torch.bfloat16,
                        device="cuda",
                        requires_grad=backward,
                    )
                    dy = torch.randn(
                        (32 * tokens, out_features), dtype=torch.bfloat16, device="cuda"
                    )
                    fp4 = precision == "nvfp4"
                    # Build the reusable workspace once. Weights remain unchanged throughout this fixture.
                    if cache == "reuse":
                        with torch.no_grad(), te.autocast(enabled=fp4, recipe=recipe):
                            layer(x, [tokens] * 32, is_first_microbatch=True)

                    def call(
                        layer=layer,
                        x=x,
                        dy=dy,
                        backward=backward,
                        fp4=fp4,
                        cache=cache,
                        tokens=tokens,
                    ):
                        layer.zero_grad(set_to_none=True)
                        x.grad = None
                        with torch.set_grad_enabled(backward):
                            with (
                                record_function("expert_forward"),
                                te.autocast(enabled=fp4, recipe=recipe),
                            ):
                                y = layer(
                                    x,
                                    [tokens] * 32,
                                    is_first_microbatch={
                                        "none": None,
                                        "refresh": True,
                                        "reuse": False,
                                    }[cache],
                                )
                            if backward:
                                with record_function("expert_backward"):
                                    y.backward(dy)
                        return y

                    y = call()
                    assert torch.isfinite(y).all()
                    if backward:
                        assert torch.isfinite(x.grad).all()
                        assert all(
                            p.grad is not None and torch.isfinite(p.grad).all()
                            for p in layer.parameters()
                        )
                    label = f"{projection}_m{tokens}_{precision}_{cache}_{mode}"
                    row = dict(
                        projection=projection,
                        tokens_per_expert=tokens,
                        local_experts=32,
                        precision=precision,
                        cache=cache,
                        mode=mode,
                        **measure(call),
                    )
                    results.append(row)
                    print("OPERATOR_RESULT", json.dumps(row), flush=True)
                    if tokens == 256:
                        trace(call, out, label)
                    del y, call, layer, x, dy
                    gc.collect()
                    torch.cuda.empty_cache()

    # Actual native receiver packing: no IPC/broadcast, kernel-layout conversion or model loading.
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        _quantize_moe_weight_to_nvfp4,
    )

    for projection, shape in (("w13", (128, 1536, 2048)), ("w2", (128, 2048, 768))):
        torch.manual_seed(123)
        weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        def pack(weight=weight):
            with record_function("native_receiver_packing"):
                return _quantize_moe_weight_to_nvfp4(weight)

        row = dict(
            component="native_receiver_packing",
            projection=projection,
            shape=shape,
            **measure(pack, repeats=5),
        )
        results.append(row)
        print("PACKING_RESULT", json.dumps(row), flush=True)
        trace(pack, out, f"receiver_{projection}")
        del pack, weight
        gc.collect()
        torch.cuda.empty_cache()
    (out / "measurements.json").write_text(json.dumps(results, indent=2))
    assert len(results) == 50
    print("EFFICIENCY_OPERATOR_PASS", len(results), flush=True)


if __name__ == "__main__":
    main()
