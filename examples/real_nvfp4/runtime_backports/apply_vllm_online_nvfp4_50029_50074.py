#!/usr/bin/env python3
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
"""Backport the two post-v0.26 online-NVFP4 fixes needed by Verl refit.

The replacements intentionally match the vLLM v0.26.0 source exactly.  Refuse
to modify an unexpected dependency version instead of producing a partial
backport.
"""

from importlib.metadata import distribution, version

PACKING_COMMIT = "9c22668436a4d94aab87ea74a220e060415cf1d8"
RELOAD_COMMIT = "3ac9525507b2d0de5c1b08cbca96cc94850c7c7a"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        assert old not in text, f"{label}: old and new implementations coexist"
        return text
    count = text.count(old)
    assert count == 1, f"{label}: expected one v0.26 source block, found {count}"
    return text.replace(old, new)


def main() -> None:
    assert version("vllm") == "0.26.0", version("vllm")
    path = distribution("vllm").locate_file("vllm/model_executor/layers/quantization/online/nvfp4.py")
    text = path.read_text()

    old_packing = """    num_experts, n, k = weight.shape
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight.abs().amax(dim=(1, 2)).to(torch.float32).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # scaled_fp4_quant(w, g) == scaled_fp4_quant(w * g, 1), so fold each
    # expert's scale in and quantize all experts in one call (fp32 to keep the
    # large scale precise), rather than looping per expert.
    scaled = (weight.float() * global_scale[:, None, None]).to(weight.dtype)
    scaled = scaled.reshape(-1, k)
    one = torch.ones((), device=weight.device, dtype=torch.float32)
    qweight, block_scale = scaled_fp4_quant(scaled, one, is_sf_swizzled_layout=False)
    return (
        qweight.reshape(num_experts, n, k // 2),
        block_scale.reshape(num_experts, n, k // 16),
        weight_scale_2,
    )"""
    new_packing = """    k = weight.shape[-1]
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight.abs().amax(dim=(1, 2)).to(torch.float32).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # Keep the original BF16/FP16 values as the quantizer input. Folding each
    # expert's FP32 global scale into the weight would add a BF16/FP16 rounding
    # before the group-16 scale and E2M1 values are selected.
    weight = weight.contiguous()
    quantized_experts = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_scale,
            strict=True,
        )
    ]
    qweight = torch.stack([quantized for quantized, _ in quantized_experts])
    block_scale = torch.stack([block_scale for _, block_scale in quantized_experts])
    return (
        qweight,
        block_scale,
        weight_scale_2,
    )"""
    text = replace_once(text, old_packing, new_packing, "NVFP4 expert packing")

    old_reload = """        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
            per_token_activation=True,
        )"""
    new_reload = """        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
                per_token_activation=True,
            )"""
    text = replace_once(text, old_reload, new_reload, "NVFP4 reload kernel reuse")

    path.write_text(text)
    verified = path.read_text()
    assert "quantized_experts = [" in verified
    assert "scaled = scaled.reshape(-1, k)" not in verified
    assert "if self.moe_kernel is None:" in verified
    print(
        "REAL_NVFP4_VLLM_EXPERT_PACKING_BACKPORT_PASS",
        f"upstream={PACKING_COMMIT}",
        path,
        flush=True,
    )
    print(
        "REAL_NVFP4_VLLM_RELOAD_KERNEL_BACKPORT_PASS",
        f"upstream={RELOAD_COMMIT}",
        path,
        flush=True,
    )


if __name__ == "__main__":
    main()
