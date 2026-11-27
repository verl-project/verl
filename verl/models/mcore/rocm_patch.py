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
"""Megatron-Core patches for AMD ROCm."""

# Transformer Engine on ROCm (gfx950) runs MXFP8 GEMMs on HipKittens only when every token or feature dimension is a
# multiple of 256 (K a multiple of 128). Megatron pads each routed expert's tokens to
# get_align_size_for_quantization(), which is get_fp8_align_size()'s 32 for MXFP8 (the NVIDIA requirement). An expert
# GEMM whose token count misses 256 falls back to hipBLASLt, which has no prebuilt kernel for the MXFP8 scale layout
# TE passes, so rocRoller generates one on the CPU at run time (minutes per new shape); below 128, TE's K % 128 check
# also fails in the expert weight-gradient GEMM.
ROCM_MXFP8_EXPERT_TOKEN_ALIGN = 256


def apply_patch_rocm_mxfp8_expert_alignment():
    """Pad routed-expert tokens to 256 for MXFP8 on ROCm (no-op elsewhere; idempotent).

    get_align_size_for_quantization() looks get_fp8_align_size up in moe_utils' namespace at call time, so wrapping
    that one name changes the token padding of every MoE caller (token dispatchers, expert padding) and nothing else:
    other users of get_fp8_align_size, e.g. GatedDeltaNet's in_proj width check, keep the 32 because that width is
    not a token count. Takes effect with moe_router_padding_for_fp8 (moe_router_padding_for_quantization).
    """
    import torch

    if torch.version.hip is None:
        return
    from megatron.core.transformer.moe import moe_utils

    original = moe_utils.get_fp8_align_size
    if getattr(original, "_verl_rocm_mxfp8_expert_align", False):
        return

    def get_fp8_align_size(fp8_recipe):
        if getattr(fp8_recipe, "value", fp8_recipe) == "mxfp8":
            return ROCM_MXFP8_EXPERT_TOKEN_ALIGN
        return original(fp8_recipe)

    get_fp8_align_size._verl_rocm_mxfp8_expert_align = True
    moe_utils.get_fp8_align_size = get_fp8_align_size
