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
"""Apply vLLM 0.29 NVFP4 refit storage compatibility to the installed wheel.

Packing and kernel reuse are upstream in this release. Only postprocessing from
fresh scales and recoverable activation-scale storage remain local.
"""

import hashlib
from importlib.metadata import distribution, version

BEFORE = "52ce69a32611d8eb00fa2a27cb9b2ec46b8836f34f0a1f8b4834990c3ea228a9"
AFTER = "b4aad18cce98895c618abed608dd9cd00594831153e1a1db0a93c8d025218e32"
OLD = """        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=True,
            )

        self.moe_kernel.fused_experts.process_weights_after_loading(layer)"""
NEW = """        # Postprocess with the freshly loaded tensors, not the retained kernel's
        # quant config (which still references the pre-refit tensor storage).
        # The native layerwise loader copies these processed parameters back
        # into the original storage after this method returns. Keep the original
        # kernel and all of its eager/CUDA-graph references untouched on reload.
        processing_quant_config = self.get_fused_moe_quant_config(layer)
        replace_parameter(layer, "nvfp4_a1_gscale", processing_quant_config.a1_gscale)
        replace_parameter(layer, "nvfp4_a2_gscale", processing_quant_config.a2_gscale)
        assert self.experts_cls is not None
        processing_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=processing_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            per_token_activation=True,
        )
        if self.moe_kernel is None:
            self.moe_quant_config = processing_quant_config
            self.moe_kernel = processing_kernel
        processing_kernel.fused_experts.process_weights_after_loading(layer)"""


def patch_source(text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest == AFTER:
        return text
    if digest != BEFORE or text.count(OLD) != 1:
        raise RuntimeError(f"Unexpected vLLM 0.29.0 NVFP4 source: {digest}")
    patched = text.replace(OLD, NEW)
    if hashlib.sha256(patched.encode()).hexdigest() != AFTER:
        raise RuntimeError("Unexpected patched NVFP4 source")
    return patched


def main():
    if version("vllm") != "0.29.0":
        raise RuntimeError("This compatibility adapter supports vLLM 0.29.0")
    path = distribution("vllm").locate_file("vllm/model_executor/layers/quantization/online/nvfp4.py")
    path.write_text(patch_source(path.read_text()))
    print("REAL_NVFP4_VLLM_DERIVED_SCALE_BACKPORT_PASS", path, flush=True)


if __name__ == "__main__":
    main()
