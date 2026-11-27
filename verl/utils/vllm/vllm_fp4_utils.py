# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""verl's MXFP4 routed-expert feature patch and NVFP4 kernel checks.

vLLM's reload (``start_reload`` / ``finish_reload``) restores the checkpoint
layout of these experts for a refit.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_MXFP4_SF_BLOCK = 32


def is_deepseek_v4_model(model):
    if model is None:
        return False

    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is not None and getattr(obj, "model_type", None) is not None:
            return obj.model_type == "deepseek_v4"

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    return getattr(text_config, "model_type", None) == "deepseek_v4"


def install_mxfp4_mxfp8_compressed_tensors_patch():
    """Route verl's compressed-tensors MXFP4 experts through DeepGEMM.

    vLLM 0.29 recognizes MXFP4 compressed-tensors weights but ignores their
    input-activation scheme and unconditionally constructs its W4A4 method on
    Blackwell. Reuse vLLM's canonical MXFP4 method with DeepGEMM's dynamic FP8
    group-128 activations while retaining the compressed-tensors packed
    parameter names expected by the loader.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
        compressed_tensors_moe_w4a4_mxfp4,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
    from vllm.model_executor.utils import move_parameter

    CompressedTensorsW4A4Mxfp4MoEMethod = compressed_tensors_moe_w4a4_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod
    if getattr(CompressedTensorsMoEMethod, "_verl_mxfp4_mxfp8_patch", False):
        return

    class CompressedTensorsW4A8Mxfp4MoEMethod(Mxfp4MoEMethod):
        """Compressed-tensors MXFP4 weights with DeepGEMM dynamic FP8 activations."""

        def __init__(self, moe):
            FusedMoEMethodBase.__init__(self, moe)
            self.weight_dtype = "mxfp4"
            self.activation_key = kFp8Dynamic128Sym
            self.is_k3_situ_aiter = False
            self.is_k3_situ_int4_gfx942 = False
            self.group_size = _MXFP4_SF_BLOCK
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe, activation_key=kFp8Dynamic128Sym)
            self.max_capture_size = moe.max_capture_size
            self._cache_permute_indices = {}
            self.moe_kernel = None
            self.w13_precision_config = None
            self.w2_precision_config = None

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            CompressedTensorsW4A4Mxfp4MoEMethod.create_weights(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
            self.num_experts = num_experts
            self.intermediate_size = intermediate_size_per_partition
            self.hidden_size = hidden_size

        def process_weights_after_loading(self, layer):
            # the kernel reads wN_weight; the loaded wN_weight_packed stay registered for reloads
            move_parameter(layer, "w13_weight_packed", "w13_weight")
            move_parameter(layer, "w2_weight_packed", "w2_weight")
            return super().process_weights_after_loading(layer)

    original = CompressedTensorsMoEMethod.get_moe_method

    def get_moe_method(quant_config, layer, layer_name):
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, layer_name + projection)
            for projection in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        scheme = schemes[-1]
        if not all(candidate == scheme for candidate in schemes):
            return original(quant_config, layer, layer_name)
        weight_quant = None if scheme is None else scheme.get("weights")
        input_quant = None if scheme is None else scheme.get("input_activations")
        if quant_config._is_mxfp4(weight_quant) and quant_config._is_mxfp8(input_quant):
            logger.info("Using MXFP4 weights with dynamic MXFP8 activations for compressed-tensors MoE")
            return CompressedTensorsW4A8Mxfp4MoEMethod(layer.moe_config)
        return original(quant_config, layer, layer_name)

    CompressedTensorsMoEMethod.get_moe_method = staticmethod(get_moe_method)
    CompressedTensorsMoEMethod._verl_mxfp4_mxfp8_patch = True


def _attest_nvfp4_kernel_scale_references(module):
    """Verify that the retained CUDA-graph kernel sees the current refit scales."""
    method = module.quant_method
    kernel = method.moe_kernel
    config = method.moe_quant_config
    experts = getattr(kernel, "fused_experts", None)
    if experts is None or experts.quant_config is not config:
        raise RuntimeError("nvfp4 retained kernel no longer references its retained quant config")

    for config_name, parameter_name in (
        ("g1_alphas", "w13_weight_scale_2"),
        ("g2_alphas", "w2_weight_scale_2"),
        ("w1_scale", "w13_weight_scale"),
        ("w2_scale", "w2_weight_scale"),
    ):
        actual = getattr(config, config_name, None)
        registered = getattr(module, parameter_name, None)
        if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
            raise RuntimeError(f"nvfp4 kernel scale reference missing: {config_name}/{parameter_name}")
        if actual.device != registered.device or actual.data_ptr() != registered.data_ptr():
            raise RuntimeError(f"nvfp4 kernel has stale scale reference: {config_name}/{parameter_name}")

    for config_name, input_name in (("a1_gscale", "w13_input_scale"), ("a2_gscale", "w2_input_scale")):
        actual = getattr(config, config_name, None)
        input_scale = getattr(module, input_name, None)
        if not isinstance(actual, torch.Tensor) or not isinstance(input_scale, torch.Tensor):
            raise RuntimeError(f"nvfp4 activation scale reference missing: {config_name}/{input_name}")
        if not torch.equal(actual, 1.0 / input_scale):
            raise RuntimeError(f"nvfp4 {config_name} does not match the current activation scale")

    # FlashInfer TRT-LLM derives g1_scale_c from the scales above and keeps its own reference.
    derived = getattr(experts, "g1_scale_c", None)
    if isinstance(derived, torch.Tensor):
        registered = getattr(module, "g1_scale_c", None)
        if not isinstance(registered, torch.Tensor) or derived.data_ptr() != registered.data_ptr():
            raise RuntimeError("nvfp4 kernel has a stale g1_scale_c reference")
        expected = config.a2_gscale
        if experts.moe_config.is_act_and_mul:
            expected = config.g1_alphas * config.a2_gscale
        if not torch.equal(registered, expected):
            raise RuntimeError("nvfp4 g1_scale_c does not match the current scales")
