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

import logging
from typing import Callable, Optional
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)
from torch.nn import Parameter


def generate_nvfp4_ignore_list(num_layers: int, is_moe: bool) -> list[str]:
    """
    Generate the ignore list for NVFP4 quantization based on model configuration.
    
    Args:
        num_layers: Number of hidden layers in the model (from hf_config.num_hidden_layers)
        is_moe: Whether the model is a Mixture of Experts model
        
    Returns:
        List of layer names to ignore during quantization
    """
    ignore_list = []
    
    # For MoE models, ignore the gate layers (routing layers)
    if is_moe:
        for layer_idx in range(num_layers):
            ignore_list.append(f"model.layers.{layer_idx}.mlp.gate")
    
    # Always ignore lm_head for stability
    ignore_list.append("lm_head")
    
    return ignore_list


def get_nvfp4_block_quant_kwargs(num_layers: int, is_moe: bool) -> dict:
    """
    Generate complete NVFP4 quantization configuration based on model properties.
    Args:
        num_layers: Number of hidden layers in the model (from hf_config.num_hidden_layers)
        is_moe: Whether the model is a Mixture of Experts model
        
    Returns:
        Complete quantization configuration dictionary compatible with ModelOpt
    """
    ignore_list = generate_nvfp4_ignore_list(num_layers, is_moe)
    
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": "false",
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16
                },
                "weights": {
                    "dynamic": "false",
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16
                },
                "targets": [
                    "Linear"
                ]
            }
        },
        "ignore": ignore_list,
        "quant_algo": "NVFP4",
        "producer": {
            "name": "modelopt",
        },
        "quant_method": "modelopt"
    }



def _create_param_from_subclass_attributes(custom_data: torch.Tensor, custom_weight) -> Parameter:
    """
    Helper to preserve custom attributes from ModelWeightParameter and
    PerTensorScaleParameter when creating new Parameters.
    """
    param = Parameter(custom_data, requires_grad=False)
    base_param_dir = dir(torch.nn.Parameter)
    custom_weight_dir = dir(custom_weight)
    # Find the attributes that are unique to the custom parameter
    custom_attributes = [attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")]
    # Set the custom attributes into the base parameter object
    for attr in custom_attributes:
        setattr(param, attr, getattr(custom_weight, attr))
    return param


def process_weights_after_loading_modelopt(self, layer: torch.nn.Module) -> None:
    import vllm._custom_ops as ops
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_bias,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        param = Parameter(custom_data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_weight_dir = dir(custom_weight)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_weight, attr))

        return param

    def prepare_fp4_layer_for_marlin(layer: torch.nn.Module, weight_scale_2_max: torch.Tensor) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression will "
            "be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )

        is_nvfp4 = hasattr(layer, "weight_scale_2")
        group_size = 16 if is_nvfp4 else 32

        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        param_dtype = layer.params_dtype

        assert layer.weight.shape == (part_size_n, part_size_k // 2)

        device = layer.weight.device

        # WORKSPACE
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace_new(device)

        # WEIGHT
        # Repack weights to marlin format
        perm = torch.empty(0, dtype=torch.int, device=device)
        qweight = layer.weight.view(torch.int32).T.contiguous()

        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=part_size_k,
            size_n=part_size_n,
            num_bits=4,
        )
        layer.marlin_weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

        # WEIGHT SCALES
        # Permute scales
        weight_scale = layer.weight_scale.T.contiguous()

        if not is_nvfp4:
            weight_scale = weight_scale.view(torch.float8_e8m0fnu)

        weight_scale = weight_scale.to(param_dtype)
        weight_scale = marlin_permute_scales(
            s=weight_scale, size_k=part_size_k, size_n=part_size_n, group_size=group_size
        )

        if is_nvfp4:
            weight_scale = nvfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

            weight_scale_2 = weight_scale_2_max.to(param_dtype)
            weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2)
            layer.marlin_weight_scale_2 = torch.nn.Parameter(weight_scale_2, requires_grad=False)
        else:
            weight_scale = mxfp4_marlin_process_scales(weight_scale)
            layer.marlin_weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        if hasattr(layer, "bias") and layer.bias is not None:
            assert layer.bias.shape == (part_size_n,)
            bias = marlin_permute_bias(layer.bias)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        return

    # global scales:
    input_scale_2 = layer.input_scale.data
    layer.input_scale = _create_param_from_subclass_attributes(input_scale_2, layer.input_scale)
    input_scale_2_max = input_scale_2.max().to(torch.float32)

    weight_scale_2 = layer.weight_scale_2.data
    layer.weight_scale_2 = _create_param_from_subclass_attributes(weight_scale_2, layer.weight_scale_2)
    weight_scale_2_max = weight_scale_2.max().to(torch.float32)

    layer.alpha = Parameter(input_scale_2_max * weight_scale_2_max, requires_grad=False)

    # Calculate `1 / input_scale` so that we don't need to do so at runtime
    layer.input_scale_inv = Parameter((1 / layer.input_scale).to(torch.float32), requires_grad=False)

    # Swizzle the weight blockscale.
    # contracting dimension is input dimension
    # block_size = 16;
    assert layer.weight_scale.dtype == torch.float8_e4m3fn, "Weight Block scale must be represented as FP8-E4M3"

    if self.backend == "marlin":
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        prepare_fp4_layer_for_marlin(layer, weight_scale_2_max)

        del layer.alpha
        # del layer.input_scale
    elif self.backend == "flashinfer-trtllm":
        # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
        # FlashInfer provides nvfp4_quantize to quantize + shuffle the
        # layout but we use our own quantization so we have to call
        # shuffles ourselves.
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        epilogue_tile_m = 128
        weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
        weight_scale = (
            shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale.shape)
            .view(torch.float8_e4m3fn)
        )

        layer.weight_scale = _create_param_from_subclass_attributes(weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(weight, layer.weight)
    else:
        swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
        layer.weight_scale = _create_param_from_subclass_attributes(swizzled_weight_scale, layer.weight_scale)
        layer.weight = _create_param_from_subclass_attributes(layer.weight.data, layer.weight)

def apply_modelopt(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    if self.backend == "marlin":
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.marlin_weight,
            weight_scale=layer.marlin_weight_scale,
            weight_scale_2=layer.marlin_weight_scale_2,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    output_dtype = x.dtype
    output_shape = [x.shape[0], layer.weight.shape[0]]

    # quantize BF16 or FP16 to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_scale_inv)

    # validate dtypes of quantized input, input block scale,
    # weight and weight_blockscale
    assert x_fp4.dtype == torch.uint8
    assert layer.weight.dtype == torch.uint8
    assert x_blockscale.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e4m3fn
    assert layer.alpha.dtype == torch.float32

    mm_args = (
        x_fp4,
        layer.weight,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        output_dtype,
    )
    if self.backend == "flashinfer-trtllm":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="trtllm")
    elif self.backend == "flashinfer-cutlass":
        out = flashinfer_scaled_fp4_mm(*mm_args, backend="cutlass")
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    if bias is not None:
        out = out + bias
    return out.view(*output_shape)


# =============================================================================
# ModelOptNvFp4FusedMoE Patches
# =============================================================================


def process_weights_after_loading_moe(self, layer: torch.nn.Module) -> None:
    """
    Patched process_weights_after_loading for ModelOptNvFp4FusedMoE.

    Key modifications compared to original:
    1. Preserves original weights in separate attributes (marlin_w13_weight, etc.)
    2. Uses _create_param_from_subclass_attributes to preserve parameter metadata
    3. Computes weight_scale_2_max before processing for Marlin
    """
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        FlashinferMoeBackend,
        is_flashinfer_supporting_global_sf,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale

    def prepare_moe_fp4_layer_for_marlin_patched(
        layer: torch.nn.Module,
        w13_weight_scale_2_per_expert: torch.Tensor,
        w2_weight_scale_2_per_expert: torch.Tensor,
    ) -> None:
        """
        Modified prepare_moe_fp4_layer_for_marlin that:
        1. Takes per-expert weight_scale_2 values (not max!)
        2. Saves to marlin_* attributes instead of overwriting originals

        Args:
            w13_weight_scale_2_per_expert: shape (num_experts,) - per-expert scales
            w2_weight_scale_2_per_expert: shape (num_experts,) - per-expert scales
        """
        logger.warning("Using patched prepare_moe_fp4_layer_for_marlin for NVFP4 MoE")

        group_size = 16  # NVFP4 uses group_size=16

        e = layer.num_experts
        k = layer.hidden_size
        n = layer.intermediate_size_per_partition

        device = layer.w13_weight.device
        param_dtype = layer.params_dtype

        # WORKSPACE
        if getattr(layer, "workspace", None) is None:
            layer.workspace = marlin_make_workspace_new(device, 4)

        perm = torch.empty(0, dtype=torch.int, device=device)

        # WEIGHT - Repack weights to marlin format
        for name in ["w13_weight", "w2_weight"]:
            weight = getattr(layer, name)
            tensor_list = []
            if "w13" in name:
                size_n, size_k = n * 2, k
            else:
                size_n, size_k = k, n

            assert weight.shape == (e, size_n, size_k // 2), (
                f"Weight shape mismatch for {name}: expected {(e, size_n, size_k // 2)}, got {weight.shape}"
            )

            for i in range(e):
                qweight = weight[i].view(torch.int32).T.contiguous()

                marlin_qweight = ops.gptq_marlin_repack(
                    b_q_weight=qweight,
                    perm=perm,
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4,
                )
                tensor_list.append(marlin_qweight)

            marlin_weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
            marlin_weight = Parameter(marlin_weight, requires_grad=False)

            # Save to marlin_* attribute instead of overwriting original
            marlin_attr_name = "marlin_" + name
            setattr(layer, marlin_attr_name, marlin_weight)

        # WEIGHT SCALES - Permute scales
        for name, weight_scale_2_per_expert in [
            ("w13", w13_weight_scale_2_per_expert),
            ("w2", w2_weight_scale_2_per_expert),
        ]:
            scales = getattr(layer, name + "_weight_scale")
            scales = scales.to(param_dtype)

            # Convert per-expert global scale to param_dtype
            global_scale = weight_scale_2_per_expert.to(param_dtype)

            tensor_list = []
            if "w13" in name:
                size_n, size_k = n * 2, k
            else:
                size_n, size_k = k, n

            for i in range(e):
                scale = scales[i].T

                marlin_scales = marlin_permute_scales(
                    s=scale,
                    size_k=size_k,
                    size_n=size_n,
                    group_size=group_size,
                )
                marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
                tensor_list.append(marlin_scales)

            marlin_scales_combined = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
            marlin_scales_combined = Parameter(marlin_scales_combined, requires_grad=False)

            # Save to marlin_* attribute
            setattr(layer, "marlin_" + name + "_weight_scale", marlin_scales_combined)

            # Process per-expert global scale (shape: num_experts)
            global_scale = nvfp4_marlin_process_global_scale(global_scale)
            global_scale = Parameter(global_scale, requires_grad=False)
            setattr(layer, "marlin_" + name + "_weight_scale_2", global_scale)

    # ========== Main processing logic ==========

    # GEMM 1 processing
    gemm1_weight = layer.w13_weight.data
    gemm1_weight_scale = layer.w13_weight_scale.data

    if (
        self.allow_flashinfer
        and (
            self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS
            or self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        )
        and self.moe.is_act_and_mul
    ):
        gemm1_weight, gemm1_weight_scale = reorder_w1w3_to_w3w1(gemm1_weight, gemm1_weight_scale, dim=-2)

    layer.w13_weight = _create_param_from_subclass_attributes(gemm1_weight, layer.w13_weight)
    layer.w13_weight_scale = _create_param_from_subclass_attributes(gemm1_weight_scale, layer.w13_weight_scale)

    # Common processing for w13_weight_scale_2
    # IMPORTANT: Keep the original shape (num_experts, 2) for subsequent weight loading
    # Only compute the max value for Marlin, but don't modify the original parameter shape
    if self.moe.is_act_and_mul and not torch.allclose(layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]):
        logger.warning("w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.")

    # Keep original data and shape - DO NOT reduce dimension!
    w13_weight_scale_2_data = layer.w13_weight_scale_2.data  # Keep original shape: (num_experts, 2)
    layer.w13_weight_scale_2 = _create_param_from_subclass_attributes(w13_weight_scale_2_data, layer.w13_weight_scale_2)
    # Get per-expert scales (shape: num_experts) for Marlin - NOT the max!
    # This is what the original code uses after reducing [:, 0]
    w13_weight_scale_2_per_expert = layer.w13_weight_scale_2[:, 0].clone()
    # Also keep a 1D version for g1_alphas calculation (following original logic)
    w13_weight_scale_2_1d = layer.w13_weight_scale_2[:, 0]

    # Common processing for input scales and alphas
    # IMPORTANT: Keep original input_scale shapes for subsequent weight loading
    use_global_sf = self.allow_flashinfer and is_flashinfer_supporting_global_sf(self.flashinfer_moe_backend)

    # Keep original w13_input_scale data and shape
    w13_input_scale_data = layer.w13_input_scale.data
    layer.w13_input_scale = _create_param_from_subclass_attributes(w13_input_scale_data, layer.w13_input_scale)

    # Compute derived values for runtime use
    if use_global_sf:
        w13_input_scale_for_alpha = layer.w13_input_scale.max().to(torch.float32).expand(layer.num_experts)
    else:
        w13_input_scale_for_alpha = layer.w13_input_scale.max(dim=1).values.to(torch.float32)

    layer.g1_alphas = Parameter(
        (w13_input_scale_for_alpha * w13_weight_scale_2_1d).to(torch.float32),
        requires_grad=False,
    )

    # This is for quantization, so we need to invert it.
    layer.w13_input_scale_quant = Parameter((1 / w13_input_scale_for_alpha).to(torch.float32), requires_grad=False)

    # GEMM 2 processing
    # Keep original w2_weight_scale_2 data and shape
    w2_weight_scale_2_data = layer.w2_weight_scale_2.data
    layer.w2_weight_scale_2 = _create_param_from_subclass_attributes(w2_weight_scale_2_data, layer.w2_weight_scale_2)
    # Get per-expert scales (shape: num_experts) for Marlin - NOT the max!
    w2_weight_scale_2_per_expert = layer.w2_weight_scale_2.clone()

    # Keep original w2_input_scale data and shape
    w2_input_scale_data = layer.w2_input_scale.data
    layer.w2_input_scale = _create_param_from_subclass_attributes(w2_input_scale_data, layer.w2_input_scale)

    if use_global_sf:
        w2_input_scale_for_alpha = layer.w2_input_scale.max().to(torch.float32).expand(layer.num_experts)
    else:
        w2_input_scale_for_alpha = layer.w2_input_scale
    layer.g2_alphas = Parameter(
        (w2_input_scale_for_alpha * layer.w2_weight_scale_2).to(torch.float32),
        requires_grad=False,
    )

    # This is for quantization, so we need to invert it.
    layer.w2_input_scale_quant = Parameter((1 / w2_input_scale_for_alpha).to(torch.float32), requires_grad=False)

    # ========== Backend-specific processing ==========

    if self.allow_flashinfer and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
        # TensorRT-LLM specific processing
        (
            gemm1_weights_fp4_shuffled,
            gemm1_scales_fp4_shuffled,
            gemm2_weights_fp4_shuffled,
            gemm2_scales_fp4_shuffled,
        ) = prepare_static_weights_for_trtllm_fp4_moe(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w2_weight.size(-2),  # hidden_size
            layer.w13_weight.size(-2) // 2,  # intermediate_size
            layer.w13_weight.size(0),  # num_experts
        )
        logger.debug("Finished shuffling weights for TRT-LLM MOE")

        layer.gemm1_weights_fp4_shuffled = Parameter(gemm1_weights_fp4_shuffled, requires_grad=False)
        layer.gemm2_weights_fp4_shuffled = Parameter(gemm2_weights_fp4_shuffled, requires_grad=False)
        layer.gemm1_scales_fp4_shuffled = Parameter(gemm1_scales_fp4_shuffled, requires_grad=False)
        layer.gemm2_scales_fp4_shuffled = Parameter(gemm2_scales_fp4_shuffled, requires_grad=False)

        # Additional parameter needed for TRT-LLM
        layer.g1_scale_c = Parameter(
            (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
            requires_grad=False,
        )

        # Clean up weights that won't be used by TRT-LLM
        del layer.w2_weight
        del layer.w2_weight_scale
        del layer.w13_weight
        del layer.w13_weight_scale

    elif self.use_marlin:
        # Marlin processing - use patched version
        # Pass per-expert scales (shape: num_experts), NOT scalar max values!
        prepare_moe_fp4_layer_for_marlin_patched(layer, w13_weight_scale_2_per_expert, w2_weight_scale_2_per_expert)
        # Delete attributes not needed for Marlin
        del layer.g1_alphas
        del layer.g2_alphas
        del layer.w13_input_scale_quant
        del layer.w2_input_scale_quant

    else:
        # Non-TRT-LLM processing (Cutlass or non-flashinfer)
        w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
        layer.w13_weight_scale = Parameter(w13_blockscale_swizzled, requires_grad=False)

        w13_weight = layer.w13_weight
        intermediate_size_pad = w13_blockscale_swizzled.size(1) - w13_weight.size(1)
        if intermediate_size_pad:
            # padding gated activations will require to split w1 and w3
            # and pad them individually
            assert not self.moe.is_act_and_mul, (
                "The intermediate size required padding, but padding is not implemented for gated activations"
            )

            layer.w13_weight = Parameter(
                torch.nn.functional.pad(w13_weight, (0, 0, 0, intermediate_size_pad)),
                requires_grad=False,
            )
            layer.w2_weight = Parameter(
                torch.nn.functional.pad(layer.w2_weight, (0, intermediate_size_pad // 2, 0, 0)),
                requires_grad=False,
            )
            layer.w2_weight_scale = Parameter(
                torch.nn.functional.pad(layer.w2_weight_scale, (0, intermediate_size_pad // 16)),
                requires_grad=False,
            )

        w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
        layer.w2_weight_scale = Parameter(w2_blockscale_swizzled, requires_grad=False)


def apply_moe(
    self,
    layer,  # FusedMoE
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Patched apply method for ModelOptNvFp4FusedMoE.

    Key modification for Marlin: Uses marlin_* attributes instead of originals.
    """
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        flashinfer_trtllm_fp4_moe,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        FlashinferMoeBackend,
    )
    from vllm.scalar_type import scalar_types

    if not self.moe.is_act_and_mul:
        assert self.allow_flashinfer and self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS, (
            "Non-gated activations are only supported by the flashinfer CUTLASS backend for modelopt checkpoints"
        )

    if self.allow_flashinfer and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
        if enable_eplb:
            raise NotImplementedError("EPLB not supported for `ModelOptNvFp4FusedMoE` yet.")
        return flashinfer_trtllm_fp4_moe(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=top_k,
            global_num_experts=global_num_experts,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            e_score_correction_bias=e_score_correction_bias,
        )

    topk_weights, topk_ids, _ = layer.select_experts(
        hidden_states=x,
        router_logits=router_logits,
    )

    if self.use_marlin:
        # Use marlin_* attributes instead of original attributes
        return fused_marlin_moe(
            x,
            layer.marlin_w13_weight,
            layer.marlin_w2_weight,
            None,  # bias1
            None,  # bias2
            layer.marlin_w13_weight_scale,
            layer.marlin_w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            global_scale1=layer.marlin_w13_weight_scale_2,
            global_scale2=layer.marlin_w2_weight_scale_2,
            workspace=layer.workspace,
            input_dtype=self.marlin_input_dtype,
        )

    elif self.allow_flashinfer:
        assert self.flashinfer_moe_backend in (
            FlashinferMoeBackend.CUTLASS,
            FlashinferMoeBackend.CUTEDSL,
        )
        if self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
                flashinfer_cutlass_moe_fp4,
            )

            flashinfer_fn_moe_fp4 = flashinfer_cutlass_moe_fp4
        else:
            from vllm.model_executor.layers.fused_moe.flashinfer_cutedsl_moe import (
                flashinfer_cutedsl_moe_fp4,
            )

            flashinfer_fn_moe_fp4 = flashinfer_cutedsl_moe_fp4

        assert self.moe_quant_config is not None
        return flashinfer_fn_moe_fp4(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=self.moe_quant_config,
            inplace=False,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
    else:
        # If no modular kernel is provided, use cutlass_moe_fp4 for TP case
        # only (no EP).
        from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4

        assert self.moe_quant_config is not None
        return cutlass_moe_fp4(
            a=x,
            w1_fp4=layer.w13_weight,
            w2_fp4=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=self.moe_quant_config,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            m=x.shape[0],
            n=layer.w2_weight.shape[2] * 2,
            k=x.shape[1],
            e=layer.w13_weight.shape[0],
        )


def process_weights_after_loading_kv(self, layer) -> None:
    """Modified version of BaseKVCacheMethod.process_weights_after_loading.

    Doesn't delete k_scale, v_scale, q_scale, and prob_scale parameters to allow
    for dynamic updates during refit.
    """
    # If the kv-cache dtype is auto, we enforce the k/v_scale to be 1.0
    # regardless whether the kv-scale is available in the checkpoint.
    # No need to process kv scales after loading if we are going to
    # calculate them on the fly.
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # We prefer to use separate k_scale and v_scale if present
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # If no scales were loaded (both scales are invalid negative
            # values), use the default value of 1.0
            k_scale = 1.0
            v_scale = 1.0
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        # These are used in the final Attention.forward()
        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = (
        lambda x: isinstance(x, float) or isinstance(x, torch.Tensor) and x.numel() == 1 and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factorfor fp8-quantized Q/prob")

    # These are used in the final Attention.forward()
    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale

    layer._prob_scale.copy_(prob_scale)


def apply_vllm_modelopt_patches():
    func1_path = (
        "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4LinearMethod.process_weights_after_loading"
    )
    patcher1 = patch(func1_path, process_weights_after_loading_modelopt)
    patcher1.start()
    func2_path = "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4LinearMethod.apply"
    patcher2 = patch(func2_path, apply_modelopt)
    patcher2.start()
    # Patch ModelOptNvFp4FusedMoE
    func3_path = "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4FusedMoE.process_weights_after_loading"
    patcher3 = patch(func3_path, process_weights_after_loading_moe)
    patcher3.start()
    func4_path = "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4FusedMoE.apply"
    patcher4 = patch(func4_path, apply_moe)
    patcher4.start()
    # Static scales mode: patch process_weights_after_loading to preserve k_scale/v_scale for manual updates
    func5_path = "vllm.model_executor.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading"
    patcher5 = patch(func5_path, process_weights_after_loading_kv)
    patcher5.start()