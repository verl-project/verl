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
import os

import torch

from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def should_quantize_param(param_name: str) -> bool:
    """Determine whether to quantize to FP8 based on parameter name

    Quantization rules:
    - Must end with .weight (exclude bias)
    - Exclude embedding layers
    - Exclude normalization layers
    - Exclude output layer (lm_head)
    """
    # Must be a weight parameter
    if not param_name.endswith(".weight"):
        return False

    # Layer types to exclude
    exclude_patterns = [
        "embed_tokens",  # Embedding layer
        "lm_head",  # Output layer
        "layernorm",  # LayerNorm
        "norm",  # Various Norm layers
        "ln_",  # LayerNorm variants
        "embeddings",  # Embeddings
        "mlp.gate.weight",  # MoE router
    ]

    # Check if matches exclude patterns
    param_lower = param_name.lower()
    for pattern in exclude_patterns:
        if pattern in param_lower:
            return False

    # Layer types to include (Linear layers)
    include_patterns = [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # Gate projection (for MLP)
        "up_proj",  # Up projection (for MLP)
        "down_proj",  # Down projection (for MLP)
        "fc1",  # Fully connected 1
        "fc2",  # Fully connected 2
        "mlp",  # MLP layers
    ]

    # Check if matches include patterns
    for pattern in include_patterns:
        if pattern in param_lower:
            logger.debug(f"Will quantize FP8: {param_name}")
            return True

    # Do not quantize by default
    logger.debug(f"Skip quantization: {param_name}")
    return False


def quant_weights_by_name(weights, quant_config, dtype=torch.bfloat16):
    """FP8 quantization based on parameter name using a memory-efficient generator.


    Args:
        weights: Generator or iterable of (name, tensor) pairs
        quant_config: Quantization configuration
        dtype: Data type for intermediate computation

    Yields:
        Tuples of (name, tensor) for each weight and its scale
    """
    if isinstance(quant_config, dict):
        weight_block_size = quant_config.get("weight_block_size")
    else:
        weight_block_size = getattr(quant_config, "weight_block_size", None)

    if weight_block_size is None:
        raise ValueError("weight_block_size not found in quant_config")

    for k, v in weights:
        # Check if quantization is needed
        if not should_quantize_param(k):
            yield (k, v)
            continue

        # Quantize to FP8
        try:
            if torch.distributed.get_rank() == 0:
                logger.debug(f"Quantizing to FP8 blockwise: {k}")

            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)

            # Yield the quantized weight and scale
            yield (k, param_lp)
            yield (k + "_scale_inv", param_scale)

            # Explicitly delete to help GC
            del param_lp, param_scale

        except Exception as e:
            logger.error(f"Failed to quantize {k}: {e}")
            # If quantization fails, use original weights
            yield (k, v)
