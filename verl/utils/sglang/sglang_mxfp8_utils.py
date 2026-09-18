# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from typing import Any

import torch

from verl.utils.fp8_utils import FP8QuantizerHelper
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

TARGET_MXFP8_BLOCK_SIZE = [1, 32]
# Mirror the FP8 hardcode (activation_scheme / fmt / quant_method / weight_block_size),
# plus scale_fmt for the MXFP8 UE8M0 scales.
MXFP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
    "scale_fmt": "ue8m0",
}


def get_mxfp8_quant_config() -> dict[str, Any]:
    return dict(MXFP8_BLOCK_QUANT_KWARGS)


def _quantize_with_sglang(tensor_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # SGLang's triton MXFP8 helper: FP8 E4M3 weights + UE8M0 (uint8) scales grouped
    # along the input dim in chunks of 32, in SGLang's swizzle-free layout.
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

    return mxfp8_group_quantize(tensor_2d)


class SGLangMXFP8QuantizerHelper(FP8QuantizerHelper):
    """Quantize SGLang rollout weights to MXFP8."""

    def should_quantize_param(self, param_name, tensor=None):
        # Keep the optional tensor argument for MXFP8 callers while intentionally
        # using the same name-based selection policy as the upstream FP8 helper.
        return super().should_quantize_param(param_name)

    async def quant_weights_by_name(self, weights, dtype=torch.bfloat16):
        if isinstance(self.quant_config, dict):
            weight_block_size = self.quant_config.get("weight_block_size")
        else:
            weight_block_size = getattr(self.quant_config, "weight_block_size", None)

        if weight_block_size is None:
            raise ValueError("weight_block_size not found in quant_config")

        async for param_name, tensor in ensure_async_iterator(weights):
            if not self.should_quantize_param(param_name, tensor):
                yield (param_name, tensor)
                continue

            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_rank() == 0
            ):
                logger.debug(f"Quantizing to MXFP8: {param_name}")

            # Do not silently fall back to bf16 on failure: an MXFP8-configured
            # rollout engine cannot consume an unquantized weight in this slot.
            tensor_2d = tensor.to(dtype).reshape(-1, tensor.shape[-1]).contiguous()
            param_lp, param_scale = _quantize_with_sglang(tensor_2d)
            scale = param_scale.view(
                *tensor.shape[:-1],
                tensor.shape[-1] // TARGET_MXFP8_BLOCK_SIZE[1],
            ).contiguous()

            yield (param_name, param_lp.view_as(tensor))
            yield (param_name + "_scale_inv", scale)

            del tensor_2d, param_lp, param_scale, scale
