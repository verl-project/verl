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
"""MXFP8 (OCP microscaling FP8) rollout quantization for SGLang.

Rollout weights are quantized with TransformerEngine's ``MXFP8Quantizer`` — the
same quantizer Megatron/TE applies inside FP8 GEMMs when training with
``fp8_recipe="mxfp8"`` — so SGLang serves exactly the weight grid the trainer's
forward pass saw. Quantizing with an independent kernel instead can round E8M0
scales differently at block boundaries and reintroduce train-inference mismatch.
"""

import logging
import os

import torch

from verl.utils.mxfp8_quant import MXFP8_GROUP_SIZE, TE_MXFP8_ROW_ALIGNMENT, mxfp8_quantize
from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config
from verl.workers.rollout.utils import ensure_async_iterator

__all__ = [
    "MXFP8_GROUP_SIZE",
    "TE_MXFP8_ROW_ALIGNMENT",
    "SGLangMXFP8QuantizerHelper",
    "build_sglang_mxfp8_quant_config",
    "check_sglang_mxfp8_support",
    "mxfp8_quantize",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def check_sglang_mxfp8_support() -> None:
    """Fail fast when the installed sglang cannot serve MXFP8 checkpoints."""
    try:
        from sglang.srt.layers.quantization import BASE_QUANTIZATION_METHODS
    except ImportError as err:
        raise ImportError("sglang is required for MXFP8 rollout quantization") from err
    if "mxfp8" not in BASE_QUANTIZATION_METHODS:
        raise ValueError(
            "The installed sglang does not support quantization='mxfp8'; upgrade to a version "
            "whose quantization registry includes 'mxfp8'."
        )


def build_sglang_mxfp8_quant_config(hf_config=None, ignored_layers=None) -> dict:
    """Build the SGLang MXFP8 config shared by server init and weight sync.

    Reuses the FP8 builder's ignored-layers merging (hf quantization_config,
    modules_to_not_convert, SGLANG_FP8_IGNORED_LAYERS env), then switches the
    method to mxfp8 with the OCP-mandated [1, 32] block size.
    """
    quant_config = build_sglang_fp8_quant_config(hf_config, ignored_layers)
    quant_config["quant_method"] = "mxfp8"
    quant_config["weight_block_size"] = [1, 32]
    return quant_config


class SGLangMXFP8QuantizerHelper(SGLangFP8QuantizerHelper):
    """Weight-sync quantizer for MXFP8 rollout.

    Same name-based selection and ignored-layers handling as the FP8 helper,
    but quantizes with TE's MXFP8Quantizer to keep rollout weights on the
    training grid. Unlike the FP8 helper, quantization errors propagate: the
    server expects fp8-serialized tensors, so silently falling back to bf16
    would fail later with a much less actionable error.
    """

    async def quant_weights_by_name(self, weights, dtype=torch.bfloat16):
        async for k, v in ensure_async_iterator(weights):
            if not self.should_quantize_param(k):
                yield (k, v)
                continue

            if v.shape[-1] % MXFP8_GROUP_SIZE != 0:
                # Falling back to bf16 here would desync from the server, whose
                # parameters for this layer were built expecting fp8 + scales.
                raise ValueError(
                    f"Cannot MXFP8-quantize '{k}': last dim {v.shape[-1]} is not divisible by "
                    f"{MXFP8_GROUP_SIZE}. Exclude this layer from quantization (consistently for "
                    "server init and weight sync) via quantization_config.ignored_layers or the "
                    "SGLANG_FP8_IGNORED_LAYERS env var, e.g. SGLANG_FP8_IGNORED_LAYERS=visual "
                    "for vision-language models."
                )

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.debug(f"Quantizing to MXFP8: {k}")
            param_lp, param_scale = mxfp8_quantize(v.to(dtype))
            yield (k, param_lp)
            yield (k + "_scale_inv", param_scale)
            del param_lp, param_scale
