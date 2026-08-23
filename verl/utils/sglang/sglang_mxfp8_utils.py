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

from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MXFP8_GROUP_SIZE = 32
# TE's MXFP8 quantizer requires both dims 32-aligned; weights whose row count
# is not a multiple of 32 are zero-padded before quantization and sliced after.
TE_MXFP8_ROW_ALIGNMENT = 32


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


def mxfp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to rowwise MXFP8 (E4M3 elements + per-32 UE8M0 scales).

    Returns ``(qweight, scale)`` where ``qweight`` keeps the input shape in
    ``float8_e4m3fn`` and ``scale`` is ``uint8`` UE8M0 with shape
    ``[*weight.shape[:-1], k // 32]`` — the compact, unswizzled layout SGLang
    expects for ``weight_scale_inv``.
    """
    try:
        from transformer_engine.pytorch import MXFP8Quantizer
        from transformer_engine.pytorch.constants import TE_DType
    except ImportError as err:
        raise ImportError(
            "TransformerEngine (>=2.1, with MXFP8 support) is required for mxfp8 rollout "
            "quantization: rollout weights must be quantized with the same TE quantizer the "
            "trainer's FP8 GEMMs use."
        ) from err

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP8_GROUP_SIZE} for MXFP8.")

    weight_flat = weight.view(-1, k)
    num_rows = weight_flat.shape[0]
    pad_rows = (-num_rows) % TE_MXFP8_ROW_ALIGNMENT
    if pad_rows:
        padding = torch.zeros((pad_rows, k), device=weight.device, dtype=weight.dtype)
        weight_flat = torch.cat((weight_flat, padding), dim=0)

    quantizer = MXFP8Quantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
    )
    quantized = quantizer.quantize(weight_flat)
    qweight = quantized._rowwise_data[:num_rows, :k].contiguous()
    qweight = qweight.view(torch.float8_e4m3fn).view_as(weight)
    scale = quantized._rowwise_scale_inv[:num_rows, : k // MXFP8_GROUP_SIZE]
    scale = scale.contiguous().view(*weight.shape[:-1], k // MXFP8_GROUP_SIZE)
    return qweight, scale


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
