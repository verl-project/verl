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
"""Engine-neutral MXFP8 (OCP microscaling FP8) weight quantization.

Rollout weights are quantized with TransformerEngine's ``MXFP8Quantizer`` — the
same quantizer Megatron/TE applies inside FP8 GEMMs when training with
``fp8_recipe="mxfp8"`` — so the rollout engine (SGLang or vLLM) serves exactly
the weight grid the trainer's forward pass saw. Quantizing with an independent
kernel instead can round E8M0 scales differently at block boundaries and
reintroduce train-inference mismatch.
"""

import torch

MXFP8_GROUP_SIZE = 32
# Layers that must stay in high precision under MXFP8 rollout quantization.
#
# ``lm_head`` projects the hidden state onto the full vocabulary. Its dynamic
# range does not survive MXFP8's 32-element blocks along K: quantizing it makes
# the logits ``nan``, which silently degrades sampling — generations never emit
# EOS and run to ``max_response_length``, so reward stays at 0 while the run
# still exits 0. Verified on 2xB200 (Qwen3-8B, gsm8k): with ``lm_head``
# quantized, ``training/rollout_probs_diff_mean`` and ``rollout_corr/*`` are all
# ``nan`` and reward is 0.0 for every step; excluding it restores reward
# (0.268 -> 0.393 over 3 steps) and clears every ``nan``.
#
# The embedding is listed for symmetry: it shares the vocabulary dimension and
# the same argument applies if an engine ever routes it through a quantized
# linear.
MXFP8_KEEP_HIGH_PRECISION_LAYERS = ("lm_head", "model.embed_tokens")
# TE's MXFP8 quantizer requires both dims 32-aligned; weights whose row count
# is not a multiple of 32 are zero-padded before quantization and sliced after.
TE_MXFP8_ROW_ALIGNMENT = 32


def mxfp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to rowwise MXFP8 (E4M3 elements + per-32 UE8M0 scales).

    Returns ``(qweight, scale)`` where ``qweight`` keeps the input shape in
    ``float8_e4m3fn`` and ``scale`` is ``uint8`` UE8M0 with shape
    ``[*weight.shape[:-1], k // 32]`` — the compact, unswizzled layout both
    SGLang (``weight_scale_inv``) and vLLM ModelOpt (``weight_scale``) expect
    at load time.
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
