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
"""Routed-expert NVFP4 activation QAT: emulates the vLLM rollout's activation quantizer.

This reproduces the quantization rule, not the rest of the served MoE kernel (TP-rank-local
rounding, expert finalize order); see the model module for known differences.

The rollout serves routed experts with the FlashInfer TRT-LLM NVFP4 MoE
(``TrtLlmNvFp4LoRAExperts``) and *per-token dynamic* activation scales; the
checkpoint's ``input_global_scale`` is not used there.

Both inputs go through FlashInfer's per-token NVFP4 quantizer
(``invokeNvfp4QuantAndPerTokenScale``): the FC1 input (hidden states, full row
``H``) through vLLM ``quantize_nvfp4_per_token_input``, the FC2 input (BF16
activation output) inside the TRT-LLM MoE kernel, on each tensor-parallel shard
of the intermediate dim (``I / tp_shards`` columns, the rollout's TP size)::

      s = amax * (1/2688) (FP32, not rounded); enc = 1/s
      sf = E4M3_sat(enc * blockmax16 / 6); code = E2M1_RNE_sat(x / (sf / enc))

``s`` multiplies the row's (FC2: the shard's) FP32 partial accumulator.

``code * sf`` has at most 6 significant bits, so it is exact in BF16 and is the
GEMM operand; every outer scale (row/shard scale, ``1/weight_global_scale``) is
applied to the FP32 accumulator, as the deployed kernels do. The backward pass
is a straight-through estimator: the gradient w.r.t. the input is the gradient
of the unquantized linear ``x @ W.T / weight_global_scale``, computed with the
same accumulator discipline (BF16 operands, FP32 accumulator, scale, one
rounding). LoRA branches are outside this function and see the unquantized
input, as in vLLM.

Where the kernel uses ``rcp.approx`` this uses IEEE division; the unit tests
measure the difference.
"""

from __future__ import annotations

import logging

import torch

NVFP4_ROW_SCALE_MAX = 6.0 * 448.0  # 2688
NVFP4_BLOCK = 16
# Magnitude thresholds of round-to-nearest-even E2M1 (cvt.rn.satfinite.e2m1):
# (threshold, strict). Ties go to the even code (0, 1, 2, 4).
_E2M1_THRESHOLDS = ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True))
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _div(a: torch.Tensor, b) -> torch.Tensor:
    # A tensor divisor: PyTorch turns division by a Python scalar into a reciprocal multiply.
    return torch.div(a, torch.as_tensor(b, dtype=a.dtype, device=a.device))


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """FP32 values rounded to E2M1 (nearest-even, saturating at +-6), as FP32."""
    mag = v.abs()
    index = torch.zeros_like(mag, dtype=torch.long)
    for threshold, strict in _E2M1_THRESHOLDS:
        index += (mag > threshold) if strict else (mag >= threshold)
    table = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=v.device)
    return torch.copysign(table[index], v)


def fc1_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, H]`` BF16 -> (``code * sf`` exact BF16 ``[M, H]``, FP32 token scale ``[M]``).

    The per-token rule of :func:`fc2_quantize` over the full row (one shard).
    """
    (unit,), (scale,) = fc2_quantize(x, 1)
    return unit, scale


def fc2_quantize(a: torch.Tensor, shards: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``[M, I]`` BF16 -> per shard (``code * sf`` exact BF16 ``[M, I/shards]``, FP32 scale ``[M]``)."""
    units, scales = [], []
    inv_max = torch.tensor(1.0 / NVFP4_ROW_SCALE_MAX, dtype=torch.float32, device=a.device)
    sixth = _div(torch.ones((), device=a.device), 6.0)
    for part in a.chunk(shards, dim=-1):
        rows, cols = part.shape
        x = part.float()
        s = (x.abs().amax(-1) if rows else x.new_zeros(0)) * inv_max
        enc = torch.where(s != 0, _div(torch.ones_like(s), s), torch.zeros_like(s))
        blocks = x.view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
        bmax = blocks.abs().amax(-1) if rows else blocks.new_zeros(0, cols // NVFP4_BLOCK)
        sf = (enc[:, None] * (bmax * sixth)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        # outScale = rcp(sf * rcp(enc)); sf == 0 (underflow) gives code * 0 = 0 below.
        out_scale = torch.where(
            (bmax != 0) & (sf != 0),
            _div(torch.ones_like(sf), sf * _div(torch.ones_like(enc), enc)[:, None]),
            torch.zeros_like(sf),
        )
        unit = e2m1_round(blocks * out_scale[..., None]) * sf[..., None]
        units.append(unit.view(rows, cols).to(torch.bfloat16))
        scales.append(s)
    return units, scales


_E2M1_CODE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


_DECODE_LUT: dict[torch.device, torch.Tensor] = {}


def _decode_lut(device: torch.device) -> torch.Tensor:
    """[256, 2] BF16: the E2M1 values of a packed byte's low and high nibble."""
    lut = _DECODE_LUT.get(device)
    if lut is None:
        values = torch.tensor(_E2M1_CODE_VALUES, dtype=torch.bfloat16, device=device)
        byte = torch.arange(256, device=device)
        lut = _DECODE_LUT[device] = torch.stack((values[byte & 0xF], values[byte >> 4]), -1)
    return lut


def _decode(packed: torch.Tensor, sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Packed E2M1 (low nibble first) + linear E4M3 block scales -> exact BF16 ``code * sf``.

    A code has at most 2 significant bits and an E4M3 scale 4, so the product is exact in BF16
    and is formed there directly.
    """
    codes = _decode_lut(packed.device)[packed.view(torch.uint8).reshape(rows, cols // 2).int()]
    sf = sf.view(torch.float8_e4m3fn).reshape(rows, -1)[:, : cols // NVFP4_BLOCK].float().to(torch.bfloat16)
    return (codes.view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK) * sf[..., None]).view(rows, cols)


def fc1_quantize_kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`fc1_quantize` through FlashInfer's per-token NVFP4 quantizer, as vLLM's
    ``quantize_nvfp4_per_token_input`` runs it on the FC1 input (bitwise equal to that quantizer)."""
    (unit,), (scale,) = fc2_quantize_kernel(x, 1)
    return unit, scale


def fc2_quantize_kernel(a: torch.Tensor, shards: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """:func:`fc2_quantize` through FlashInfer's per-token NVFP4 quantizer, the same
    ``invokeNvfp4QuantAndPerTokenScale`` the TRT-LLM MoE runs on its FC2 input."""
    from flashinfer import SfLayout, nvfp4_quantize

    if a.shape[0] == 0:
        return fc2_quantize(a, shards)
    units, scales = [], []
    for part in a.chunk(shards, dim=-1):
        rows, cols = part.shape
        packed, sf, token_scale = nvfp4_quantize(
            part.contiguous(), 1.0 / NVFP4_ROW_SCALE_MAX, sfLayout=SfLayout.layout_linear, per_token_activation=True
        )
        units.append(_decode(packed, sf, rows, cols))
        scales.append(token_scale.float().reshape(rows))
    return units, scales


_QUANTIZERS = {
    "torch": (fc1_quantize, fc2_quantize),
    "kernel": (fc1_quantize_kernel, fc2_quantize_kernel),
}

_MM_OUT_DTYPE_OK: dict[str, bool] = {}
logger = logging.getLogger(__name__)


def _mm_fp32(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 ``a @ w.T`` with an FP32 result (no BF16 rounding of the accumulator)."""
    key = a.device.type
    if _MM_OUT_DTYPE_OK.get(key, True):
        try:
            return torch.mm(a, w.t(), out_dtype=torch.float32)
        except (TypeError, RuntimeError, NotImplementedError) as exc:
            _MM_OUT_DTYPE_OK[key] = False
            logger.warning(
                "torch.mm(out_dtype=float32) unavailable on %s (%s); using the FP32 matmul fallback", key, exc
            )
    return a.float() @ w.float().t()


class NVFP4ActQuantLinear(torch.autograd.Function):
    """``y = Q(x) @ W.T / weight_global_scale`` with the rollout's activation quantization, STE backward.

    ``mode`` is ``"fc1"``/``"fc2"`` (deployed NVFP4 activation quantization) or
    ``"none"`` (``Q`` is the identity). In every mode the GEMM accumulator stays
    FP32 until all outer scales are applied, and the result is rounded once:
    to BF16, or not at all when ``out_fp32`` (FC1 feeds the activation from the
    FP32 accumulator).
    """

    @staticmethod
    def forward(ctx, x, weight, global_scale, mode: str, shards: int, backend: str = "kernel", out_fp32: bool = False):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        with torch.autocast(device_type=x.device.type, enabled=False):
            w = weight.to(torch.bfloat16)
            if mode == "none":
                acc = _mm_fp32(x2, w)
            elif mode == "fc1":
                unit, scale = _QUANTIZERS[backend][0](x2)
                acc = _mm_fp32(unit, w) * scale[:, None]
            elif mode == "fc2":
                units, scales = _QUANTIZERS[backend][1](x2, shards)
                acc = None
                for unit, w_part, scale in zip(units, w.chunk(shards, dim=1), scales, strict=True):
                    part = _mm_fp32(unit, w_part) * scale[:, None]
                    acc = part if acc is None else acc + part
            else:
                raise ValueError(f"Unknown activation QAT mode {mode!r}")
            out = acc * (1.0 / global_scale.float())
        ctx.save_for_backward(weight, global_scale)
        ctx.in_dtype = x.dtype
        ctx.in_shape = shape
        if not out_fp32:
            out = out.to(torch.bfloat16)
        return out.view(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        weight, global_scale = ctx.saved_tensors
        g = grad_out.reshape(-1, grad_out.shape[-1])
        with torch.autocast(device_type=grad_out.device.type, enabled=False):
            # Same order as the forward: FP32 accumulator, the global scale on the
            # accumulator, one final rounding. An FP32 upstream gradient (the FP32
            # output feeding the activation) is used as is; rounding it to BF16
            # first would cancel small differences between output channels.
            if g.dtype == torch.float32:
                acc = g @ weight.float()
            else:
                acc = _mm_fp32(g.to(torch.bfloat16), weight.to(torch.bfloat16).t())
            grad_x = (acc * (1.0 / global_scale.float())).to(ctx.in_dtype)
        return grad_x.view(ctx.in_shape), None, None, None, None, None, None
