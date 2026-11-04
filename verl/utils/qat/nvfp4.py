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

"""Engine-neutral NVFP4 quantization for trainer-side export and QAT.

:func:`quantize_nvfp4` follows the arithmetic of the CUDA NVFP4 quantization kernel the
inference engines share (vLLM ``scaled_fp4_quant``, which vLLM's online NVFP4 methods use):
16-element blocks, an E4M3 block scale ``global_scale * amax / 6`` and round-to-nearest-even
E2M1 values. The kernel computes reciprocals with ``rcp.approx.ftz.f32``, so this Triton
kernel does too; a GPU test checks it bitwise against vLLM.

:func:`fake_quantize_nvfp4` returns the dequantized values of the same weight quantization, and
:func:`fake_quantize_nvfp4_per_token` those of FlashInfer's per-token activation quantization
(``nvfp4_quantize(..., per_token_activation=True)``), which the per-token MoE kernel applies to
its input. That kernel derives one global scale per row from the row's amax, and switches to
IEEE divisions when ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`` or
``TRTLLM_DISABLE_FP4_QUANT_FAST_MATH`` is ``1``; the fake quantization follows the same switch.
"""

import os

import torch
import triton
import triton.language as tl

__all__ = ["fake_quantize_nvfp4", "fake_quantize_nvfp4_per_token", "nvfp4_global_scale", "quantize_nvfp4"]

_BLOCK = 16
_FP4_MAX = 6.0
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# FlashInfer's per-token base scale, vLLM's ``NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE``.
_PER_TOKEN_SCALE_INV = 1.0 / (_FP8_MAX * _FP4_MAX)


def nvfp4_global_scale(*tensors: torch.Tensor) -> torch.Tensor:
    """vLLM's online NVFP4 global scale: (fp4_max * fp8_max) / amax, amax clamped away from zero."""
    amax = torch.stack([t.float().abs().amax() for t in tensors]).amax().clamp_min(1e-8)
    return (_FP4_MAX * _FP8_MAX / amax).reshape(1)


def _fp4_quant_fast_math() -> bool:
    # FlashInfer's getEnvDisableFP4QuantFastMath reads either variable and accepts only "1".
    return not any(
        os.environ.get(name) == "1"
        for name in ("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH")
    )


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _e2m1_code(x):
    # Round-to-nearest-even onto {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at 6.
    a = tl.abs(x)
    code = tl.where(
        a <= 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a <= 1.25,
                2,
                tl.where(a < 1.75, 3, tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7)))),
            ),
        ),
    )
    # The sign bit survives rounding to zero, as in the hardware conversion.
    sign = tl.where(x.to(tl.int32, bitcast=True) < 0, 8, 0)
    return (code + sign).to(tl.uint8)


@triton.jit
def _e2m1_value(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 4, 2.0, tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0))),
    )
    return tl.where(code >= 8, -value, value)


@triton.jit
def _block_scales(amax, encode, FAST_MATH: tl.constexpr):
    """The E4M3 block scale and the factor that maps a block onto E2M1 (``cvt_warp_fp16_to_fp4``)."""
    if FAST_MATH:
        block_scale = (encode * (amax * _rcp_approx(tl.full(amax.shape, 6.0, tl.float32)))).to(tl.float8e4nv)
        scale = block_scale.to(tl.float32)
        out_scale = tl.where(scale != 0.0, _rcp_approx(scale * _rcp_approx(encode)), 0.0)
    else:
        block_scale = (amax * (encode * (1.0 / 6.0))).to(tl.float8e4nv)
        scale = block_scale.to(tl.float32)
        ones = tl.full(amax.shape, 1.0, tl.float32)
        out_scale = tl.where(
            amax != 0.0,
            tl.minimum(tl.math.div_rn(ones, scale * tl.math.div_rn(ones, encode)), 3.4028234663852886e38),
            0.0,
        )
    return block_scale, scale, out_scale


@triton.jit
def _quantize_nvfp4_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    M,
    N,
    stride_xm,
    TILE_M: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    rows = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    blocks = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blocks[:, None] * 16 + tl.arange(0, 16)[None, :]
    row_mask = rows < M
    block_mask = blocks < N // 16
    mask = row_mask[:, None, None] & block_mask[None, :, None]

    x = tl.load(x_ptr + rows[:, None, None] * stride_xm + cols[None, :, :], mask=mask, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr)

    amax = tl.max(tl.abs(x), axis=2)
    block_scale, _, out_scale = _block_scales(amax, tl.full(amax.shape, 0.0, tl.float32) + global_scale, True)

    codes = _e2m1_code(x * out_scale[:, :, None])
    low, high = tl.split(tl.reshape(codes, (TILE_M, BLOCKS, 8, 2)))
    packed = low | (high << 4)

    packed_cols = blocks[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(
        packed_ptr + rows[:, None, None] * (N // 2) + packed_cols[None, :, :],
        packed,
        mask=row_mask[:, None, None] & block_mask[None, :, None],
    )
    tl.store(
        scale_ptr + rows[:, None] * (N // 16) + blocks[None, :],
        block_scale.to(tl.uint8, bitcast=True),
        mask=row_mask[:, None] & block_mask[None, :],
    )


@triton.jit
def _fake_quantize_nvfp4_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    N,
    stride_xm,
    scale_inv,
    PER_TOKEN: tl.constexpr,
    FAST_MATH: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    rows = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    blocks = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blocks[:, None] * 16 + tl.arange(0, 16)[None, :]
    row_mask = rows < M
    mask = row_mask[:, None, None] & (blocks < N // 16)[None, :, None]
    offsets = rows[:, None, None] * stride_xm + cols[None, :, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if PER_TOKEN:
        # nvfp4QuantAndPerTokenScaleKernel: ``scale_ptr`` holds each row's amax.
        row_amax = tl.load(scale_ptr + rows, mask=row_mask, other=0.0)
        if FAST_MATH:
            decode = row_amax * scale_inv
            encode = tl.where(decode != 0.0, _rcp_approx(decode), 0.0)
        else:
            ones = tl.full(row_amax.shape, 1.0, tl.float32)
            global_scale = tl.math.div_rn(ones, ones * scale_inv)
            encode = tl.where(
                row_amax != 0.0,
                tl.minimum(tl.math.div_rn(global_scale, row_amax), 3.4028234663852886e38),
                3.4028234663852886e38,
            )
            encode = tl.where(encode != 0.0, encode, 1.0)
            decode = tl.math.div_rn(ones, encode)
    else:
        # ``scale_ptr`` holds the tensor's global scale.
        ones = tl.full((TILE_M,), 1.0, tl.float32)
        encode = ones * tl.load(scale_ptr)
        decode = tl.math.div_rn(ones, encode)

    amax = tl.max(tl.abs(x), axis=2)
    _, scale, out_scale = _block_scales(amax, tl.zeros_like(amax) + encode[:, None], FAST_MATH)
    values = _e2m1_value(_e2m1_code(x * out_scale[:, :, None]))
    out = values * (scale * decode[:, None])[:, :, None]
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def _check_input(x: torch.Tensor, what: str) -> None:
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda:
        raise ValueError(f"{what} takes a 2D BF16/FP16 CUDA tensor, got {x.dtype} {tuple(x.shape)}")
    if x.shape[1] % _BLOCK:
        raise ValueError(f"{what} needs a multiple of {_BLOCK} columns, got {x.shape[1]}")


def quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D BF16/FP16 CUDA tensor to NVFP4 with a given float32 global scale.

    Returns the packed E2M1 values ``[M, N / 2]`` (uint8, low nibble first) and the
    row-major E4M3 block scales ``[M, N / 16]``.
    """
    _check_input(x, "NVFP4 quantization")
    m, n = x.shape
    x = x.contiguous()
    global_scale = global_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty((m, n // _BLOCK), dtype=torch.uint8, device=x.device)
    tile_m, blocks = 16, 16
    grid = (triton.cdiv(m, tile_m), triton.cdiv(n // _BLOCK, blocks))
    _quantize_nvfp4_kernel[grid](x, packed, scale, global_scale, m, n, x.stride(0), TILE_M=tile_m, BLOCKS=blocks)
    return packed, scale.view(torch.float8_e4m3fn)


def _fake_quantize(x: torch.Tensor, scale: torch.Tensor, *, per_token: bool, fast_math: bool) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    if m == 0:
        # An EP rank whose experts receive no tokens.
        return out
    tile_m, blocks = 16, 16
    grid = (triton.cdiv(m, tile_m), triton.cdiv(n // _BLOCK, blocks))
    _fake_quantize_nvfp4_kernel[grid](
        x,
        out,
        scale,
        m,
        n,
        x.stride(0),
        _PER_TOKEN_SCALE_INV,
        PER_TOKEN=per_token,
        FAST_MATH=fast_math,
        TILE_M=tile_m,
        BLOCKS=blocks,
    )
    return out


def fake_quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Dequantized :func:`quantize_nvfp4` of a 2D BF16/FP16 CUDA tensor, in its dtype."""
    _check_input(x, "NVFP4 fake quantization")
    global_scale = global_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    return _fake_quantize(x.contiguous(), global_scale, per_token=False, fast_math=True)


def fake_quantize_nvfp4_per_token(x: torch.Tensor) -> torch.Tensor:
    """Dequantized per-token NVFP4 activation quantization of a 2D BF16/FP16 CUDA tensor, in its dtype."""
    _check_input(x, "Per-token NVFP4 fake quantization")
    x = x.contiguous()
    row_amax = x.abs().amax(dim=1).float()
    return _fake_quantize(x, row_amax, per_token=True, fast_math=_fp4_quant_fast_math())
