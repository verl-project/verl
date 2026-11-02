# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 E4M3 block quantization shared by QAT and weight export."""

import math
from collections.abc import Sequence
from functools import cache
from typing import Literal

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

__all__ = ["fp8_quant"]

_FP8 = "float8_e4m3"
_E8M0 = "float8_e8m0fnu"
_FP32 = "float32"
_TORCH_TO_TILELANG_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
ScaleFormat = Literal["e8m0", "ue8m0"]


def _round_up_to_power_of_two(value):
    """Round a positive FP32 value upward to the nearest power of two."""
    bits = T.reinterpret(value, "uint32")
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    ceil_exponent = T.Cast("int32", exponent - 127 + T.if_then_else(mantissa != 0, 1, 0))
    return T.reinterpret((ceil_exponent + 127) << 23, "float32")


def _mx_scale(amax, elem_offset: int):
    """Generate a standard MX scale from a block maximum."""
    bits = T.reinterpret(amax, "uint32")
    rounded_bits = (bits + (1 << 21)) & 0xFF800000
    rounded_amax = T.reinterpret(rounded_bits, "float32")
    scale = rounded_amax * (2.0 ** (-elem_offset))
    return T.max(scale, 2.0**-127)


def _fp8_scale(amax, scale_fmt: ScaleFormat | None, eps: float):
    safe_amax = T.max(amax, eps)
    if scale_fmt == "e8m0":
        return T.if_then_else(amax == 0, 2.0**-127, _mx_scale(safe_amax, 8))
    scale = safe_amax * (1.0 / 448.0)
    if scale_fmt == "ue8m0":
        scale = _round_up_to_power_of_two(scale)
    if scale_fmt is None:
        scale = T.if_then_else(amax == 0, 2.0**-23, scale)
    return scale


@cache
def _row_quant_kernel(
    cols: int,
    block_cols: int,
    in_dtype: str,
    scale_dtype: str,
    scale_fmt: ScaleFormat | None,
    dequant: bool,
    eps: float,
):
    """Compile row-wise quantization with several independent rows per CTA."""
    rows = T.dynamic("rows")
    scale_cols = cols // block_cols
    cta_rows = 32
    fp8_max = 448.0
    out_dtype = in_dtype if dequant else _FP8

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def row_quant(
            X: T.Tensor[(rows, cols), in_dtype],
            Y: T.Tensor[(rows, cols), out_dtype],
            S: T.Tensor[(rows, scale_cols), scale_dtype],
        ):
            with T.Kernel(scale_cols, T.ceildiv(rows, cta_rows), threads=128) as (pid_n, pid_m):
                x_shared = T.alloc_shared((cta_rows, block_cols), in_dtype)
                y_shared = T.alloc_shared((cta_rows, block_cols), out_dtype)
                amax_rows = T.alloc_fragment((cta_rows,), _FP32)
                scale_local = T.alloc_fragment((cta_rows,), _FP32)
                scale_shared = T.alloc_shared((cta_rows,), _FP32)

                T.copy(X[pid_m * cta_rows, pid_n * block_cols], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_rows, dim=1)
                for i in T.Parallel(cta_rows):
                    scale_local[i] = _fp8_scale(amax_rows[i], scale_fmt, eps)
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(cta_rows, block_cols):
                    quantized = T.Cast(_FP8, T.clamp(x_shared[i, j] / scale_shared[i], -fp8_max, fp8_max))
                    if dequant:
                        y_shared[i, j] = T.Cast(out_dtype, T.Cast(_FP32, quantized) * scale_shared[i])
                    else:
                        y_shared[i, j] = quantized

                T.copy(y_shared, Y[pid_m * cta_rows, pid_n * block_cols], disable_tma=True)
                for i in T.Parallel(cta_rows):
                    row = pid_m * cta_rows + i
                    if row < rows:
                        S[row, pid_n] = T.Cast(scale_dtype, scale_shared[i])

        return row_quant

    return kernel()


@cache
def _block_quant_kernel(
    cols: int,
    block_rows: int,
    block_cols: int,
    in_dtype: str,
    scale_dtype: str,
    scale_fmt: ScaleFormat | None,
    dequant: bool,
    eps: float,
):
    """Compile two-dimensional quantization with one independently scaled block per CTA."""
    rows = T.dynamic("rows")
    scale_rows = rows // block_rows
    scale_cols = cols // block_cols
    fp8_max = 448.0
    out_dtype = in_dtype if dequant else _FP8

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def block_quant(
            X: T.Tensor[(rows, cols), in_dtype],
            Y: T.Tensor[(rows, cols), out_dtype],
            S: T.Tensor[(scale_rows, scale_cols), scale_dtype],
        ):
            with T.Kernel(scale_cols, scale_rows, threads=128) as (pid_n, pid_m):
                x_shared = T.alloc_shared((block_rows, block_cols), in_dtype)
                y_shared = T.alloc_shared((block_rows, block_cols), out_dtype)
                amax_rows = T.alloc_fragment((block_rows,), _FP32)
                scale_local = T.alloc_fragment((1,), _FP32)
                scale_shared = T.alloc_shared((1,), _FP32)

                T.copy(X[pid_m * block_rows, pid_n * block_cols], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_rows, dim=1)
                T.reduce_absmax(amax_rows, scale_local, dim=0)
                scale_local[0] = _fp8_scale(scale_local[0], scale_fmt, eps)
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(block_rows, block_cols):
                    quantized = T.Cast(_FP8, T.clamp(x_shared[i, j] / scale_shared[0], -fp8_max, fp8_max))
                    if dequant:
                        y_shared[i, j] = T.Cast(out_dtype, T.Cast(_FP32, quantized) * scale_shared[0])
                    else:
                        y_shared[i, j] = quantized

                T.copy(y_shared, Y[pid_m * block_rows, pid_n * block_cols], disable_tma=True)
                if T.get_thread_binding(0) == 0:
                    S[pid_m, pid_n] = T.Cast(scale_dtype, scale_shared[0])

        return block_quant

    return kernel()


def _normalize_block_shape(block_shape: Sequence[int]) -> tuple[int, int]:
    if len(block_shape) != 2:
        raise ValueError(f"block_shape must contain two dimensions, got {tuple(block_shape)}")
    block_rows, block_cols = (int(size) for size in block_shape)
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(f"block dimensions must be positive, got {(block_rows, block_cols)}")
    return block_rows, block_cols


def fp8_quant(
    x: torch.Tensor,
    block_shape: Sequence[int] = (128, 128),
    *,
    scale_dtype: torch.dtype = torch.float32,
    scale_fmt: ScaleFormat | None = None,
    dequant: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last two dimensions of a tensor in independent FP8 blocks.

    Leading dimensions are treated as a batch of matrices. Matrix dimensions
    must be divisible by their corresponding block dimensions.

    Args:
        x: Contiguous or strided CUDA tensor in BF16, FP16, or FP32.
        block_shape: ``(rows, columns)`` sharing one scale. Common recipes use
            ``(128, 128)`` for FP8 weights, ``(1, 128)`` for FP8 activations,
            and ``(1, 32)`` for MXFP8 operands.
        scale_dtype: Scale storage dtype, either FP32 or E8M0.
        scale_fmt: ``None`` keeps the exact FP32 scale. ``"e8m0"`` and
            ``"ue8m0"`` round it upward to a power of two.
        dequant: Return the fused quantize-dequantize result in ``x.dtype``.
            Otherwise return the FP8 tensor and its scales.
        eps: Lower bound applied to each block's absolute maximum.

    Returns:
        If ``dequant`` is false, ``(quantized, scales)``. The scale shape is
        ``x.shape[:-2] + (ceil(rows / block_rows), ceil(cols / block_cols))``.
        If ``dequant`` is true, only the dequantized tensor is returned.
    """
    if x.ndim < 2:
        raise ValueError(f"fp8_quant expects at least two dimensions, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("fp8_quant expects a CUDA tensor")
    if x.dtype not in _TORCH_TO_TILELANG_DTYPE:
        raise TypeError(f"fp8_quant supports BF16, FP16, and FP32 inputs, got {x.dtype}")
    if x.shape[-2] == 0 or x.shape[-1] == 0:
        raise ValueError(f"fp8_quant does not support empty matrix dimensions, got shape {tuple(x.shape)}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if scale_fmt not in (None, "e8m0", "ue8m0"):
        raise ValueError(f"scale_fmt must be None, 'e8m0', or 'ue8m0', got {scale_fmt!r}")
    if scale_dtype not in (torch.float32, torch.float8_e8m0fnu):
        raise TypeError(f"scale_dtype must be torch.float32 or torch.float8_e8m0fnu, got {scale_dtype}")
    if scale_dtype == torch.float8_e8m0fnu and scale_fmt is None:
        raise ValueError("E8M0 scale storage requires scale_fmt='e8m0' or 'ue8m0'")

    block_rows, block_cols = _normalize_block_shape(block_shape)
    rows, cols = x.shape[-2:]
    if rows % block_rows or cols % block_cols:
        raise ValueError(f"matrix shape {(rows, cols)} is not divisible by block_shape {(block_rows, block_cols)}")
    scale_rows = rows // block_rows
    scale_cols = cols // block_cols
    scale_shape = (*x.shape[:-2], scale_rows, scale_cols)

    x = x.contiguous()
    output = torch.empty_like(x) if dequant else torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(scale_shape, dtype=scale_dtype, device=x.device)
    in_dtype = _TORCH_TO_TILELANG_DTYPE[x.dtype]
    tl_scale_dtype = _E8M0 if scale_dtype == torch.float8_e8m0fnu else _FP32

    if block_rows == 1:
        kernel = _row_quant_kernel(
            cols,
            block_cols,
            in_dtype,
            tl_scale_dtype,
            scale_fmt,
            dequant,
            eps,
        )
    else:
        kernel = _block_quant_kernel(
            cols,
            block_rows,
            block_cols,
            in_dtype,
            tl_scale_dtype,
            scale_fmt,
            dequant,
            eps,
        )
    kernel(
        x.view(-1, cols),
        output.view(-1, cols),
        scales.view(-1, scale_cols),
    )

    if dequant:
        return output
    return output, scales
