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

from collections.abc import Sequence
from functools import cache
from typing import Literal

import torch

try:
    import tilelang
    import tilelang.language as T
except ModuleNotFoundError:
    tilelang = None
    T = None

if tilelang is not None:
    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
else:
    pass_configs = {}

__all__ = ["fp8_grouped_fake_quant", "fp8_quant"]

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


def _fp8_scale(amax, scale_fmt: ScaleFormat | None):
    """Match DeepSeek-V4's FP8 scale recipe."""
    scale = T.max(amax, 1e-4) * (1.0 / 448.0)
    if scale_fmt is not None:
        scale = _round_up_to_power_of_two(scale)
    return scale


@cache
def _row_quant_kernel(
    cols: int,
    block_cols: int,
    in_dtype: str,
    scale_dtype: str,
    scale_fmt: ScaleFormat | None,
    dequant: bool,
):
    """Compile row-wise quantization with several independent rows per CTA."""
    assert tilelang is not None and T is not None
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
                    scale_local[i] = _fp8_scale(amax_rows[i], scale_fmt)
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
):
    """Compile two-dimensional quantization with one independently scaled block per CTA."""
    assert tilelang is not None and T is not None
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
                scale_local[0] = _fp8_scale(scale_local[0], scale_fmt)
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


@cache
def _grouped_row_fake_quant_kernel(
    num_groups: int,
    rows: int,
    cols: int,
    block_cols: int,
    in_dtype: str,
    scale_fmt: ScaleFormat | None,
):
    """Compile row-wise QDQ over independently allocated expert matrices."""
    assert tilelang is not None and T is not None
    scale_cols = cols // block_cols
    cta_rows = 32
    fp8_max = 448.0

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def grouped_row_fake_quant(
            X_ptrs: T.Tensor[(num_groups,), T.ptr],
            Y: T.Tensor[(num_groups, rows, cols), in_dtype],
        ):
            with T.Kernel(scale_cols, num_groups, T.ceildiv(rows, cta_rows), threads=128) as (
                pid_n,
                pid_g,
                pid_m,
            ):
                X = T.make_tensor(X_ptrs[pid_g], (rows, cols), in_dtype)
                x_shared = T.alloc_shared((cta_rows, block_cols), in_dtype)
                y_shared = T.alloc_shared((cta_rows, block_cols), in_dtype)
                amax_rows = T.alloc_fragment((cta_rows,), _FP32)
                scale_local = T.alloc_fragment((cta_rows,), _FP32)
                scale_shared = T.alloc_shared((cta_rows,), _FP32)

                T.copy(X[pid_m * cta_rows, pid_n * block_cols], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_rows, dim=1)
                for i in T.Parallel(cta_rows):
                    scale_local[i] = _fp8_scale(amax_rows[i], scale_fmt)
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(cta_rows, block_cols):
                    quantized = T.Cast(_FP8, T.clamp(x_shared[i, j] / scale_shared[i], -fp8_max, fp8_max))
                    y_shared[i, j] = T.Cast(in_dtype, T.Cast(_FP32, quantized) * scale_shared[i])

                T.copy(
                    y_shared,
                    Y[pid_g, pid_m * cta_rows, pid_n * block_cols],
                    disable_tma=True,
                )

        return grouped_row_fake_quant

    return kernel()


@cache
def _grouped_block_fake_quant_kernel(
    num_groups: int,
    rows: int,
    cols: int,
    block_rows: int,
    block_cols: int,
    in_dtype: str,
    scale_fmt: ScaleFormat | None,
):
    """Compile 2D block QDQ over independently allocated expert matrices."""
    assert tilelang is not None and T is not None
    scale_rows = rows // block_rows
    scale_cols = cols // block_cols
    fp8_max = 448.0

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def grouped_block_fake_quant(
            X_ptrs: T.Tensor[(num_groups,), T.ptr],
            Y: T.Tensor[(num_groups, rows, cols), in_dtype],
        ):
            with T.Kernel(scale_cols, num_groups, scale_rows, threads=128) as (pid_n, pid_g, pid_m):
                X = T.make_tensor(X_ptrs[pid_g], (rows, cols), in_dtype)
                x_shared = T.alloc_shared((block_rows, block_cols), in_dtype)
                y_shared = T.alloc_shared((block_rows, block_cols), in_dtype)
                amax_rows = T.alloc_fragment((block_rows,), _FP32)
                scale_local = T.alloc_fragment((1,), _FP32)
                scale_shared = T.alloc_shared((1,), _FP32)

                T.copy(X[pid_m * block_rows, pid_n * block_cols], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_rows, dim=1)
                T.reduce_absmax(amax_rows, scale_local, dim=0)
                scale_local[0] = _fp8_scale(scale_local[0], scale_fmt)
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(block_rows, block_cols):
                    quantized = T.Cast(_FP8, T.clamp(x_shared[i, j] / scale_shared[0], -fp8_max, fp8_max))
                    y_shared[i, j] = T.Cast(in_dtype, T.Cast(_FP32, quantized) * scale_shared[0])

                T.copy(
                    y_shared,
                    Y[pid_g, pid_m * block_rows, pid_n * block_cols],
                    disable_tma=True,
                )

        return grouped_block_fake_quant

    return kernel()


def _normalize_block_shape(block_shape: Sequence[int]) -> tuple[int, int]:
    if len(block_shape) != 2:
        raise ValueError(f"block_shape must contain two dimensions, got {tuple(block_shape)}")
    block_rows, block_cols = (int(size) for size in block_shape)
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(f"block dimensions must be positive, got {(block_rows, block_cols)}")
    return block_rows, block_cols


def _use_tilelang() -> bool:
    return tilelang is not None and torch.version.hip is None


def _round_up_to_power_of_two_torch(value: torch.Tensor) -> torch.Tensor:
    """Round a positive FP32 tensor upward to the nearest power of two."""
    bits = value.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    ceil_exponent = exponent - 127 + (mantissa != 0).to(torch.int32)
    return ((ceil_exponent + 127) << 23).view(torch.float32)


def _fp8_scale_torch(amax: torch.Tensor, scale_fmt: ScaleFormat | None) -> torch.Tensor:
    scale = amax.clamp_min(1e-4) / 448.0
    if scale_fmt is not None:
        scale = _round_up_to_power_of_two_torch(scale)
    return scale


def _fp8_quant_torch(
    x: torch.Tensor,
    block_rows: int,
    block_cols: int,
    *,
    scale_dtype: torch.dtype,
    scale_fmt: ScaleFormat | None,
    dequant: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape[-2:]
    scale_rows = rows // block_rows
    scale_cols = cols // block_cols
    blocks = x.float().reshape(*x.shape[:-2], scale_rows, block_rows, scale_cols, block_cols)
    scales_f32 = _fp8_scale_torch(blocks.abs().amax(dim=(-3, -1)), scale_fmt)
    quantized = (blocks / scales_f32.unsqueeze(-1).unsqueeze(-3)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if dequant:
        return (quantized.float() * scales_f32.unsqueeze(-1).unsqueeze(-3)).reshape(x.shape).to(x.dtype)
    return quantized.reshape(x.shape).contiguous(), scales_f32.to(scale_dtype)


def fp8_grouped_fake_quant(
    tensors: Sequence[torch.Tensor],
    block_shape: Sequence[int] = (128, 128),
    *,
    scale_fmt: ScaleFormat | None = None,
) -> torch.Tensor:
    """Fake-quantize separate expert matrices without first stacking their inputs."""
    if not tensors:
        raise ValueError("fp8_grouped_fake_quant expects at least one tensor")
    first = tensors[0]
    if first.ndim != 2:
        raise ValueError(f"fp8_grouped_fake_quant expects 2D tensors, got shape {tuple(first.shape)}")
    if not first.is_cuda:
        raise ValueError("fp8_grouped_fake_quant expects CUDA tensors")
    if first.dtype not in _TORCH_TO_TILELANG_DTYPE:
        raise TypeError(f"fp8_grouped_fake_quant supports BF16, FP16, and FP32 inputs, got {first.dtype}")
    if scale_fmt not in (None, "e8m0", "ue8m0"):
        raise ValueError(f"scale_fmt must be None, 'e8m0', or 'ue8m0', got {scale_fmt!r}")
    if any(
        tensor.shape != first.shape or tensor.dtype != first.dtype or tensor.device != first.device
        for tensor in tensors
    ):
        raise ValueError("fp8_grouped_fake_quant requires tensors with identical shapes, dtypes, and devices")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fp8_grouped_fake_quant requires contiguous tensors")

    block_rows, block_cols = _normalize_block_shape(block_shape)
    rows, cols = first.shape
    if rows % block_rows or cols % block_cols:
        raise ValueError(f"matrix shape {(rows, cols)} is not divisible by block_shape {(block_rows, block_cols)}")

    if not _use_tilelang():
        stacked = torch.stack(tuple(tensors), dim=0)
        result = _fp8_quant_torch(
            stacked,
            block_rows,
            block_cols,
            scale_dtype=torch.float32,
            scale_fmt=scale_fmt,
            dequant=True,
        )
        assert isinstance(result, torch.Tensor)
        return result

    pointer_table = torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        device=first.device,
    )
    output = torch.empty((len(tensors), rows, cols), dtype=first.dtype, device=first.device)
    kernel_args = (len(tensors), rows, cols)
    if block_rows == 1:
        kernel = _grouped_row_fake_quant_kernel(
            *kernel_args,
            block_cols,
            _TORCH_TO_TILELANG_DTYPE[first.dtype],
            scale_fmt,
        )
    else:
        kernel = _grouped_block_fake_quant_kernel(
            *kernel_args,
            block_rows,
            block_cols,
            _TORCH_TO_TILELANG_DTYPE[first.dtype],
            scale_fmt,
        )
    kernel(pointer_table, output)
    return output


def fp8_quant(
    x: torch.Tensor,
    block_shape: Sequence[int] = (128, 128),
    *,
    scale_dtype: torch.dtype = torch.float32,
    scale_fmt: ScaleFormat | None = None,
    dequant: bool = False,
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
        scale_fmt: ``None`` keeps the FP32 scale. ``"e8m0"`` and ``"ue8m0"``
            both select DeepSeek's upward power-of-two rounding recipe.
        dequant: Return the fused quantize-dequantize result in ``x.dtype``.
            Otherwise return the FP8 tensor and its scales.
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
    if not _use_tilelang():
        return _fp8_quant_torch(
            x,
            block_rows,
            block_cols,
            scale_dtype=scale_dtype,
            scale_fmt=scale_fmt,
            dequant=dequant,
        )
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
        )
    kernel(
        x.view(-1, cols),
        output.view(-1, cols),
        scales.view(-1, scale_cols),
    )

    if dequant:
        return output
    return output, scales
