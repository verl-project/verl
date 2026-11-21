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

"""MXFP4 E2M1 row-wise quantization shared by QAT and weight export."""

from functools import cache

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

__all__ = ["MXFP4_GROUP_SIZE", "fp4_grouped_fake_quant", "fp4_quant"]

MXFP4_GROUP_SIZE = 32

_FP4 = "float4_e2m1fn"
_E8M0 = "float8_e8m0fnu"
_FP32 = "float32"
_TORCH_TO_TILELANG_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _round_up_to_power_of_two(value):
    """Round a positive FP32 value upward to the nearest power of two."""
    bits = T.reinterpret(value, "uint32")
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    ceil_exponent = T.Cast("int32", exponent - 127 + T.if_then_else(mantissa != 0, 1, 0))
    return T.reinterpret((ceil_exponent + 127) << 23, "float32")


def _mxfp4_scale(amax):
    """Match DeepSeek-V4's FP4 UE8M0 scale recipe."""
    scale = T.max(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
    return _round_up_to_power_of_two(scale)


@cache
def _fp4_quant_kernel(
    cols: int,
    in_dtype: str,
    dequant: bool,
):
    """Compile row-wise MXFP4 quantization with 32 rows per CTA."""
    rows = T.dynamic("rows")
    tile_rows = 32
    fp4_max = 6.0
    out_dtype = in_dtype if dequant else _FP4

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def fp4_quant(
            X: T.Tensor[(rows, cols), in_dtype],
            Y: T.Tensor[(rows, cols), out_dtype],
            S: T.Tensor[(rows, cols // MXFP4_GROUP_SIZE), _E8M0],
        ):
            with T.Kernel(T.ceildiv(rows, tile_rows), cols // MXFP4_GROUP_SIZE, threads=128) as (pid_m, pid_n):
                x_shared = T.alloc_shared((tile_rows, MXFP4_GROUP_SIZE), in_dtype)
                y_shared = T.alloc_shared((tile_rows, MXFP4_GROUP_SIZE), out_dtype)
                amax_local = T.alloc_fragment((tile_rows,), _FP32)
                scale_local = T.alloc_fragment((tile_rows,), _FP32)
                scale_shared = T.alloc_shared((tile_rows,), _FP32)

                T.copy(X[pid_m * tile_rows, pid_n * MXFP4_GROUP_SIZE], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_local, dim=1)
                for i in T.Parallel(tile_rows):
                    scale_local[i] = _mxfp4_scale(amax_local[i])
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(tile_rows, MXFP4_GROUP_SIZE):
                    quantized = T.Cast(
                        _FP4,
                        T.clamp(x_shared[i, j] / scale_shared[i], -fp4_max, fp4_max),
                    )
                    if dequant:
                        y_shared[i, j] = T.Cast(out_dtype, T.Cast(_FP32, quantized) * scale_shared[i])
                    else:
                        y_shared[i, j] = quantized

                T.copy(
                    y_shared,
                    Y[pid_m * tile_rows, pid_n * MXFP4_GROUP_SIZE],
                    disable_tma=True,
                )
                for i in T.Parallel(tile_rows):
                    row = pid_m * tile_rows + i
                    if row < rows:
                        S[row, pid_n] = T.Cast(_E8M0, scale_shared[i])

        return fp4_quant

    return kernel()


@cache
def _fp4_grouped_fake_quant_kernel(
    num_groups: int,
    rows: int,
    cols: int,
    in_dtype: str,
):
    """Compile MXFP4 QDQ over independently allocated expert matrices."""
    tile_rows = 32
    fp4_max = 6.0

    @tilelang.jit(pass_configs=pass_configs)
    def kernel():
        @T.prim_func
        def fp4_grouped_fake_quant(
            X_ptrs: T.Tensor[(num_groups,), T.ptr],
            Y: T.Tensor[(num_groups, rows, cols), in_dtype],
        ):
            with T.Kernel(
                num_groups,
                T.ceildiv(rows, tile_rows),
                cols // MXFP4_GROUP_SIZE,
                threads=128,
            ) as (pid_g, pid_m, pid_n):
                X = T.make_tensor(X_ptrs[pid_g], (rows, cols), in_dtype)
                x_shared = T.alloc_shared((tile_rows, MXFP4_GROUP_SIZE), in_dtype)
                y_shared = T.alloc_shared((tile_rows, MXFP4_GROUP_SIZE), in_dtype)
                amax_local = T.alloc_fragment((tile_rows,), _FP32)
                scale_local = T.alloc_fragment((tile_rows,), _FP32)
                scale_shared = T.alloc_shared((tile_rows,), _FP32)

                T.copy(X[pid_m * tile_rows, pid_n * MXFP4_GROUP_SIZE], x_shared, disable_tma=True)
                T.reduce_absmax(x_shared, amax_local, dim=1)
                for i in T.Parallel(tile_rows):
                    scale_local[i] = _mxfp4_scale(amax_local[i])
                T.copy(scale_local, scale_shared, disable_tma=True)

                for i, j in T.Parallel(tile_rows, MXFP4_GROUP_SIZE):
                    quantized = T.Cast(
                        _FP4,
                        T.clamp(x_shared[i, j] / scale_shared[i], -fp4_max, fp4_max),
                    )
                    y_shared[i, j] = T.Cast(in_dtype, T.Cast(_FP32, quantized) * scale_shared[i])

                T.copy(
                    y_shared,
                    Y[pid_g, pid_m * tile_rows, pid_n * MXFP4_GROUP_SIZE],
                    disable_tma=True,
                )

        return fp4_grouped_fake_quant

    return kernel()


def fp4_grouped_fake_quant(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Fake-quantize separate expert matrices without first stacking their inputs."""
    if not tensors:
        raise ValueError("fp4_grouped_fake_quant expects at least one tensor")
    first = tensors[0]
    if first.ndim != 2:
        raise ValueError(f"fp4_grouped_fake_quant expects 2D tensors, got shape {tuple(first.shape)}")
    if not first.is_cuda:
        raise ValueError("fp4_grouped_fake_quant expects CUDA tensors")
    if first.dtype not in _TORCH_TO_TILELANG_DTYPE:
        raise TypeError(f"fp4_grouped_fake_quant supports BF16, FP16, and FP32 inputs, got {first.dtype}")
    if any(
        tensor.shape != first.shape or tensor.dtype != first.dtype or tensor.device != first.device
        for tensor in tensors
    ):
        raise ValueError("fp4_grouped_fake_quant requires tensors with identical shapes, dtypes, and devices")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fp4_grouped_fake_quant requires contiguous tensors")

    rows, cols = first.shape
    if cols == 0 or cols % MXFP4_GROUP_SIZE:
        raise ValueError(f"the last dimension must be a nonzero multiple of {MXFP4_GROUP_SIZE}, got {cols}")

    pointer_table = torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        device=first.device,
    )
    output = torch.empty((len(tensors), rows, cols), dtype=first.dtype, device=first.device)
    kernel = _fp4_grouped_fake_quant_kernel(
        len(tensors),
        rows,
        cols,
        _TORCH_TO_TILELANG_DTYPE[first.dtype],
    )
    kernel(pointer_table, output)
    return output


def fp4_quant(
    x: torch.Tensor,
    *,
    dequant: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension using the MXFP4 1x32 recipe.

    Leading dimensions are flattened into independent rows. The quantized
    output packs two E2M1 values per byte, while each 32-value group stores one
    E8M0 scale. With ``dequant=True``, the kernel returns the fused QDQ result
    in the input dtype for use with a straight-through estimator.
    """
    if x.ndim < 1:
        raise ValueError("fp4_quant expects at least one dimension")
    if not x.is_cuda:
        raise ValueError("fp4_quant expects a CUDA tensor")
    if x.dtype not in _TORCH_TO_TILELANG_DTYPE:
        raise TypeError(f"fp4_quant supports BF16, FP16, and FP32 inputs, got {x.dtype}")
    cols = x.shape[-1]
    if cols == 0 or cols % MXFP4_GROUP_SIZE:
        raise ValueError(
            f"the last dimension must be a nonzero multiple of {MXFP4_GROUP_SIZE}, got shape {tuple(x.shape)}"
        )

    x = x.contiguous()
    scale_shape = (*x.shape[:-1], cols // MXFP4_GROUP_SIZE)
    scales = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=x.device)
    if dequant:
        output = torch.empty_like(x)
    else:
        output = torch.empty((*x.shape[:-1], cols // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)

    kernel = _fp4_quant_kernel(
        cols,
        _TORCH_TO_TILELANG_DTYPE[x.dtype],
        dequant,
    )
    kernel(
        x.view(-1, cols),
        output.view(-1, output.shape[-1]),
        scales.view(-1, cols // MXFP4_GROUP_SIZE),
    )

    if dequant:
        return output
    return output, scales
