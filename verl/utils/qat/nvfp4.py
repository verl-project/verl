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

"""Engine-neutral NVFP4 weight packing for trainer-side export.

:func:`quantize_nvfp4` follows the arithmetic of the CUDA NVFP4 quantization kernel the
inference engines share (vLLM ``scaled_fp4_quant``, which vLLM's online NVFP4 methods use):
16-element blocks, an E4M3 block scale ``global_scale * amax / 6`` and round-to-nearest-even
E2M1 values. The kernel computes reciprocals with ``rcp.approx.ftz.f32``, so this Triton
kernel does too; a GPU test checks it bitwise against vLLM.
"""

import torch
import triton
import triton.language as tl

__all__ = ["quantize_nvfp4"]

_BLOCK = 16


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
    block_scale = (global_scale * (amax * _rcp_approx(tl.full(amax.shape, 6.0, tl.float32)))).to(tl.float8e4nv)
    scale = block_scale.to(tl.float32)
    inv_global = _rcp_approx(tl.full(amax.shape, 0.0, tl.float32) + global_scale)
    out_scale = tl.where(scale != 0.0, _rcp_approx(scale * inv_global), 0.0)

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


def quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D BF16/FP16 CUDA tensor to NVFP4 with a given float32 global scale.

    Returns the packed E2M1 values ``[M, N / 2]`` (uint8, low nibble first) and the
    row-major E4M3 block scales ``[M, N / 16]``.
    """
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda:
        raise ValueError(f"NVFP4 quantization takes a 2D BF16/FP16 CUDA tensor, got {x.dtype} {tuple(x.shape)}")
    m, n = x.shape
    if n % _BLOCK:
        raise ValueError(f"NVFP4 quantization needs a multiple of {_BLOCK} columns, got {n}")
    x = x.contiguous()
    global_scale = global_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty((m, n // _BLOCK), dtype=torch.uint8, device=x.device)
    tile_m, blocks = 16, 16
    grid = (triton.cdiv(m, tile_m), triton.cdiv(n // _BLOCK, blocks))
    _quantize_nvfp4_kernel[grid](x, packed, scale, global_scale, m, n, x.stride(0), TILE_M=tile_m, BLOCKS=blocks)
    return packed, scale.view(torch.float8_e4m3fn)
