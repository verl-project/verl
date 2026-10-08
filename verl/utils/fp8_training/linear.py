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

"""Real FP8 (torch.float8_e4m3fn) training Linear layer.

Unlike verl.utils.qat.linear.QATLinear (which fake-quantizes in full precision
to simulate a future low-bit *export*), this module performs the forward
matmul on an actual float8 payload. The nn.Parameter stays in the model's
master dtype (bf16/fp32) -- FSDP sharding and the optimizer step are
unaffected -- and this module quantizes a fresh float8 copy from the master
weight on every forward call rather than caching one, which keeps the
autograd story simple and avoids any "did we forget to refresh after
optimizer.step()" staleness bug, at the cost of paying the quantize cost
every call (acceptable for this PoC; see verl/utils/fp8_training/core.py).

On hardware with no native low-precision matrix-engine path (confirmed true
for Intel XPU today -- see unslothai/unsloth#12535), this is expected to
reduce the memory footprint of the matmul operands without improving
throughput. Report both numbers; a flat-or-worse tok/s is an expected
outcome here, not a bug.
"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

_FP8_E4M3_MAX = 448.0  # max finite magnitude representable by torch.float8_e4m3fn


class FP8Mode(str, Enum):
    ROWWISE = "rowwise"  # torch._scaled_mm, one scale for the whole tensor
    BLOCKWISE = "blockwise"  # scaled_fp8_blockwise + dequant + regular matmul (slow fallback)


def _quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor absmax scale, cast to float8_e4m3fn. Returns (x_fp8, scale)."""
    amax = x.detach().abs().amax().clamp(min=1e-12).to(torch.float32)
    scale = (amax / _FP8_E4M3_MAX).reshape(1, 1)
    x_fp8 = (x.detach().to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def _dequant_blockwise(fp8_w: torch.Tensor, descale: torch.Tensor, block_size: tuple[int, int]) -> torch.Tensor:
    """Inverse of scaled_fp8_blockwise: reconstruct a full-precision tensor from
    its float8 blocks and per-block descale factors."""
    bm, bn = block_size
    m, n = fp8_w.shape
    w = fp8_w.to(torch.float32).reshape(m // bm, bm, n // bn, bn)
    w = w * descale.reshape(m // bm, 1, n // bn, 1)
    return w.reshape(m, n)


class _FP8RowwiseMatmulFn(torch.autograd.Function):
    """y = x @ weight.T computed via torch._scaled_mm on fresh per-call fp8 casts.

    Backward dequantizes the saved fp8 operands and uses standard matmuls --
    this introduces no further approximation beyond what forward already
    quantized (not a straight-through-estimator shortcut), at the cost of
    backward not itself being a fast fp8 op. That trade is intentional for a
    correctness-first PoC.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x_fp8, x_scale = _quantize_rowwise(x2d)
        w_fp8, w_scale = _quantize_rowwise(weight)
        out = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale, scale_b=w_scale, out_dtype=out_dtype)
        ctx.save_for_backward(x_fp8, x_scale, w_fp8, w_scale)
        ctx.out_dtype = out_dtype
        ctx.orig_shape = orig_shape
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_fp8, x_scale, w_fp8, w_scale = ctx.saved_tensors
        x = (x_fp8.to(ctx.out_dtype) * x_scale.to(ctx.out_dtype)).reshape(-1, x_fp8.shape[-1])
        w = w_fp8.to(ctx.out_dtype) * w_scale.to(ctx.out_dtype)
        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1]).to(ctx.out_dtype)
        grad_x = (grad_out2d @ w).reshape(ctx.orig_shape)
        grad_w = grad_out2d.t() @ x
        return grad_x, grad_w, None


class _FP8BlockwiseMatmulFn(torch.autograd.Function):
    """y = x @ weight.T via scaled_fp8_blockwise quantize-then-dequant-then-matmul.

    Intentionally not a fast fp8 GEMM -- mirrors Unsloth's documented XPU
    block-wise fallback path (quantize for the memory win, compute in full
    precision because no fast kernel exists to call instead).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, block_size: tuple[int, int], out_dtype: torch.dtype):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)
        w32 = weight.detach().to(torch.float32)

        x_fp8, x_descale = scaled_fp8_blockwise(x2d, list(block_size))
        w_fp8, w_descale = scaled_fp8_blockwise(w32, list(block_size))

        x_dq = _dequant_blockwise(x_fp8, x_descale, block_size).to(out_dtype)
        w_dq = _dequant_blockwise(w_fp8, w_descale, block_size).to(out_dtype)

        out = x_dq @ w_dq.t()
        ctx.save_for_backward(x_dq, w_dq)
        ctx.orig_shape = orig_shape
        return out.reshape(*orig_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_dq, w_dq = ctx.saved_tensors
        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1]).to(w_dq.dtype)
        grad_x = (grad_out2d @ w_dq).reshape(ctx.orig_shape)
        grad_w = grad_out2d.t() @ x_dq
        return grad_x, grad_w, None, None


class FP8Linear(nn.Linear):
    """nn.Linear whose forward matmul runs on a real float8_e4m3fn cast of its
    operands, quantized fresh from the (unmodified, master-dtype) weight on
    every call. See module docstring for why there's no persistent fp8 cache.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: FP8Mode | str = FP8Mode.ROWWISE,
        block_size: tuple[int, int] = (128, 128),
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fp8_mode = FP8Mode(mode)
        self.block_size = tuple(block_size)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: FP8Mode | str = FP8Mode.ROWWISE,
        block_size: tuple[int, int] = (128, 128),
    ) -> "FP8Linear":
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            mode=mode,
            block_size=block_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_mode == FP8Mode.ROWWISE:
            out = _FP8RowwiseMatmulFn.apply(x, self.weight, x.dtype)
        else:
            out = _FP8BlockwiseMatmulFn.apply(x, self.weight, self.block_size, x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, fp8_mode={self.fp8_mode.value}, block_size={self.block_size}"
