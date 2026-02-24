# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import logging
import os

import torch

logger = logging.getLogger(__name__)

# Check if Triton is available
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    logger.debug("Triton not available, FP8 Triton kernels will not be used")

# Environment variable to control Triton FP8 usage (set to "1" to disable)
_DISABLE_TRITON_FP8 = os.environ.get("VERL_DISABLE_TRITON_FP8", "0").lower() in ("1", "true", "yes")

# FP8 constants
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = -FP8_MAX


def ceil_div(x: int, y: int) -> int:
    """Perform ceiling division of two integers."""
    return (x + y - 1) // y


def is_triton_available() -> bool:
    """Check if Triton is available for FP8 kernels."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _blockwise_cast_to_fp8_kernel(
        X,
        Y,
        S,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        eps,
        fp8_min,
        fp8_max,
        BLOCK_M: tl.constexpr = 128,
        BLOCK_N: tl.constexpr = 128,
    ):
        """Triton kernel for blockwise FP8 quantization.

        Each program instance handles one block of size (BLOCK_M, BLOCK_N).
        Computes per-block scale and quantizes to FP8 in a single pass.

        Refer to https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/kernels/fp8_kernel.py
        """
        pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
        pid_n = tl.cast(tl.program_id(axis=1), tl.int64)

        # Compute block offsets
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Create masks for boundary handling
        mask_m = off_m < M
        mask_n = off_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        # Load input block and convert to float32 for precision
        x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)

        # Compute block-wise absolute maximum with epsilon for numerical stability
        _absmax = tl.maximum(tl.max(tl.abs(x)), eps)

        # Compute scale: scale = absmax / fp8_max
        x_s = _absmax / fp8_max

        # Compute inverse scale for quantization
        s_inv = 1.0 / x_s

        # Quantize: clamp(x * s_inv, fp8_min, fp8_max)
        y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

        # Store quantized values and scale
        tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
        tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)

    def blockwise_cast_to_fp8_triton(
        x: torch.Tensor,
        weight_block_size: list[int] | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 2D tensor to FP8 using blockwise quantization with Triton.

        This function provides high-performance FP8 quantization with minimal memory overhead.
        All computations (abs, max, scale, clamp) are performed in a single Triton kernel,
        eliminating intermediate tensor allocations.

        Args:
            x: Input tensor of shape (M, N), must be 2D.
            weight_block_size: Block size for quantization as [BLOCK_M, BLOCK_N].
                              Defaults to [128, 128] if None.

        Returns:
            Tuple of (quantized_tensor, scale_tensor):
                - quantized_tensor: FP8 quantized tensor of shape (M, N)
                - scale_tensor: Per-block scale factors of shape (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
                               This is the inverse scale (multiply to dequantize).
        """
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"

        # Default block size
        BLOCK_M, BLOCK_N = 128, 128
        if weight_block_size is not None:
            BLOCK_M, BLOCK_N = weight_block_size[0], weight_block_size[1]

        M, N = x.shape

        # Pre-allocate output tensors (only memory allocation in this function)
        y = torch.empty(M, N, device=x.device, dtype=FP8_DTYPE)
        s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)

        # Grid: one program per block
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        # Tune kernel parameters based on memory layout
        if x.is_contiguous():
            kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
        else:
            kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}

        # Launch kernel
        _blockwise_cast_to_fp8_kernel[grid](
            x,
            y,
            s,
            *x.stride(),
            *y.stride(),
            *s.stride(),
            M,
            N,
            1e-10,  # eps for numerical stability
            FP8_MIN,
            FP8_MAX,
            **kwargs,
        )

        return y, s


def scaled_fp8_blockwise_triton(
    data_hp: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-performance FP8 blockwise quantization using Triton kernel.

    This is the recommended function to use for FP8 quantization when Triton is available.
    It handles padding automatically and returns results in the expected format.

    Args:
        data_hp: Input high-precision tensor of shape (M, N).
        weight_block_size: Block size for quantization as [BLOCK_M, BLOCK_N].

    Returns:
        Tuple of (fp8_data, descale):
            - fp8_data: FP8 quantized tensor of original shape
            - descale: Per-block descale factors (inverse of scale, for dequantization)

    Raises:
        RuntimeError: If Triton is not available.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for scaled_fp8_blockwise_triton but is not available")

    block_size0 = weight_block_size[0]
    block_size1 = weight_block_size[1]

    # Save original shape for potential cropping
    original_shape = data_hp.shape

    # Pad dimensions to be multiples of block size if needed
    pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
    pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1

    if pad_dim0 > 0 or pad_dim1 > 0:
        logger.debug(
            f"Padding weight from {data_hp.shape} to "
            f"({data_hp.shape[0] + pad_dim0}, {data_hp.shape[1] + pad_dim1}) "
            f"for blockwise FP8 quantization"
        )
        data_hp = torch.nn.functional.pad(data_hp, (0, pad_dim1, 0, pad_dim0), mode="constant", value=0)

    # Call Triton kernel
    fp_data, scale = blockwise_cast_to_fp8_triton(data_hp, weight_block_size)

    # Remove padding to restore original shape
    if pad_dim0 > 0 or pad_dim1 > 0:
        fp_data = fp_data[: original_shape[0], : original_shape[1]].contiguous()

    # Return scale as descale (the Triton kernel returns scale, we need to return it as-is
    # since it's already the inverse scale format expected by vLLM/SGLang)
    return fp_data, scale


def _scaled_fp8_blockwise_pytorch(
    data_hp: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of blockwise FP8 quantization.

    Memory-optimized implementation that:
    - Uses in-place operations where possible
    - Explicitly deletes intermediate tensors
    - Minimizes peak memory usage during quantization

    Args:
        data_hp: Input high-precision tensor of shape (M, N).
        weight_block_size: Block size for quantization as [BLOCK_M, BLOCK_N].

    Returns:
        Tuple of (fp8_data, descale):
            - fp8_data: FP8 quantized tensor
            - descale: Per-block descale factors for dequantization
    """
    block_size0 = weight_block_size[0]
    block_size1 = weight_block_size[1]
    assert block_size0 == block_size1, "Block sizes must be equal"

    # Save unpadded shape for later cropping
    original_shape = data_hp.shape

    # Pad dimensions to be multiples of block size if needed
    pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
    pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1

    if pad_dim0 > 0 or pad_dim1 > 0:
        logger.debug(
            f"Padding weight from {data_hp.shape} to "
            f"({data_hp.shape[0] + pad_dim0}, {data_hp.shape[1] + pad_dim1}) "
            f"for blockwise FP8 quantization"
        )
        data_hp = torch.nn.functional.pad(data_hp, (0, pad_dim1, 0, pad_dim0), mode="constant", value=0)

    # FP8
    max_dtype = FP8_MAX

    padded_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    # Reshape and permute - these are views, no memory allocation
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3).contiguous()

    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) in float32 for precision
    data_hp = data_hp.to(torch.float32).flatten(start_dim=2)

    # Calculate max absolute value per block - use fused abs+amax
    max_abs = data_hp.abs().amax(dim=-1, keepdim=True)

    # Compute scale in-place where possible
    scale_fp = torch.empty_like(max_abs)
    torch.div(max_dtype, max_abs, out=scale_fp)
    # Handle edge cases: zero and inf
    scale_fp = torch.where(max_abs == 0, torch.ones_like(scale_fp), scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, torch.ones_like(scale_fp), scale_fp)
    del max_abs  # Free max_abs memory

    # Compute descale before modifying data
    descale_fp = torch.reciprocal(scale_fp)

    # Scale and clamp in a memory-efficient way
    data_hp.mul_(scale_fp)
    del scale_fp  # Free scale memory
    data_hp.clamp_(min=-max_dtype, max=max_dtype)

    # Convert to FP8
    fp_data = data_hp.to(FP8_DTYPE)
    del data_hp  # Free float32 data

    # Reshape back to original layout
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(padded_shape)

    # Remove padding to restore original shape
    if original_shape[0] != padded_shape[0] or original_shape[1] != padded_shape[1]:
        fp_data = fp_data[: original_shape[0], : original_shape[1]].contiguous()

    return fp_data, descale_fp


def scaled_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast tensor from high precision to FP8 with blockwise quantization.

    This function automatically selects the best available implementation:
    1. Triton kernel (if available): Highest performance, minimal memory overhead
    2. PyTorch fallback: Memory-optimized implementation using in-place operations

    To disable Triton and force PyTorch fallback, set environment variable:
        VERL_DISABLE_TRITON_FP8=1

    Args:
        data_hp: Input tensor of shape (M, N) in high precision (bf16/fp16/fp32).
        weight_block_size: Block size for quantization as [BLOCK_M, BLOCK_N].

    Returns:
        Tuple of (fp8_data, descale):
            - fp8_data: FP8 quantized tensor
            - descale: Per-block descale factors for dequantization
    """
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    # Use Triton kernel if available and not disabled
    if _TRITON_AVAILABLE and not _DISABLE_TRITON_FP8:
        return scaled_fp8_blockwise_triton(data_hp, weight_block_size)

    # PyTorch fallback implementation (memory-optimized)
    return _scaled_fp8_blockwise_pytorch(data_hp, weight_block_size)
