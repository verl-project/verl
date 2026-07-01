# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import prepare_chunk_indices, make_tensor_descriptor, input_guard, is_amd


def _ensure_slice_ops() -> bool:
    """Probe and attach tl.extract_slice / insert_slice if missing; return success."""
    if hasattr(tl, "extract_slice") and hasattr(tl, "insert_slice"):
        return True
    try:
        from triton.language.extra.cann.extension import extract_slice, insert_slice
        tl.extract_slice = extract_slice
        tl.insert_slice = insert_slice
        return True
    except ImportError:
        return False

_TRITON_SLICE_AVAILABLE: bool = _ensure_slice_ops()
FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_loop_kernel_paral_v3(
        A_ptr,
        Ad_ptr,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        LARGE_BLOCK_T: tl.constexpr,
        NT: tl.constexpr,
        BH: tl.constexpr,
):
    worker_id = tl.program_id(0)
    total_tasks = NT * BH
    num_tasks = total_tasks // 48
    remainder = total_tasks - num_tasks * 48
    upper_bound = min(total_tasks, num_tasks * (worker_id + 1) + min(worker_id + 1, remainder))
    lower_bound = num_tasks * worker_id + min(worker_id, remainder)
    for task_id in range(lower_bound, upper_bound):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                chunk_indices + i_t * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        A = A_ptr + (bos * H + i_h) * BT
        Ad = Ad_ptr + (bos * H + i_h) * 16

        base_t = i_t * LARGE_BLOCK_T

        NTASKS: tl.constexpr = 2
        N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

        for taskid in range(0, NTASKS):
            base_t += taskid * (LARGE_BLOCK_T // NTASKS)

            b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
            for blkid in range(0, N_BLOCKS):
                row_start_o = base_t + blkid * 16
                col_start_o = row_start_o % BT
                # using ptr with mask instead of tl.load(block_ptr)
                offs_rows_in_block = tl.arange(0, 16)
                offs_cols_in_block = tl.arange(0, 16)
                ptr_A_subrec16 = (
                        A
                        + row_start_o * H * BT
                        + col_start_o
                        + offs_rows_in_block[:, None] * H * BT
                        + offs_cols_in_block[None, :]
                )
                global_rows = row_start_o + offs_rows_in_block[:, None]
                global_cols = col_start_o + offs_cols_in_block[None, :]
                load_mask = (global_rows < T) & (global_cols < BT)
                b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                    tl.float32
                )
                b_A = tl.insert_slice(
                    ful=b_A,
                    sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                    offsets=[blkid, 0, 0],
                    sizes=[1, 16, 16],
                    strides=[1, 1, 1],
                )

            # load multi 16x16
            local_ori_A = tl.trans(b_A, (1, 0, 2))
            local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

            # change mask into matrix elementwise action
            tmp = tl.arange(0, 16).to(tl.float32)
            rows = tmp[:, None]
            cols = tmp[None, :]
            is_lower = (rows > cols).to(b_A.dtype)
            b_A = -b_A * is_lower

            for i in range(1, 16):
                nblks_vec16 = -tl.extract_slice(
                    local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
                )
                b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

                dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
                dot_product = tl.sum(dot_tmp, 0)
                b_a = b_a + dot_product  # (N_BLOCKS, 16)

                b_a_new_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
                b_A = tl.insert_slice(
                    ful=b_A,
                    sub=b_a_new_expanded,
                    offsets=[0, i, 0],
                    sizes=[N_BLOCKS, 1, 16],
                    strides=[1, 1, 1],
                )

            on_diagonal = rows == cols
            b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

            b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
            # using ptr with mask instead of tl.load(block_ptr)
            offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
            offs_cols_to_store = tl.arange(0, 16)
            p_Ai = (
                    Ad
                    + base_t * H * 16
                    + 0
                    + offs_rows_to_store[:, None] * H * 16
                    + offs_cols_to_store[None, :]
            )
            global_store_rows = base_t + offs_rows_to_store[:, None]
            store_mask = global_store_rows < T
            tl.store(
                p_Ai,
                b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
                mask=store_mask,
            )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_loop_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NT: tl.constexpr,
        BH: tl.constexpr,
):
    worker_id = tl.program_id(0)
    total_tasks = NT * BH
    num_tasks = total_tasks // 24
    remainder = total_tasks - num_tasks * 24
    upper_bound = min(total_tasks, num_tasks * (worker_id + 1) + min(worker_id + 1, remainder))
    lower_bound = num_tasks * worker_id + min(worker_id, remainder)
    for task_id in range(lower_bound, upper_bound):
        i_tt = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_tt * 2).to(tl.int32), tl.load(
                chunk_indices + i_tt * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_tt

        A_ptr = A + (bos * H + i_h) * BT
        Ad_ptr = Ad + (bos * H + i_h) * 16
        Ai_ptr = Ai + (bos * H + i_h) * 32

        p_A_21 = tl.make_block_ptr(
            A_ptr, (T, BT), (H * BT, 1), (i_t * 32 + 16, 0 + i_t % (BT // 32) * 32), (16, 16), (1, 0)
        )
        p_Ad_11 = tl.make_block_ptr(
            Ad_ptr, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
        )
        p_Ad_22 = tl.make_block_ptr(
            Ad_ptr, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_11 = tl.make_block_ptr(
            Ai_ptr, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai_ptr, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai_ptr, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
        )

        A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
        Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
        Ai_21 = -tl.dot(
            tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
        )
        tl.store(
            p_Ai_11,
            Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_32x32_to_64x64_loop_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NT: tl.constexpr,
        BH: tl.constexpr,
):
    worker_id = tl.program_id(0)
    total_tasks = NT * BH
    num_tasks = total_tasks // 24
    remainder = total_tasks - num_tasks * 24
    upper_bound = min(total_tasks, num_tasks * (worker_id + 1) + min(worker_id + 1, remainder))
    lower_bound = num_tasks * worker_id + min(worker_id, remainder)
    for task_id in range(lower_bound, upper_bound):
        i_tt = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_tt * 2).to(tl.int32), tl.load(
                chunk_indices + i_tt * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_tt

        A_ptr = A + (bos * H + i_h) * BT
        Ad_ptr = Ad + (bos * H + i_h) * 32
        Ai_ptr = Ai + (bos * H + i_h) * 64

        p_A_21 = tl.make_block_ptr(
            A_ptr, (T, BT), (H * BT, 1), (i_t * 64 + 32, 0 + i_t % (BT // 64) * 64), (32, 32), (1, 0)
        )

        p_Ad_11 = tl.make_block_ptr(
            Ad_ptr, (T, 32), (H * 32, 1), (i_t * 64, 0), (32, 32), (1, 0)
        )
        p_Ad_22 = tl.make_block_ptr(
            Ad_ptr, (T, 32), (H * 32, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
        )

        p_Ai_11 = tl.make_block_ptr(
            Ai_ptr, (T, 64), (H * 64, 1), (i_t * 64, 0), (32, 32), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai_ptr, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (32, 32), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai_ptr, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
        )

        A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
        Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
        Ai_21 = -tl.dot(
            tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
        )
        tl.store(
            p_Ai_11,
            Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=['T'])
def solve_tril_64x64_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 64)
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 64

    offset = (i_t * 64) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * 64, offset), (64, 64), (1, 0))
        b_A = -tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [64, 64])
        desc_o = make_tensor_descriptor(Ai, [T, 64], [H * 64, 1], [64, 64])
        b_A = -desc.load([i_t * 64, offset]).to(tl.float32)

    for i in range(2, min(64, T - i_t * 64)):
        b_a = -tl.load(A + (i_t * 64 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (64, 64), (1, 0))
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * 64, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


def solve_tril_64(
        A: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.float,
    ):
    B, T, H, BT = A.shape
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    solve_tril_64x64_kernel[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=False,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return Ai


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    output_dtype = A.dtype if output_dtype is None else output_dtype
    if not _TRITON_SLICE_AVAILABLE:
        if A.shape[-1] not in [64]:
            raise ValueError(
                f"A shape BT should in [64], but current is {A.shape[-1]}"
            )
        return solve_tril_64(A, cu_seqlens, output_dtype)
    if A.shape[-1] not in [16, 32, 64]:
        raise ValueError(
            f"A shape BT should in [16, 32, 64], but current is {A.shape[-1]}"
        )

    B, T, H, BT = A.shape
    # If BT matches the current processing level (final step), use output_dtype
    # (e.g. BF16) so the kernel can downcast internally, avoiding an extra
    # external cast that hurts performance. Otherwise, keep FP32 to preserve
    # precision for subsequent computation stages.
    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    LARGE_BLOCK_T = 608 * 2

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T)
        if cu_seqlens is not None
        else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, LARGE_BLOCK_T)
    solve_tril_16x16_loop_kernel_paral_v3[(48,)](
        A,
        Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        NT=NT,
        BH=B * H,
    )

    if BT == 16:
        return Ad

    # Same dtype logic as above: output_dtype for the final step, FP32 otherwise.
    Ai = torch.zeros(
        B, T, H, 32, device=A.device, dtype=torch.float if BT != 32 else output_dtype
    )

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 32) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 32)
    merge_16x16_to_32x32_loop_inverse_kernel[(24,)](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        NT=NT,
        BH=B * H,
    )
    if BT == 32:
        return Ai

    Ad = Ai
    # Same dtype logic as above: output_dtype for the final step, FP32 otherwise.
    Ai = torch.zeros(
        B, T, H, 64, device=A.device, dtype=torch.float if BT != 64 else output_dtype
    )
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 64)
    merge_32x32_to_64x64_loop_inverse_kernel[(24,)](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        NT=NT,
        BH=B * H,
    )
    if BT == 64:
        return Ai
    return Ai
