# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# pylint: disable=no-name-in-module,relative-beyond-top-level

from typing import Optional

import torch
import triton
import triton.language as tl


# Compatibility shim: in newer triton-ascend, slice ops live in
# `triton.language.extra.cann.extension` instead of `triton.language`.
# Re-expose them on `tl` so kernel code works on both versions.
try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice

    if not hasattr(tl, "extract_slice"):
        tl.extract_slice = extract_slice
    if not hasattr(tl, "insert_slice"):
        tl.insert_slice = insert_slice
except ImportError:
    pass

from .utils import get_vector_num, input_guard, prepare_chunk_indices


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D

        # Tail-of-allocation guard: block end in absolute packed rows must not
        # exceed B*T, else MTE DMA touches unmapped pages.
        is_tail_chunk = (bos + i_t * BT + BT) > (B * T)

        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

        b_y = tl.zeros((BT, BD), dtype=tl.float32)

        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]

                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # We keep intra loop load because preloading will cause ub overflow under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])

                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w

                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d

                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                    tl.float32
                )

                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == 'swish' or ACTIVATION == 'silu':  # pylint: disable=consider-using-in
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            if is_tail_chunk:
                o_t_r = i_t * BT + tl.arange(0, BT)
                m_t_r = (o_t_r >= 0) & (o_t_r < T_len)
                b_residual = tl.load(
                    residual + bos * D + o_t_r[:, None] * D + o_d[None, :],
                    mask=m_t_r[:, None] & m_d[None, :],
                    other=0.0,
                )
            else:
                p_residual = tl.make_block_ptr(
                    residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
                )
                b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        if is_tail_chunk:
            o_t_y = i_t * BT + tl.arange(0, BT)
            m_t_y = (o_t_y >= 0) & (o_t_y < T_len)
            b_y_cast = tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(
                y + bos * D + o_t_y[:, None] * D + o_d[None, :],
                b_y_cast,
                mask=m_t_y[:, None] & m_d[None, :],
            )
        else:
            p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_BLKS_D: tl.int32,
    NUM_CHKS: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Total packed rows = allocation upper bound of x / dy / dx, used to
    # detect tail chunks whose block end would overshoot the packed tensor
    # and trigger MTE "DDR address out of range". varlen or not, the tensor
    # shape is always [B, T, D].
    TOTAL_ROWS = B * T

    total_tasks = NUM_CHKS * NUM_BLKS_D

    for task_id in range(pid, total_tasks, num_programs):  # pylint: disable=too-many-nested-blocks
        i_d = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        if IS_VARLEN:
            i_t = i_chk

            idx_chk = i_chk

            i_tg = idx_chk

            ptr = chunk_indices + idx_chk * 2
            i_n = tl.load(ptr).to(tl.int32)
            i_t_offset = tl.load(ptr + 1).to(tl.int32)

            i_t = i_t_offset

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)

            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_tg = i_chk

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D

        is_tail_chunk = (bos + i_t * BT + BT * W) > TOTAL_ROWS

        if HAS_WEIGHT:
            if is_tail_chunk:
                o_t_x = i_t * BT + tl.arange(0, BT)
                m_t_x = (o_t_x >= 0) & (o_t_x < T_len)
                b_x = tl.load(
                    x + bos * D + o_t_x[:, None] * D + o_d[None, :],
                    mask=m_t_x[:, None] & m_d[None, :],
                    other=0,
                )
            else:
                p_x = tl.make_block_ptr(x + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_x = tl.load(p_x, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")

        b_dx = tl.zeros((BT, BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)

        if not USE_FINAL_STATE and not USE_INITIAL_STATE:
            b_dw = tl.zeros((W, BD), dtype=tl.float32)

            if is_tail_chunk:
                o_t_full = i_t * BT + tl.arange(0, BT * W)
                m_t_full = (o_t_full >= 0) & (o_t_full < T_len)
                b_dy = tl.load(
                    dy + bos * D + o_t_full[:, None] * D + o_d[None, :],
                    mask=m_t_full[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)

                if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                    b_y = tl.load(
                        y + bos * D + o_t_full[:, None] * D + o_d[None, :],
                        mask=m_t_full[:, None] & m_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
            else:
                p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT * W, BD), (1, 0))
                b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

                if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                    p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT * W, BD), (1, 0))
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)

            for i_w in tl.static_range(0, W):
                b_dy_sub = tl.extract_slice(b_dy, [i_w, 0], [BT, BD], [1, 1])

                if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                    b_y_sub = tl.extract_slice(b_y, [i_w, 0], [BT, BD], [1, 1])  # pylint: disable=used-before-assignment
                    b_ys = tl.sigmoid(b_y_sub)
                    b_dy_sub = b_dy_sub * b_ys * (1 + b_y_sub * (1 - b_ys))

                b_wdy = b_dy_sub
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw_sub = tl.sum(b_dy_sub * b_x, 0)  # [BT, BD] * [BT, BD] --> sum(0) = [BD]
                    b_dw = tl.insert_slice(b_dw, b_dw_sub[None, :], [W - i_w - 1, 0], [1, BD], [1, 1])

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_sub, 0)
                b_dx += b_wdy

            p_dw = tl.make_block_ptr(dw + i_tg * W * D, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            tl.store(p_dw, b_dw.to(dw.dtype.element_ty))
        elif i_t * BT >= W:
            for i_w in tl.static_range(0, W):
                if is_tail_chunk:
                    o_t_iw = i_t * BT + i_w + tl.arange(0, BT)
                    m_t_iw = (o_t_iw >= 0) & (o_t_iw < T_len)
                    b_dy = tl.load(
                        dy + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                        mask=m_t_iw[:, None] & m_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                        b_y = tl.load(
                            y + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                            mask=m_t_iw[:, None] & m_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                else:
                    p_dy = tl.make_block_ptr(
                        dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                        p_y = tl.make_block_ptr(
                            y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                        )
                        b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(dw + i_tg * W * D + (W - i_w - 1) * D + o_d, b_dw.to(dw.dtype.element_ty), mask=m_d)
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(0, W):
                if is_tail_chunk:
                    o_t_iw = i_t * BT + i_w + tl.arange(0, BT)
                    m_t_iw = (o_t_iw >= 0) & (o_t_iw < T_len)
                    b_dy_shift = tl.load(
                        dy + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                        mask=m_t_iw[:, None] & m_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                        b_y = tl.load(
                            y + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                            mask=m_t_iw[:, None] & m_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                else:
                    p_dy = tl.make_block_ptr(
                        dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                        p_y = tl.make_block_ptr(
                            y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                        )
                        b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                if HAS_WEIGHT:
                    b_dw = tl.sum(b_dy_shift * b_x, 0)

                    if USE_INITIAL_STATE:
                        mask_head_rows = o_t < i_w

                        b_dy_head = tl.load(
                            dy + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)
                        if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                            b_y_head = tl.load(
                                y + bos * D + o_t[:, None] * D + o_d,
                                mask=(mask_head_rows[:, None] & m_d[None, :]),
                                other=0.0,
                            ).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t

                        mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                        b_xc = tl.load(
                            initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                            mask=(mask_c[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)

                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(dw + i_tg * W * D + (W - i_w - 1) * D + o_d, b_dw.to(dw.dtype.element_ty), mask=m_d)

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                b_wdy = (
                    b_dy_shift
                    if not HAS_WEIGHT
                    else (b_dy_shift * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1]))
                )
                b_dx += b_wdy

            if USE_INITIAL_STATE:
                for i_w in tl.static_range(1, W):
                    # dh0[i_w] = sum_{t=0}^{i_w-1} dy0[t, :] * w[i_w-1-t, :]
                    # 逐行 load dy0 避免预加载 [BT,BD] 炸 UB，消除三维 i1 broadcast
                    b_dh0_s = tl.zeros((BD,), dtype=tl.float32)
                    for i_t2 in tl.static_range(0, W - 1):
                        if i_t2 < i_w:
                            dy0_row = tl.load(dy + bos * D + (i_t * BT + i_t2) * D + o_d, mask=m_d, other=0.0).to(
                                tl.float32
                            )
                            if ACTIVATION == "swish" or ACTIVATION == "silu":  # pylint: disable=consider-using-in
                                y0_row = tl.load(y + bos * D + (i_t * BT + i_t2) * D + o_d, mask=m_d, other=0.0).to(
                                    tl.float32
                                )
                                y0_s = tl.sigmoid(y0_row)
                                dy0_row = dy0_row * y0_s * (1 + y0_row * (1 - y0_s))
                            if HAS_WEIGHT:
                                w_row = tl.extract_slice(b_w, [i_w - 1 - i_t2, 0], [1, BD], [1, 1])
                                b_dh0_s += tl.sum(dy0_row[None, :] * w_row, 0).to(tl.float32)
                            else:
                                b_dh0_s += dy0_row

                    tl.store(
                        dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                        b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
                        mask=m_d,
                    )

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if i_t * BT + BT >= T_len - W:
                # final_state[b,d,w] = x[b, T_len-W+w, d]，w ∈ [0, W)
                # 所以 dx[t] += dht[b, d, t-(T_len-W)]，当 t ∈ [T_len-W, T_len-1]
                row_arange = tl.arange(0, BT)
                for i_w in tl.static_range(0, W):
                    target_row = T_len - W + i_w
                    local_row = target_row - i_t * BT
                    in_chunk = (local_row >= 0) & (local_row < BT) & (target_row >= 0) & (target_row < T_len)
                    b_dht_row = tl.load(
                        dht + i_n * D * W + o_d * W + i_w,
                        mask=m_d,
                        other=0.0,
                    ).to(tl.float32)
                    row_match = (row_arange == local_row) & in_chunk
                    b_dx += tl.where(
                        row_match[:, None] & m_d[None, :],
                        b_dht_row[None, :],
                        0.0,
                    )

        if is_tail_chunk:
            o_t_dx = i_t * BT + tl.arange(0, BT)
            m_t_dx = (o_t_dx >= 0) & (o_t_dx < T_len)
            b_dx_cast = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(
                dx + bos * D + o_t_dx[:, None] * D + o_d[None, :],
                b_dx_cast,
                mask=m_t_dx[:, None] & m_d[None, :],
            )
        else:
            p_dx = tl.make_block_ptr(dx + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            tl.store(
                p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1)
            )


@input_guard
def causal_conv1d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("x [B, T, D], weight [W, D], please check.")
    B, T, D, W = *x.shape, weight.shape[0]
    NUM_CORES = get_vector_num()
    # USE_INITIAL_STATE: the else-branch (first chunk) uses tl.static_range which
    # unrolls W iterations, each keeping both x-load and initial_state-load live.
    # Combined with NPU multi-buffering this easily overflows the ~192 KB UB.
    # Reduce BD and cap BT to keep peak UB within budget.
    if initial_state is not None:
        BD = 32
        BT = min(16, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    else:
        BD = 256
        BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    if D % BD != 0:
        raise ValueError("D must be divisible by BD.")
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None

        NUM_CHKS = triton.cdiv(T, BT) * B

    y = torch.empty_like(x)

    grid = (NUM_CORES,)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
    )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    return y.view(shape), final_state


@input_guard
def causal_conv1d_bwd_impl(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    shape = x.shape
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("x [B, T, D], weight [W, D], please check.")

    B, T, D = x.shape
    W = weight.shape[0] if weight is not None else None

    NUM_CORES = get_vector_num()
    # ---- UB-aware tile sizing for backward ----
    # UB capacity: 192 KB = 196608 bytes. With multi-buffering (up to 3x),
    # effective budget ≈ 64 KB per "live set".
    #
    # Path C (USE_INITIAL_STATE, worst case with activation) peak live buffers:
    #   b_x[BT,BD], b_w[W,BD]                          — input dtype (es bytes each)
    #   b_dx[BT,BD], b_dy_shift[BT,BD], b_y[BT,BD],
    #   b_dy_head[BT,BD], b_xc[BT,BD]                  — fp32 (4 bytes each)
    #   b_dw[BD], b_db[BD]                              — fp32 (small)
    #
    # Peak ≈ BT*BD*(2*es + 5*4) + W*BD*es + 2*BD*4
    #      ≈ BT*BD*(es*2 + 20) + W*BD*es   (ignoring small terms)
    #
    # Budget: BT*BD*(es*2 + 20) + W*BD*es ≤ 65536 bytes
    #
    # With BT=8: BD ≤ 65536 / (8*(4*2+20) + W*4) = 65536 / (8*28 + W*4)
    #   W=4: BD ≤ 65536 / 240 = 273 → clamp to 256 (but too aggressive)
    #   Conservative: fix BT=8, BD=32 gives 8*32*28 + 4*32*4 = 7168+512 = 7.5KB ✓✓✓
    #
    # We use BT=8, BD=32 as safe defaults that work for all W≤8 and all dtypes.
    # This matches the front-end's conservative approach but accounts for the
    # extra gradient buffers in backward.
    if initial_state is not None:
        BD = 32
        BT = min(8, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    else:
        BD = 32
        BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    if D % BD != 0:
        raise ValueError("D must be divisible by BD.")
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NUM_CHKS = len(chunk_indices)

        NT = len(chunk_indices)
    else:
        chunk_indices = None

        NT = triton.cdiv(T, BT)
        NUM_CHKS = NT * B

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B * NT, W, D, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    if initial_state is not None:
        if cu_seqlens is not None:
            eff_NT = len(chunk_indices)
        else:
            eff_NT = triton.cdiv(T, BT)

        dh0 = initial_state.new_zeros(min(eff_NT, triton.cdiv(W, BT)), *initial_state.shape)
    else:
        dh0 = None

    grid = (NUM_CORES,)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dh0=dh0,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_BLKS_D=NUM_BLKS_D,
        NUM_CHKS=NUM_CHKS,
    )

    if weight is not None:
        dw = dw.sum(0).contiguous().to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    if initial_state is not None:
        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)

    return dx.view(shape), dw, db, dr, dh0


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = eos - BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - BW + tl.arange(0, BW)
    m_t = o_t >= tl.maximum(bos, eos - W)
    m_d = o_d < D
    m_w = (o_w >= 0) & (o_w < W)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
    if USE_INITIAL_STATE:
        if T < BW:
            o_c = W - (BW - T) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(initial_state + i_n * D * W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
            b_x += b_cache

    tl.store(final_state + i_n * D * W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


@input_guard
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, D, W = *x.shape, state_len
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 256)
    BW = W
    grid = (triton.cdiv(D, BD), N)
    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        BW=BW,
        BD=BD,
    )
    return final_state


@triton.jit()
def causal_conv1d_update_kernel_bdt_fwd(
    x_ptr,  # [B, D, T]
    conv_state_ptr,  # [B, D, ST]
    conv_state_update_ptr,
    weight_ptr,  # [D, W]
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,  # [B, D, out_len]
    batch: tl.constexpr,
    dim: tl.constexpr,
    state_len: tl.constexpr,  # ST
    seq_len: tl.constexpr,  # T
    width: tl.constexpr,  # W
    out_len: tl.constexpr,  # output time
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    T_CHK_SIZE: tl.constexpr,
    D_CHK_SIZE: tl.constexpr,
    NUM_T_CHK: tl.constexpr,
    NUM_D_CHK: tl.constexpr,
    ST_STORE_HEAD_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    total_task = batch * NUM_D_CHK * NUM_T_CHK

    for task_id in tl.range(pid, total_task, pnum):
        di = task_id % NUM_D_CHK
        bti = task_id // NUM_D_CHK
        bi = bti // NUM_T_CHK
        ti = bti % NUM_T_CHK

        w = tl.load(
            tl.make_block_ptr(
                weight_ptr,
                shape=(dim, width),
                strides=(width, 1),
                offsets=(di * D_CHK_SIZE, 0),
                block_shape=(D_CHK_SIZE, width),
                order=(1, 0),
            ),
            boundary_check=(0, 1),
            padding_option="zero",
        )

        if ti == 0:
            st_b = tl.load(
                tl.make_block_ptr(
                    conv_state_ptr + bi * state_len * dim,
                    shape=(dim, state_len),
                    strides=(state_len, 1),
                    offsets=(di * D_CHK_SIZE, state_len - (width - 1)),
                    block_shape=(D_CHK_SIZE, (width - 1) + T_CHK_SIZE),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )
            offset0_x = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1_x = ti * T_CHK_SIZE + tl.arange(0, T_CHK_SIZE)
            mask_x = (offset0_x < dim)[:, None] & ((offset1_x >= 0) & (offset1_x < seq_len))[None, :]
            block_off_x = bi * dim * seq_len + offset0_x[:, None] * seq_len + offset1_x[None, :]
            x_b_tmp = tl.load(x_ptr + block_off_x, mask=mask_x, other=0)
            x_b = tl.insert_slice(st_b, x_b_tmp, (0, width - 1), (D_CHK_SIZE, T_CHK_SIZE), (1, 1))
        else:
            offset0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1 = ti * T_CHK_SIZE - (width - 1) + tl.arange(0, T_CHK_SIZE + width - 1)
            mask = (offset0 < dim)[:, None] & ((offset1 >= 0) & (offset1 < seq_len))[None, :]
            block_off = bi * dim * seq_len + offset0[:, None] * seq_len + offset1[None, :]
            x_b = tl.load(x_ptr + block_off, mask=mask, other=0)

        out_block = tl.zeros((T_CHK_SIZE, D_CHK_SIZE), dtype=x_ptr.dtype.element_ty)
        x_b = tl.trans(x_b, (1, 0))
        w = tl.trans(w, (1, 0))

        new_state_start_off = seq_len - state_len
        t_start_off = ti * T_CHK_SIZE - (width - 1)
        t_end_off = (ti + 1) * T_CHK_SIZE
        if t_end_off >= new_state_start_off:
            t_off = t_start_off - new_state_start_off
            if t_off < -(width - 1):
                # NOTE: In order to avoid use tl.maximum for negative offset,
                #       we pre-compute a fix head tile size (ST_STORE_HEAD_TILE_SIZE)
                #       to store the scene of negative address
                x_new_h = tl.extract_slice(x_b, (-t_off, 0), (ST_STORE_HEAD_TILE_SIZE, D_CHK_SIZE), (1, 1))
                x_new_h = tl.trans(x_new_h, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1_h = tl.arange(0, ST_STORE_HEAD_TILE_SIZE)[None, :]
                nst_mask_h = (nst_off_y0 < dim) & (nst_off_y1_h >= 0) & (nst_off_y1_h < state_len)
                block_ptr_h = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1_h
                tl.store(conv_state_update_ptr + block_ptr_h, x_new_h, mask=nst_mask_h)
            else:
                x_new_s = tl.extract_slice(x_b, (width - 1, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
                x_new_s = tl.trans(x_new_s, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1 = width - 1 + t_off + tl.arange(0, T_CHK_SIZE)[None, :]
                nst_mask = (nst_off_y0 < dim) & (nst_off_y1 >= 0) & (nst_off_y1 < state_len)
                block_ptr = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1
                tl.store(conv_state_update_ptr + block_ptr, x_new_s, mask=nst_mask)

        for owi in tl.range(0, width):
            new_x = tl.extract_slice(x_b, (owi, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
            w_chl_wi = tl.extract_slice(w, (owi, 0), (1, D_CHK_SIZE), (1, 1))
            x_mul_chl_wi = new_x * w_chl_wi
            out_block += x_mul_chl_wi
        out_block = tl.trans(out_block, (1, 0))

        if SILU_ACTIVATION:
            out_block = out_block * tl.sigmoid(out_block)
        tl.store(
            tl.make_block_ptr(
                out_ptr,
                shape=(batch, dim, out_len),
                strides=(dim * out_len, out_len, 1),
                offsets=(bi, di * D_CHK_SIZE, ti * T_CHK_SIZE),
                block_shape=(1, D_CHK_SIZE, T_CHK_SIZE),
                order=(2, 1, 0),
            ),
            out_block[None, :, :],
            boundary_check=(0, 1, 2),
        )


@input_guard
def causal_conv1d_update_bdt_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[str] = None,
):
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        if activation not in ["silu", "swish"]:
            raise ValueError("activation must be one of 'silu' or 'swish'.")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    out = torch.empty_like(x)

    NUM_CORES = get_vector_num()
    T_CHK_SIZE = 256
    D_CHK_SIZE = 16

    if T_CHK_SIZE < width:
        raise ValueError("T_CHK_SIZE must be >= width.")

    NUM_T_CHK = triton.cdiv(out.shape[-1], T_CHK_SIZE)
    NUM_D_CHK = triton.cdiv(dim, D_CHK_SIZE)
    conv_state_update = torch.empty_like(conv_state)

    # A const tile size variable to update negative address of conv state
    ST_STORE_HEAD_TILE_SIZE = width if (seqlen % T_CHK_SIZE) > width else (width - seqlen % T_CHK_SIZE) % T_CHK_SIZE
    causal_conv1d_update_kernel_bdt_fwd[(NUM_CORES, 1)](
        x,
        conv_state,
        conv_state_update,
        weight,
        bias,
        conv_state_indices,
        out,
        batch=int(batch),
        dim=int(dim),
        state_len=int(conv_state.shape[-1]),
        seq_len=int(x.shape[-1]),
        width=int(width),
        out_len=int(out.shape[-1]),
        x_batch_stride=x.stride()[0],
        conv_batch_stride=conv_state.stride()[0],
        out_batch_stride=out.stride()[0],
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        T_CHK_SIZE=T_CHK_SIZE,
        D_CHK_SIZE=D_CHK_SIZE,
        NUM_T_CHK=NUM_T_CHK,
        NUM_D_CHK=NUM_D_CHK,
        ST_STORE_HEAD_TILE_SIZE=int(ST_STORE_HEAD_TILE_SIZE),
    )
    conv_state.copy_(conv_state_update)
    if unsqueeze:
        out = out.squeeze(-1)
    return out
