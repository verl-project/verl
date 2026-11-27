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
import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S


@functools.lru_cache(maxsize=16)
def _device_core_counts(device_idx: int) -> tuple[int, int]:
    props = torch.npu.get_device_properties(device_idx)
    cube_cores = props.cube_core_num
    vector_cores = getattr(props, "vector_core_num", 2 * cube_cores)
    return cube_cores, vector_cores


def _num_cube_cores() -> int:
    return _device_core_counts(torch.npu.current_device())[0]


def _num_vector_cores() -> int:
    return _device_core_counts(torch.npu.current_device())[1]


def _normalize_dtype(dtype):
    if dtype in (torch.bfloat16, "bfloat16"):
        return T.bfloat16
    if dtype in (torch.float16, "float16"):
        return T.float16
    if dtype in (torch.float32, "float32"):
        return T.float32
    return dtype


def _select_block_k(K: int) -> int:
    if K >= 256 and K % 256 == 0:
        return 256
    if K >= 128 and K % 128 == 0:
        return 128
    return 64


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_target_logits(
    N: int,
    K: int,
    topk: int,
    dtype=T.bfloat16,
    num_threads=128,
):
    """Compute selected target logits on Ascend vector cores.

    Partitions by token row so `hidden[row, :]` is transferred to UB once per row
    instead of `topk` times, and uses pipelined MTE2 DMA + SimdVF vector dot products
    for in-shard target labels while skipping weight loads for out-of-shard labels.
    """
    dtype = _normalize_dtype(dtype)
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    is_fp32 = dtype == T.float32
    use_simd = (K >= 128) and (K % 128 == 0) and (K <= 16384)
    VL = 64
    topk_tile = min(topk, 256)
    topk_pad = ((topk_tile + 15) // 16) * 16

    if use_simd:

        @T.prim_func
        def kernel(
            hidden: T.Tensor([M, K], dtype),
            weight: T.Tensor([N, K], dtype),
            labels: T.Tensor([M, topk], T.int64),
            inv_temperature: T.float32,
            vocab_start: T.int32,
            target_logits: T.Tensor([M, topk], T.float32),
        ):
            with T.Kernel(num_cores) as core_id:
                labels_i32 = T.view(labels, (M, topk * 2), dtype=T.int32)
                h_ub = T.alloc_shared((K,), dtype)
                h_fp32 = T.alloc_shared((K,), T.float32)
                w_ub = T.alloc_shared((K,), dtype)
                lbl_ub = T.alloc_shared((topk_pad * 2,), T.int32)
                out_ub = T.alloc_shared((topk_pad,), T.float32)

                for row in T.Persistent([M], num_cores, core_id):
                    T.copy(hidden[row, 0:K], h_ub)
                    with T.SimdVF():
                        mask = S.pset(32)
                        for c in range(0, K, VL):
                            h_vec = S.vld(h_ub[c]) if is_fp32 else S.vcvt(S.vld(h_ub[c], "UNPK_B16"), T.float32, mask)
                            S.vsts(h_fp32[c], h_vec, mask)

                    for k_base in range(0, topk, topk_tile):
                        cur_topk = T.min(topk_tile, topk - k_base)
                        T.copy(labels_i32[row, k_base * 2 : (k_base + cur_topk) * 2], lbl_ub[0 : cur_topk * 2])
                        for j in range(cur_topk):
                            local_label = lbl_ub[2 * j] - vocab_start
                            if (local_label >= 0) and (local_label < N):
                                T.copy(weight[local_label, 0:K], w_ub)
                                with T.SimdVF():
                                    mask = S.pset(32)
                                    one_mask = S.pset(32, "PAT_VL1")
                                    acc0 = S.alloc_var(T.float32)
                                    acc1 = S.alloc_var(T.float32)
                                    acc0 = S.vdup(0.0, T.float32, mask)
                                    acc1 = S.vdup(0.0, T.float32, mask)
                                    for c in range(0, K, 128):
                                        w0 = (
                                            S.vld(w_ub[c])
                                            if is_fp32
                                            else S.vcvt(S.vld(w_ub[c], "UNPK_B16"), T.float32, mask)
                                        )
                                        w1 = (
                                            S.vld(w_ub[c + 64])
                                            if is_fp32
                                            else S.vcvt(S.vld(w_ub[c + 64], "UNPK_B16"), T.float32, mask)
                                        )
                                        S.vmula(acc0, S.vld(h_fp32[c]), w0, mask)
                                        S.vmula(acc1, S.vld(h_fp32[c + 64]), w1, mask)
                                    dot = S.vcadd(S.vadd(acc0, acc1, mask), mask)
                                    S.vsts(out_ub[j], S.vmuls(dot, inv_temperature, one_mask), one_mask, "ONEPT_B32")
                            else:
                                with T.SimdVF():
                                    one_mask = S.pset(32, "PAT_VL1")
                                    S.vsts(out_ub[j], S.vdup(0.0, T.float32, one_mask), one_mask, "ONEPT_B32")
                        T.copy(out_ub[0:cur_topk], target_logits[row, k_base : k_base + cur_topk])

        return kernel

    @T.prim_func
    def fallback_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        target_logits: T.Tensor([M, topk], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            for row in T.Persistent([M], num_cores, core_id):
                for label_idx in range(topk):
                    local_label = labels[row, label_idx] - vocab_start
                    if (local_label >= 0) and (local_label < N):
                        with T.SimtVF(threads=num_threads):
                            dot = T.alloc_reducer((1,), T.float32, op="sum")
                            T.reducer_init(dot)
                            for k in T.Parallel(K):
                                T.reducer_update(
                                    dot[0],
                                    T.cast(hidden[row, k], T.float32) * T.cast(weight[local_label, k], T.float32),
                                )
                            result = T.alloc_fragment((1,), T.float32)
                            T.finalize_reducer(dot, result)
                            if T.get_thread_binding() == 0:
                                target_logits[row, label_idx] = result[0] * inv_temperature
                    else:
                        with T.SimtVF(threads=num_threads):
                            if T.get_thread_binding() == 0:
                                target_logits[row, label_idx] = 0.0

    return fallback_kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_mainloop(
    N: int,
    K: int,
    dtype=T.bfloat16,
    BLOCK_M=64,
    BLOCK_N=256,
    BLOCK_K=64,
    VOCAB_PER_CTA=512,
    num_threads=128,
    num_stages=2,
    min_blocks_per_sm=1,
):
    """Compute rank-local online softmax and weighted-logit statistics on Ascend mixed cores.

    Uses 256x256xBLOCK_K Cube tiles with hardware unit-flag L0C->UB handoff, L2-grouped
    split persistence, and barrier-free `SimdVF` vector reductions (`vcmax`, `vexpdif`, `vcadd`).
    """
    del BLOCK_M, BLOCK_K, num_threads, min_blocks_per_sm
    dtype = _normalize_dtype(dtype)
    assert VOCAB_PER_CTA % BLOCK_N == 0
    assert BLOCK_N % 64 == 0

    TILE_M = 256
    TILE_N = BLOCK_N
    TILE_K = _select_block_k(K)
    half_m = TILE_M // 2
    VL = 64
    M = T.dynamic("M")
    num_splits = T.ceildiv(N, VOCAB_PER_CTA)
    num_vocab_tiles = VOCAB_PER_CTA // TILE_N
    k_tiles = T.ceildiv(K, TILE_K)
    num_cores = _num_cube_cores()
    need_col_mask = (N % VOCAB_PER_CTA) != 0
    # Keep ~2-4 splits (~2048-4096 vocab rows) hot in the 128 MB L2 cache while streaming hidden.
    split_group_size = max(1, min(4, 4096 // max(1, VOCAB_PER_CTA)))

    @T.prim_func
    def kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        inv_temperature: T.float32,
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
    ):
        with T.MixedKernel(num_cores, sids=2) as (core_id, sid):
            hidden_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            weight_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            logits_l0c = T.alloc_l0c((TILE_M, TILE_N), T.float32)
            logits_ub = T.alloc_shared((half_m, TILE_N), T.float32)
            local_max_ub = T.alloc_shared((half_m,), T.float32)
            local_sum_ub = T.alloc_shared((half_m,), T.float32)
            local_wsum_ub = T.alloc_shared((half_m,), T.float32)
            old_max_ub = T.alloc_shared((half_m,), T.float32)
            tile_sum_ub = T.alloc_shared((half_m,), T.float32)
            tile_wsum_ub = T.alloc_shared((half_m,), T.float32)
            T.annotate_buffer_versions({hidden_l1: num_stages, weight_l1: num_stages})

            for split, m_tile in T.Persistent(
                [num_splits, T.ceildiv(M, TILE_M)],
                num_cores,
                core_id,
                group_size=split_group_size,
            ):
                m_offset = m_tile * TILE_M
                with T.SimdVF():
                    mask = S.pset(32)
                    ninf = S.vdup(-T.infinity(T.float32), T.float32, mask)
                    zero = S.vdup(0.0, T.float32, mask)
                    for r in range(0, half_m, VL):
                        S.vsts(local_max_ub[r], ninf, mask)
                        S.vsts(local_sum_ub[r], zero, mask)
                        S.vsts(local_wsum_ub[r], zero, mask)

                for vocab_tile in range(num_vocab_tiles):
                    vocab_offset = split * VOCAB_PER_CTA + vocab_tile * TILE_N
                    if vocab_offset < N:
                        for k_tile in T.Pipelined(k_tiles, num_stages=num_stages):
                            T.copy(
                                hidden[m_offset : m_offset + TILE_M, k_tile * TILE_K : (k_tile + 1) * TILE_K],
                                hidden_l1,
                            )
                            T.copy(
                                weight[vocab_offset : vocab_offset + TILE_N, k_tile * TILE_K : (k_tile + 1) * TILE_K],
                                weight_l1,
                                l2_cache_ctrl="NOTALLOC_KEEP",
                            )
                            T.gemm(
                                hidden_l1,
                                weight_l1,
                                logits_l0c,
                                transpose_B=True,
                                clear_accum=(k_tile == 0),
                                unit_flag_ctrl=T.Select(k_tile == k_tiles - 1, 3, 2),
                            )
                        T.dual_copy(logits_l0c, logits_ub, unit_flag_ctrl=3)

                        with T.SimdVF():
                            mask = S.pset(32)
                            one_mask = S.pset(32, "PAT_VL1")
                            lanes = S.vci(0, T.int32)
                            ninf = S.vdup(-T.infinity(T.float32), T.float32, mask)
                            zero = S.vdup(0.0, T.float32, mask)
                            for r in range(0, half_m, VL):
                                S.vsts(old_max_ub[r], S.vld(local_max_ub[r]), mask)

                            for r in range(half_m):
                                rmax_vec = S.alloc_var(T.float32)
                                rmax_vec = ninf
                                for c in range(0, TILE_N, VL):
                                    s_raw = S.vmuls(S.vld(logits_ub[r, c]), inv_temperature, mask)
                                    valid_c = S.vcmps(S.vadds(lanes, vocab_offset + c), N, mask, op="lt")
                                    s_val = S.vsel(s_raw, ninf, valid_c) if need_col_mask else s_raw
                                    S.vsts(logits_ub[r, c], s_val, mask)
                                    rmax_vec = S.vmax(rmax_vec, s_val, mask)
                                S.vsts(local_max_ub[r], S.vcmax(rmax_vec, mask), one_mask, "ONEPT_B32")
                            S.mem_bar("VST_VLD")

                            for r in range(0, half_m, VL):
                                S.vsts(
                                    local_max_ub[r],
                                    S.vmax(S.vld(old_max_ub[r]), S.vld(local_max_ub[r]), mask),
                                    mask,
                                )
                            S.mem_bar("VST_VLD")

                            for r in range(half_m):
                                m_brc = S.vld(local_max_ub[r], "BRC_B32")
                                sum_vec = S.alloc_var(T.float32)
                                wsum_vec = S.alloc_var(T.float32)
                                sum_vec = zero
                                wsum_vec = zero
                                for c in range(0, TILE_N, VL):
                                    l_val = S.vld(logits_ub[r, c])
                                    e_val = S.vexpdif(l_val, m_brc, mask)
                                    valid_c = S.vcmps(S.vadds(lanes, vocab_offset + c), N, mask, op="lt")
                                    c_val = S.vsel(l_val, zero, valid_c) if need_col_mask else l_val
                                    sum_vec = S.vadd(sum_vec, e_val, mask)
                                    S.vmula(wsum_vec, c_val, e_val, mask)
                                S.vsts(tile_sum_ub[r], S.vcadd(sum_vec, mask), one_mask, "ONEPT_B32")
                                S.vsts(tile_wsum_ub[r], S.vcadd(wsum_vec, mask), one_mask, "ONEPT_B32")
                            S.mem_bar("VST_VLD")

                            for r in range(0, half_m, VL):
                                rescale = S.vexpdif(S.vld(old_max_ub[r]), S.vld(local_max_ub[r]), mask)
                                S.vsts(
                                    local_sum_ub[r],
                                    S.vadd(S.vmul(S.vld(local_sum_ub[r]), rescale, mask), S.vld(tile_sum_ub[r]), mask),
                                    mask,
                                )
                                S.vsts(
                                    local_wsum_ub[r],
                                    S.vadd(
                                        S.vmul(S.vld(local_wsum_ub[r]), rescale, mask),
                                        S.vld(tile_wsum_ub[r]),
                                        mask,
                                    ),
                                    mask,
                                )

                with T.SimtVF(threads=128):
                    for i in T.Parallel(half_m):
                        row = m_offset + sid * half_m + i
                        if row < M:
                            partial_max[row, split] = local_max_ub[i]
                            partial_sum[row, split] = local_sum_ub[i]
                            partial_weighted_sum[row, split] = local_wsum_ub[i]

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_local_reduce(
    num_splits: int,
    BLOCK_M=8,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge rank-local vocabulary-split statistics without cross-thread barriers."""
    del BLOCK_M, BLOCK_S, num_threads
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = 64
    use_fast_dma = num_splits <= 192
    ELEMS = ROWS * max(1, min(num_splits, 192))

    @T.prim_func
    def kernel(
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
        local_max: T.Tensor([M], T.float32),
        local_sum: T.Tensor([M], T.float32),
        local_weighted_sum: T.Tensor([M], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            pm_1d = T.view(partial_max, (M * num_splits,), dtype=T.float32)
            ps_1d = T.view(partial_sum, (M * num_splits,), dtype=T.float32)
            pws_1d = T.view(partial_weighted_sum, (M * num_splits,), dtype=T.float32)
            pm_ub = T.alloc_shared((ELEMS,), T.float32)
            ps_ub = T.alloc_shared((ELEMS,), T.float32)
            pws_ub = T.alloc_shared((ELEMS,), T.float32)
            out_max_ub = T.alloc_shared((ROWS,), T.float32)
            out_sum_ub = T.alloc_shared((ROWS,), T.float32)
            out_wsum_ub = T.alloc_shared((ROWS,), T.float32)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    eb = row_base * num_splits
                    T.copy(pm_1d[eb : eb + ROWS * num_splits], pm_ub)
                    T.copy(ps_1d[eb : eb + ROWS * num_splits], ps_ub)
                    T.copy(pws_1d[eb : eb + ROWS * num_splits], pws_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            base = i * num_splits
                            rmax = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                            for s in range(num_splits):
                                v = pm_ub[base + s]
                                rmax = T.max(rmax, v)
                            rsum = T.alloc_var(T.float32, init=0.0)
                            rwsum = T.alloc_var(T.float32, init=0.0)
                            for s in range(num_splits):
                                coeff = T.exp(pm_ub[base + s] - rmax)
                                rsum = rsum + coeff * ps_ub[base + s]
                                rwsum = rwsum + coeff * pws_ub[base + s]
                            out_max_ub[i] = rmax
                            out_sum_ub[i] = rsum
                            out_wsum_ub[i] = rwsum
                    T.copy(out_max_ub, local_max[row_base : row_base + ROWS])
                    T.copy(out_sum_ub, local_sum[row_base : row_base + ROWS])
                    T.copy(out_wsum_ub, local_weighted_sum[row_base : row_base + ROWS])
                else:
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            if row < M:
                                rmax = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                                for s in range(num_splits):
                                    rmax = T.max(rmax, partial_max[row, s])
                                rsum = T.alloc_var(T.float32, init=0.0)
                                rwsum = T.alloc_var(T.float32, init=0.0)
                                for s in range(num_splits):
                                    coeff = T.exp(partial_max[row, s] - rmax)
                                    rsum = rsum + coeff * partial_sum[row, s]
                                    rwsum = rwsum + coeff * partial_weighted_sum[row, s]
                                local_max[row] = rmax
                                local_sum[row] = rsum
                                local_weighted_sum[row] = rwsum

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_finalize_local(
    num_splits: int,
    topk: int,
    BLOCK_M=8,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge local statistics and produce single-device outputs."""
    del BLOCK_M, BLOCK_S, num_threads
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = 64
    use_fast_dma = (num_splits <= 192) and (topk <= 192)
    SPLIT_ELEMS = ROWS * max(1, min(num_splits, 192))
    TOPK_ELEMS = ROWS * max(1, min(topk, 192))

    @T.prim_func
    def kernel(
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
        logprobs: T.Tensor([M, topk], T.float32),
        entropy: T.Tensor([M], T.float32),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            pm_1d = T.view(partial_max, (M * num_splits,), dtype=T.float32)
            ps_1d = T.view(partial_sum, (M * num_splits,), dtype=T.float32)
            pws_1d = T.view(partial_weighted_sum, (M * num_splits,), dtype=T.float32)
            tlog_1d = T.view(target_logits, (M * topk,), dtype=T.float32)
            lp_1d = T.view(logprobs, (M * topk,), dtype=T.float32)

            pm_ub = T.alloc_shared((SPLIT_ELEMS,), T.float32)
            ps_ub = T.alloc_shared((SPLIT_ELEMS,), T.float32)
            pws_ub = T.alloc_shared((SPLIT_ELEMS,), T.float32)
            tlog_ub = T.alloc_shared((TOPK_ELEMS,), T.float32)
            lp_ub = T.alloc_shared((TOPK_ELEMS,), T.float32)
            ent_ub = T.alloc_shared((ROWS,), T.float32)
            lse_ub = T.alloc_shared((ROWS,), T.float32)
            mean_ub = T.alloc_shared((ROWS,), T.float32)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    sb = row_base * num_splits
                    kb = row_base * topk
                    T.copy(pm_1d[sb : sb + ROWS * num_splits], pm_ub)
                    T.copy(ps_1d[sb : sb + ROWS * num_splits], ps_ub)
                    T.copy(pws_1d[sb : sb + ROWS * num_splits], pws_ub)
                    T.copy(tlog_1d[kb : kb + ROWS * topk], tlog_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            s_off = i * num_splits
                            rmax = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                            for s in range(num_splits):
                                rmax = T.max(rmax, pm_ub[s_off + s])
                            rsum = T.alloc_var(T.float32, init=0.0)
                            rwsum = T.alloc_var(T.float32, init=0.0)
                            for s in range(num_splits):
                                coeff = T.exp(pm_ub[s_off + s] - rmax)
                                rsum = rsum + coeff * ps_ub[s_off + s]
                                rwsum = rwsum + coeff * pws_ub[s_off + s]
                            row_lse = rmax + T.log(rsum)
                            row_mean = rwsum / rsum
                            lse_ub[i] = row_lse
                            mean_ub[i] = row_mean
                            ent_ub[i] = row_lse - row_mean
                            k_off = i * topk
                            for k in range(topk):
                                lp_ub[k_off + k] = tlog_ub[k_off + k] - row_lse
                    T.copy(lp_ub, lp_1d[kb : kb + ROWS * topk])
                    T.copy(ent_ub, entropy[row_base : row_base + ROWS])
                    T.copy(lse_ub, lse[row_base : row_base + ROWS])
                    T.copy(mean_ub, mean_logits[row_base : row_base + ROWS])
                else:
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            if row < M:
                                rmax = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                                for s in range(num_splits):
                                    rmax = T.max(rmax, partial_max[row, s])
                                rsum = T.alloc_var(T.float32, init=0.0)
                                rwsum = T.alloc_var(T.float32, init=0.0)
                                for s in range(num_splits):
                                    coeff = T.exp(partial_max[row, s] - rmax)
                                    rsum = rsum + coeff * partial_sum[row, s]
                                    rwsum = rwsum + coeff * partial_weighted_sum[row, s]
                                row_lse = rmax + T.log(rsum)
                                row_mean = rwsum / rsum
                                lse[row] = row_lse
                                mean_logits[row] = row_mean
                                entropy[row] = row_lse - row_mean
                                for k in range(topk):
                                    logprobs[row, k] = target_logits[row, k] - row_lse

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_prepare_reduce(
    topk: int,
    BLOCK_M=64,
    num_threads=128,
):
    """Rescale local statistics to the TP-global maximum and pack them."""
    del BLOCK_M, num_threads
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = 64
    out_cols = topk + 2
    use_fast_dma = topk <= 192
    IN_ELEMS = ROWS * max(1, min(topk, 192))
    OUT_ELEMS = ROWS * max(3, min(out_cols, 194))

    @T.prim_func
    def kernel(
        local_max: T.Tensor([M], T.float32),
        global_max: T.Tensor([M], T.float32),
        local_sum: T.Tensor([M], T.float32),
        local_weighted_sum: T.Tensor([M], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
        reduced: T.Tensor([M, out_cols], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            tlog_1d = T.view(target_logits, (M * topk,), dtype=T.float32)
            red_1d = T.view(reduced, (M * out_cols,), dtype=T.float32)
            lmax_ub = T.alloc_shared((ROWS,), T.float32)
            gmax_ub = T.alloc_shared((ROWS,), T.float32)
            lsum_ub = T.alloc_shared((ROWS,), T.float32)
            lwsum_ub = T.alloc_shared((ROWS,), T.float32)
            tlog_ub = T.alloc_shared((IN_ELEMS,), T.float32)
            red_ub = T.alloc_shared((OUT_ELEMS,), T.float32)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    T.copy(local_max[row_base : row_base + ROWS], lmax_ub)
                    T.copy(global_max[row_base : row_base + ROWS], gmax_ub)
                    T.copy(local_sum[row_base : row_base + ROWS], lsum_ub)
                    T.copy(local_weighted_sum[row_base : row_base + ROWS], lwsum_ub)
                    T.copy(tlog_1d[row_base * topk : (row_base + ROWS) * topk], tlog_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            scale = T.exp(lmax_ub[i] - gmax_ub[i])
                            out_off = i * out_cols
                            in_off = i * topk
                            red_ub[out_off] = lsum_ub[i] * scale
                            red_ub[out_off + 1] = lwsum_ub[i] * scale
                            for k in range(topk):
                                red_ub[out_off + 2 + k] = tlog_ub[in_off + k]
                    T.copy(red_ub, red_1d[row_base * out_cols : (row_base + ROWS) * out_cols])
                else:
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            if row < M:
                                scale = T.exp(local_max[row] - global_max[row])
                                reduced[row, 0] = local_sum[row] * scale
                                reduced[row, 1] = local_weighted_sum[row] * scale
                                for k in range(topk):
                                    reduced[row, k + 2] = target_logits[row, k]

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_finalize(
    topk: int,
    BLOCK_M=64,
    num_threads=128,
):
    """Produce TP-global logprobs and entropy."""
    del BLOCK_M, num_threads
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = 64
    in_cols = topk + 2
    use_fast_dma = topk <= 192
    IN_ELEMS = ROWS * max(3, min(in_cols, 194))
    OUT_ELEMS = ROWS * max(1, min(topk, 192))

    @T.prim_func
    def kernel(
        global_max: T.Tensor([M], T.float32),
        reduced: T.Tensor([M, in_cols], T.float32),
        logprobs: T.Tensor([M, topk], T.float32),
        entropy: T.Tensor([M], T.float32),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            red_1d = T.view(reduced, (M * in_cols,), dtype=T.float32)
            lp_1d = T.view(logprobs, (M * topk,), dtype=T.float32)
            gmax_ub = T.alloc_shared((ROWS,), T.float32)
            red_ub = T.alloc_shared((IN_ELEMS,), T.float32)
            lp_ub = T.alloc_shared((OUT_ELEMS,), T.float32)
            ent_ub = T.alloc_shared((ROWS,), T.float32)
            lse_ub = T.alloc_shared((ROWS,), T.float32)
            mean_ub = T.alloc_shared((ROWS,), T.float32)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    T.copy(global_max[row_base : row_base + ROWS], gmax_ub)
                    T.copy(red_1d[row_base * in_cols : (row_base + ROWS) * in_cols], red_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            in_off = i * in_cols
                            out_off = i * topk
                            rsum = red_ub[in_off]
                            rwsum = red_ub[in_off + 1]
                            row_lse = gmax_ub[i] + T.log(rsum)
                            row_mean = rwsum / rsum
                            lse_ub[i] = row_lse
                            mean_ub[i] = row_mean
                            ent_ub[i] = row_lse - row_mean
                            for k in range(topk):
                                lp_ub[out_off + k] = red_ub[in_off + 2 + k] - row_lse
                    T.copy(lp_ub, lp_1d[row_base * topk : (row_base + ROWS) * topk])
                    T.copy(ent_ub, entropy[row_base : row_base + ROWS])
                    T.copy(lse_ub, lse[row_base : row_base + ROWS])
                    T.copy(mean_ub, mean_logits[row_base : row_base + ROWS])
                else:
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            if row < M:
                                rsum = reduced[row, 0]
                                rwsum = reduced[row, 1]
                                row_lse = global_max[row] + T.log(rsum)
                                row_mean = rwsum / rsum
                                lse[row] = row_lse
                                mean_logits[row] = row_mean
                                entropy[row] = row_lse - row_mean
                                for k in range(topk):
                                    logprobs[row, k] = reduced[row, k + 2] - row_lse

    return kernel


@tilelang.jit(target="ascend")
def _bwd_dlogits_dense(
    N: int,
    K: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
):
    """Recompute dense logits chunk and write softmax + entropy gradient contribution."""
    dtype = _normalize_dtype(dtype)
    is_fp32 = dtype == T.float32
    TILE_M = 256
    TILE_N = 256 if VOCAB_CHUNK % 256 == 0 else 128
    TILE_K = _select_block_k(K)
    half_m = TILE_M // 2
    VL = 64
    M = T.dynamic("M")
    num_cores = _num_cube_cores()
    k_tiles = T.ceildiv(K, TILE_K)
    need_col_mask = (N % TILE_N) != 0

    @T.prim_func
    def kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
        grad_lp_sum: T.Tensor([M], T.float32),
        grad_entropy: T.Tensor([M], T.float32),
        inv_temperature: T.float32,
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.MixedKernel(num_cores, sids=2) as (core_id, sid):
            hidden_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            weight_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            logits_l0c = T.alloc_l0c((TILE_M, TILE_N), T.float32)
            logits_ub = T.alloc_shared((half_m, TILE_N), T.float32)
            dlogits_ub = T.view(logits_ub, (half_m if is_fp32 else half_m * 2, TILE_N), dtype=dtype)
            lse_ub = T.alloc_shared((half_m,), T.float32)
            mean_ub = T.alloc_shared((half_m,), T.float32)
            glp_ub = T.alloc_shared((half_m,), T.float32)
            gent_ub = T.alloc_shared((half_m,), T.float32)
            T.annotate_buffer_versions({hidden_l1: 2, weight_l1: 2})

            for vocab_tile, m_tile in T.Persistent(
                [T.ceildiv(T.min(VOCAB_CHUNK, N - vocab_offset), TILE_N), T.ceildiv(M, TILE_M)],
                num_cores,
                core_id,
                group_size=16,
            ):
                m_offset = m_tile * TILE_M
                row_base = m_offset + sid * half_m
                local_vocab_offset = vocab_offset + vocab_tile * TILE_N

                for k_tile in T.Pipelined(k_tiles, num_stages=2):
                    T.copy(
                        hidden[m_offset : m_offset + TILE_M, k_tile * TILE_K : (k_tile + 1) * TILE_K],
                        hidden_l1,
                    )
                    T.copy(
                        weight[
                            local_vocab_offset : local_vocab_offset + TILE_N, k_tile * TILE_K : (k_tile + 1) * TILE_K
                        ],
                        weight_l1,
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.gemm(
                        hidden_l1,
                        weight_l1,
                        logits_l0c,
                        transpose_B=True,
                        clear_accum=(k_tile == 0),
                        unit_flag_ctrl=T.Select(k_tile == k_tiles - 1, 3, 2),
                    )

                if row_base + half_m <= M:
                    T.copy(lse[row_base : row_base + half_m], lse_ub)
                    if HAS_LOGPROBS_GRAD:
                        T.copy(grad_lp_sum[row_base : row_base + half_m], glp_ub)
                    if HAS_ENTROPY_GRAD:
                        T.copy(mean_logits[row_base : row_base + half_m], mean_ub)
                        T.copy(grad_entropy[row_base : row_base + half_m], gent_ub)
                else:
                    with T.SimtVF(threads=half_m):
                        for i in T.Parallel(half_m):
                            row = row_base + i
                            lse_ub[i] = lse[row] if row < M else 0.0
                            if HAS_LOGPROBS_GRAD:
                                glp_ub[i] = grad_lp_sum[row] if row < M else 0.0
                            if HAS_ENTROPY_GRAD:
                                mean_ub[i] = mean_logits[row] if row < M else 0.0
                                gent_ub[i] = grad_entropy[row] if row < M else 0.0

                T.dual_copy(logits_l0c, logits_ub, unit_flag_ctrl=3)

                with T.SimdVF():
                    mask = S.pset(32)
                    zero = S.vdup(0.0, T.float32, mask)
                    lanes = S.vci(0, T.int32)
                    for r in range(half_m):
                        lse_brc = S.vld(lse_ub[r], "BRC_B32")
                        if HAS_LOGPROBS_GRAD:
                            neg_glp = S.vmuls(S.vld(glp_ub[r], "BRC_B32"), -inv_temperature, mask)
                        if HAS_ENTROPY_GRAD:
                            mean_brc = S.vld(mean_ub[r], "BRC_B32")
                            neg_gent = S.vmuls(S.vld(gent_ub[r], "BRC_B32"), -inv_temperature, mask)
                        for c in range(0, TILE_N, VL):
                            logit = S.vmuls(S.vld(logits_ub[r, c]), inv_temperature, mask)
                            prob = S.vexpdif(logit, lse_brc, mask)
                            if HAS_LOGPROBS_GRAD and HAS_ENTROPY_GRAD:
                                term = S.vadd(neg_glp, S.vmul(neg_gent, S.vsub(logit, mean_brc, mask), mask), mask)
                                g_raw = S.vmul(prob, term, mask)
                            elif HAS_LOGPROBS_GRAD:
                                g_raw = S.vmul(prob, neg_glp, mask)
                            elif HAS_ENTROPY_GRAD:
                                g_raw = S.vmul(prob, S.vmul(neg_gent, S.vsub(logit, mean_brc, mask), mask), mask)
                            else:
                                g_raw = zero
                            valid_c = S.vcmps(S.vadds(lanes, local_vocab_offset + c), N, mask, op="lt")
                            g = S.vsel(g_raw, zero, valid_c) if need_col_mask else g_raw
                            if is_fp32:
                                S.vsts(logits_ub[r, c], g, mask)
                            else:
                                S.vsts(dlogits_ub[r, c], S.vcvt(g, dtype, mask), mask, dist="PK_B32")

                T.dual_copy(
                    dlogits_ub[0:half_m, 0:TILE_N],
                    dlogits[m_offset : m_offset + TILE_M, vocab_tile * TILE_N : (vocab_tile + 1) * TILE_N],
                    l2_cache_ctrl="NOTALLOC_PW",
                )

    return kernel


@tilelang.jit(target="ascend")
def _bwd_dlogits_scatter_top1(
    N: int,
    VOCAB_CHUNK: int,
    dtype=T.bfloat16,
):
    """Fast O(M) sparse target-label update for single-label (`topk == 1`) backward."""
    dtype = _normalize_dtype(dtype)
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = 128

    @T.prim_func
    def kernel(
        labels: T.Tensor([M, 1], T.int64),
        grad_logprobs: T.Tensor([M, 1], T.float32),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.Kernel(num_cores) as core_id:
            labels_1d = T.view(labels, (M * 2,), dtype=T.int32)
            glp_1d = T.view(grad_logprobs, (M,), dtype=T.float32)
            lbl_ub = T.alloc_shared((ROWS * 2,), T.int32)
            glp_ub = T.alloc_shared((ROWS,), T.float32)
            chunk_end = T.min(vocab_offset + VOCAB_CHUNK, N)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if row_base + ROWS <= M:
                    T.copy(labels_1d[row_base * 2 : (row_base + ROWS) * 2], lbl_ub)
                    T.copy(glp_1d[row_base : row_base + ROWS], glp_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            lbl = lbl_ub[2 * i] - vocab_start
                            if (lbl >= vocab_offset) and (lbl < chunk_end):
                                col = lbl - vocab_offset
                                row = row_base + i
                                cur = T.cast(dlogits[row, col], T.float32)
                                dlogits[row, col] = T.cast(cur + inv_temperature * glp_ub[i], dtype)
                else:
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            if row < M:
                                lbl = T.cast(labels[row, 0], T.int32) - vocab_start
                                if (lbl >= vocab_offset) and (lbl < chunk_end):
                                    col = lbl - vocab_offset
                                    cur = T.cast(dlogits[row, col], T.float32)
                                    dlogits[row, col] = T.cast(cur + inv_temperature * grad_logprobs[row, 0], dtype)

    return kernel


@tilelang.jit(target="ascend")
def _bwd_dlogits_prep_multi(N: int, topk: int):
    """One-time pass per backward: compute row sum of `grad_logprobs` and deduplicate in-shard labels."""
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = max(1, min(64, 4096 // max(1, topk)))
    # Round ROWS to a power of two for SimtVF threads and 32B DMA alignment when >= 8.
    ROWS = 64 if ROWS >= 64 else (32 if ROWS >= 32 else (16 if ROWS >= 16 else (8 if ROWS >= 8 else 1)))
    use_fast_dma = (ROWS >= 8) and (ROWS * topk <= 8192)
    ELEMS = max(1, ROWS * min(topk, 8192))

    @T.prim_func
    def kernel(
        labels: T.Tensor([M, topk], T.int64),
        grad_logprobs: T.Tensor([M, topk], T.float32),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        local_labels: T.Tensor([M, topk], T.int32),
        scaled_glp: T.Tensor([M, topk], T.float32),
        grad_lp_sum: T.Tensor([M], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            labels_1d = T.view(labels, (M * topk * 2,), dtype=T.int32)
            glp_1d = T.view(grad_logprobs, (M * topk,), dtype=T.float32)
            out_lbl_1d = T.view(local_labels, (M * topk,), dtype=T.int32)
            out_glp_1d = T.view(scaled_glp, (M * topk,), dtype=T.float32)
            lbl_ub = T.alloc_shared((ELEMS * 2,), T.int32)
            glp_ub = T.alloc_shared((ELEMS,), T.float32)
            out_lbl_ub = T.alloc_shared((ELEMS,), T.int32)
            out_glp_ub = T.alloc_shared((ELEMS,), T.float32)
            sum_ub = T.alloc_shared((ROWS,), T.float32)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    eb = row_base * topk
                    T.copy(labels_1d[eb * 2 : (eb + ROWS * topk) * 2], lbl_ub)
                    T.copy(glp_1d[eb : eb + ROWS * topk], glp_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            base = i * topk
                            rsum = T.alloc_var(T.float32, init=0.0)
                            for k in range(topk):
                                g = glp_ub[base + k]
                                rsum = rsum + g
                                lbl = lbl_ub[2 * (base + k)] - vocab_start
                                if (lbl < 0) or (lbl >= N):
                                    out_lbl_ub[base + k] = -1
                                    out_glp_ub[base + k] = 0.0
                                else:
                                    out_lbl_ub[base + k] = lbl
                                    out_glp_ub[base + k] = g * inv_temperature
                            sum_ub[i] = rsum
                        for i in T.Parallel(ROWS):
                            base = i * topk
                            for k in range(topk):
                                lbl = out_lbl_ub[base + k]
                                if lbl >= 0:
                                    gsum = T.alloc_var(T.float32, init=out_glp_ub[base + k])
                                    for nxt in range(k + 1, topk):
                                        if out_lbl_ub[base + nxt] == lbl:
                                            gsum = gsum + out_glp_ub[base + nxt]
                                            out_lbl_ub[base + nxt] = -1
                                            out_glp_ub[base + nxt] = 0.0
                                    out_glp_ub[base + k] = gsum
                    T.copy(out_lbl_ub, out_lbl_1d[eb : eb + ROWS * topk])
                    T.copy(out_glp_ub, out_glp_1d[eb : eb + ROWS * topk])
                    T.copy(sum_ub, grad_lp_sum[row_base : row_base + ROWS])
                else:
                    with T.SimtVF(threads=max(32, ROWS)):
                        for i in T.Parallel(max(32, ROWS)):
                            row = row_base + i
                            if (i < ROWS) and (row < M):
                                rsum = T.alloc_var(T.float32, init=0.0)
                                for k in range(topk):
                                    rsum = rsum + grad_logprobs[row, k]
                                grad_lp_sum[row] = rsum
                                for k in range(topk):
                                    lbl = T.cast(labels[row, k], T.int32) - vocab_start
                                    if (lbl < 0) or (lbl >= N):
                                        local_labels[row, k] = -1
                                        scaled_glp[row, k] = 0.0
                                    else:
                                        is_first = T.alloc_var(T.int32, init=1)
                                        for prev in range(k):
                                            if T.cast(labels[row, prev], T.int32) - vocab_start == lbl:
                                                is_first = 0
                                        if is_first == 1:
                                            gsum = T.alloc_var(T.float32, init=grad_logprobs[row, k])
                                            for nxt in range(k + 1, topk):
                                                if T.cast(labels[row, nxt], T.int32) - vocab_start == lbl:
                                                    gsum = gsum + grad_logprobs[row, nxt]
                                            local_labels[row, k] = lbl
                                            scaled_glp[row, k] = gsum * inv_temperature
                                        else:
                                            local_labels[row, k] = -1
                                            scaled_glp[row, k] = 0.0

    return kernel


@tilelang.jit(target="ascend")
def _bwd_dlogits_scatter_multi(
    N: int,
    VOCAB_CHUNK: int,
    topk: int,
    dtype=T.bfloat16,
):
    """Apply pre-deduplicated sparse label gradients to the current `dlogits` vocabulary chunk."""
    dtype = _normalize_dtype(dtype)
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    ROWS = max(1, min(64, 8192 // max(1, topk)))
    ROWS = 64 if ROWS >= 64 else (32 if ROWS >= 32 else (16 if ROWS >= 16 else (8 if ROWS >= 8 else 1)))
    use_fast_dma = (ROWS >= 8) and (ROWS * topk <= 8192)
    ELEMS = max(8, ROWS * min(topk, 8192))

    @T.prim_func
    def kernel(
        local_labels: T.Tensor([M, topk], T.int32),
        scaled_glp: T.Tensor([M, topk], T.float32),
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.Kernel(num_cores) as core_id:
            lbl_1d = T.view(local_labels, (M * topk,), dtype=T.int32)
            glp_1d = T.view(scaled_glp, (M * topk,), dtype=T.float32)
            lbl_ub = T.alloc_shared((ELEMS,), T.int32)
            glp_ub = T.alloc_shared((ELEMS,), T.float32)
            chunk_end = T.min(vocab_offset + VOCAB_CHUNK, N)

            for m_tile in T.Persistent([T.ceildiv(M, ROWS)], num_cores, core_id):
                row_base = m_tile * ROWS
                if use_fast_dma and (row_base + ROWS <= M):
                    eb = row_base * topk
                    T.copy(lbl_1d[eb : eb + ROWS * topk], lbl_ub)
                    T.copy(glp_1d[eb : eb + ROWS * topk], glp_ub)
                    with T.SimtVF(threads=ROWS):
                        for i in T.Parallel(ROWS):
                            row = row_base + i
                            base = i * topk
                            for k in range(topk):
                                lbl = lbl_ub[base + k]
                                if (lbl >= vocab_offset) and (lbl < chunk_end):
                                    col = lbl - vocab_offset
                                    cur = T.cast(dlogits[row, col], T.float32)
                                    dlogits[row, col] = T.cast(cur + glp_ub[base + k], dtype)
                else:
                    with T.SimtVF(threads=max(32, ROWS)):
                        for i in T.Parallel(max(32, ROWS)):
                            row = row_base + i
                            if (i < ROWS) and (row < M):
                                for k in range(topk):
                                    lbl = local_labels[row, k]
                                    if (lbl >= vocab_offset) and (lbl < chunk_end):
                                        col = lbl - vocab_offset
                                        cur = T.cast(dlogits[row, col], T.float32)
                                        dlogits[row, col] = T.cast(cur + scaled_glp[row, k], dtype)

    return kernel


def linear_cross_entropy_bwd_dlogits(
    N: int,
    K: int,
    topk: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
    BLOCK_M=64,
    BLOCK_N=256,
    BLOCK_K=64,
    num_threads=128,
    num_stages=2,
    min_blocks_per_sm=1,
):
    """Recompute one vocabulary chunk and produce `dlogits` on Ascend.

    Decouples the dense `(M, VOCAB_CHUNK)` Cube + `SimdVF` softmax/entropy gradient
    from the sparse `O(M * topk)` target-label gradient update so runtime stays near
    raw Cube GEMM speed regardless of `topk`.
    """
    del BLOCK_M, BLOCK_N, BLOCK_K, num_threads, num_stages, min_blocks_per_sm
    norm_dtype = _normalize_dtype(dtype)
    dense_kernel = _bwd_dlogits_dense(
        N,
        K,
        VOCAB_CHUNK,
        HAS_LOGPROBS_GRAD,
        HAS_ENTROPY_GRAD,
        dtype=norm_dtype,
    )
    if HAS_LOGPROBS_GRAD:
        if topk == 1:
            prep_kernel = None
            scatter_kernel = _bwd_dlogits_scatter_top1(N, VOCAB_CHUNK, dtype=norm_dtype)
        else:
            prep_kernel = _bwd_dlogits_prep_multi(N, topk)
            scatter_kernel = _bwd_dlogits_scatter_multi(N, VOCAB_CHUNK, topk, dtype=norm_dtype)
    else:
        prep_kernel = None
        scatter_kernel = None

    state: dict[str, torch.Tensor | None] = {
        "grad_lp_sum": None,
        "local_labels": None,
        "scaled_glp": None,
    }

    def run(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        lse: torch.Tensor,
        mean_logits: torch.Tensor,
        grad_logprobs: torch.Tensor,
        grad_entropy: torch.Tensor,
        inv_temperature: float,
        vocab_start: int,
        vocab_offset: int,
        dlogits: torch.Tensor,
    ):
        M = hidden.shape[0]
        if not HAS_LOGPROBS_GRAD:
            grad_lp_sum = lse
        elif topk == 1:
            grad_lp_sum = grad_logprobs.reshape(M)
        else:
            if vocab_offset == 0 or state["grad_lp_sum"] is None:
                local_labels = torch.empty((M, topk), dtype=torch.int32, device=hidden.device)
                scaled_glp = torch.empty((M, topk), dtype=torch.float32, device=hidden.device)
                grad_lp_sum = torch.empty((M,), dtype=torch.float32, device=hidden.device)
                prep_kernel(
                    labels,
                    grad_logprobs,
                    inv_temperature,
                    vocab_start,
                    local_labels,
                    scaled_glp,
                    grad_lp_sum,
                )
                state["local_labels"] = local_labels
                state["scaled_glp"] = scaled_glp
                state["grad_lp_sum"] = grad_lp_sum
            grad_lp_sum = state["grad_lp_sum"]

        dense_kernel(
            hidden,
            weight,
            lse,
            mean_logits,
            grad_lp_sum,
            grad_entropy,
            inv_temperature,
            vocab_offset,
            dlogits,
        )

        if HAS_LOGPROBS_GRAD:
            if topk == 1:
                scatter_kernel(
                    labels,
                    grad_logprobs,
                    inv_temperature,
                    vocab_start,
                    vocab_offset,
                    dlogits,
                )
            else:
                scatter_kernel(
                    state["local_labels"],
                    state["scaled_glp"],
                    vocab_offset,
                    dlogits,
                )
                if vocab_offset + VOCAB_CHUNK >= N:
                    state["grad_lp_sum"] = None
                    state["local_labels"] = None
                    state["scaled_glp"] = None

    return run
