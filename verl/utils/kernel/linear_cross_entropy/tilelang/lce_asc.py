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
import tilelang
import tilelang.ascend.language as T
import torch


def _num_cube_cores() -> int:
    return torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num


def _num_vector_cores() -> int:
    return 2 * _num_cube_cores()


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_target_logits_ascend(
    N: int,
    K: int,
    topk: int,
    dtype=T.bfloat16,
    num_threads=128,
):
    """Compute selected logits on Ascend vector cores."""
    M = T.dynamic("M")
    num_cores = _num_vector_cores()

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
            for job in T.Persistent([M * topk], num_cores, core_id):
                row = job // topk
                label_idx = job % topk
                local_label = labels[row, label_idx] - vocab_start
                with T.SimtVF(threads=num_threads):
                    dot = T.alloc_reducer((1,), T.float32, op="sum")
                    T.reducer_init(dot)
                    for k in T.Parallel(K):
                        if (local_label >= 0) & (local_label < N):
                            T.reducer_update(
                                dot[0],
                                T.cast(hidden[row, k], T.float32) * T.cast(weight[local_label, k], T.float32),
                            )
                    result = T.alloc_fragment((1,), T.float32)
                    T.finalize_reducer(dot, result)
                    if (T.get_thread_binding() == 0) & (local_label >= 0) & (local_label < N):
                        target_logits[row, label_idx] = result[0] * inv_temperature

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_mainloop_ascend(
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
    """Compute rank-local online softmax statistics on Ascend mixed cores."""
    del min_blocks_per_sm
    assert BLOCK_M % 2 == 0
    assert VOCAB_PER_CTA % BLOCK_N == 0
    M = T.dynamic("M")
    half_m = BLOCK_M // 2
    num_splits = T.ceildiv(N, VOCAB_PER_CTA)
    num_vocab_tiles = VOCAB_PER_CTA // BLOCK_N
    num_cores = _num_cube_cores()

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
            hidden_l1 = T.alloc_l1((BLOCK_M, BLOCK_K), dtype)
            weight_l1 = T.alloc_l1((BLOCK_N, BLOCK_K), dtype)
            logits_l0c = T.alloc_l0c((BLOCK_M, BLOCK_N), T.float32)
            logits_ub = T.alloc_shared((half_m, BLOCK_N), T.float32)
            local_max_ub = T.alloc_shared((half_m,), T.float32)
            next_max_ub = T.alloc_shared((half_m,), T.float32)
            local_sum_ub = T.alloc_shared((half_m,), T.float32)
            local_weighted_sum_ub = T.alloc_shared((half_m,), T.float32)
            m_tiles = T.ceildiv(M, BLOCK_M)

            for m_tile, split in T.Persistent(
                [m_tiles, num_splits],
                num_cores,
                core_id,
                group_size=num_splits,
            ):
                m_offset = m_tile * BLOCK_M
                with T.SimtVF(threads=num_threads):
                    for i in T.Parallel(half_m):
                        local_max_ub[i] = -T.infinity(T.float32)
                        local_sum_ub[i] = 0.0
                        local_weighted_sum_ub[i] = 0.0

                for vocab_tile in T.Serial(num_vocab_tiles):
                    vocab_offset = split * VOCAB_PER_CTA + vocab_tile * BLOCK_N
                    for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                        T.copy(
                            hidden[
                                m_offset : m_offset + BLOCK_M,
                                k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                            ],
                            hidden_l1,
                        )
                        T.copy(
                            weight[
                                vocab_offset : vocab_offset + BLOCK_N,
                                k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                            ],
                            weight_l1,
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.gemm(
                            hidden_l1,
                            weight_l1,
                            logits_l0c,
                            transpose_B=True,
                            clear_accum=k_tile == 0,
                        )

                    T.dual_copy(logits_l0c, logits_ub)
                    with T.SimtVF(threads=num_threads):
                        tile_max = T.alloc_reducer((half_m,), T.float32, op="max")
                        T.reducer_init(tile_max)
                        for i, j in T.Parallel(half_m, BLOCK_N):
                            row = m_offset + sid * half_m + i
                            if (row < M) & (vocab_offset + j < N):
                                T.reducer_update(
                                    tile_max[i],
                                    logits_ub[i, j] * inv_temperature,
                                )
                        tile_max_result = T.alloc_fragment((half_m,), T.float32)
                        T.finalize_reducer(tile_max, tile_max_result)
                        for i in T.Parallel(half_m):
                            next_max_ub[i] = T.max(local_max_ub[i], tile_max_result[i])

                        tile_sum = T.alloc_reducer((half_m,), T.float32, op="sum")
                        tile_weighted_sum = T.alloc_reducer((half_m,), T.float32, op="sum")
                        T.reducer_init(tile_sum)
                        T.reducer_init(tile_weighted_sum)
                        for i, j in T.Parallel(half_m, BLOCK_N):
                            row = m_offset + sid * half_m + i
                            if (row < M) & (vocab_offset + j < N):
                                logit = logits_ub[i, j] * inv_temperature
                                exp_logit = T.exp(logit - next_max_ub[i])
                                T.reducer_update(tile_sum[i], exp_logit)
                                T.reducer_update(tile_weighted_sum[i], logit * exp_logit)
                        tile_sum_result = T.alloc_fragment((half_m,), T.float32)
                        tile_weighted_sum_result = T.alloc_fragment((half_m,), T.float32)
                        T.finalize_reducer(tile_sum, tile_sum_result)
                        T.finalize_reducer(tile_weighted_sum, tile_weighted_sum_result)
                        for i in T.Parallel(half_m):
                            rescale = T.exp(local_max_ub[i] - next_max_ub[i])
                            local_sum_ub[i] = local_sum_ub[i] * rescale + tile_sum_result[i]
                            local_weighted_sum_ub[i] = local_weighted_sum_ub[i] * rescale + tile_weighted_sum_result[i]
                            local_max_ub[i] = next_max_ub[i]

                with T.SimtVF(threads=num_threads):
                    for i in T.Parallel(half_m):
                        row = m_offset + sid * half_m + i
                        if row < M:
                            partial_max[row, split] = local_max_ub[i]
                            partial_sum[row, split] = local_sum_ub[i]
                            partial_weighted_sum[row, split] = local_weighted_sum_ub[i]

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_local_reduce_ascend(
    num_splits: int,
    BLOCK_M=8,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge rank-local vocabulary-split statistics."""
    del BLOCK_M, BLOCK_S
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    split_extent = tilelang.next_power_of_2(num_splits)

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
            for row in T.Persistent([M], num_cores, core_id):
                max_ub = T.alloc_shared((1,), T.float32)
                sum_ub = T.alloc_shared((1,), T.float32)
                weighted_sum_ub = T.alloc_shared((1,), T.float32)
                with T.SimtVF(threads=num_threads):
                    rank_max = T.alloc_reducer((1,), T.float32, op="max")
                    T.reducer_init(rank_max)
                    for split in T.Parallel(split_extent):
                        if split < num_splits:
                            T.reducer_update(rank_max[0], partial_max[row, split])
                    rank_max_result = T.alloc_fragment((1,), T.float32)
                    T.finalize_reducer(rank_max, rank_max_result)
                    T.copy(rank_max_result, max_ub)

                    rank_sum = T.alloc_reducer((1,), T.float32, op="sum")
                    rank_weighted_sum = T.alloc_reducer((1,), T.float32, op="sum")
                    T.reducer_init(rank_sum)
                    T.reducer_init(rank_weighted_sum)
                    for split in T.Parallel(split_extent):
                        if split < num_splits:
                            coefficient = T.exp(partial_max[row, split] - max_ub[0])
                            T.reducer_update(
                                rank_sum[0],
                                coefficient * partial_sum[row, split],
                            )
                            T.reducer_update(
                                rank_weighted_sum[0],
                                coefficient * partial_weighted_sum[row, split],
                            )
                    rank_sum_result = T.alloc_fragment((1,), T.float32)
                    rank_weighted_sum_result = T.alloc_fragment((1,), T.float32)
                    T.finalize_reducer(rank_sum, rank_sum_result)
                    T.finalize_reducer(rank_weighted_sum, rank_weighted_sum_result)
                    T.copy(rank_sum_result, sum_ub)
                    T.copy(rank_weighted_sum_result, weighted_sum_ub)

                    if T.get_thread_binding() == 0:
                        local_max[row] = max_ub[0]
                        local_sum[row] = sum_ub[0]
                        local_weighted_sum[row] = weighted_sum_ub[0]

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_finalize_local_ascend(
    num_splits: int,
    topk: int,
    BLOCK_M=8,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge local statistics and produce single-device outputs."""
    del BLOCK_M, BLOCK_S
    M = T.dynamic("M")
    num_cores = _num_vector_cores()
    split_extent = tilelang.next_power_of_2(num_splits)

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
            for row in T.Persistent([M], num_cores, core_id):
                max_ub = T.alloc_shared((1,), T.float32)
                sum_ub = T.alloc_shared((1,), T.float32)
                weighted_sum_ub = T.alloc_shared((1,), T.float32)
                with T.SimtVF(threads=num_threads):
                    rank_max = T.alloc_reducer((1,), T.float32, op="max")
                    T.reducer_init(rank_max)
                    for split in T.Parallel(split_extent):
                        if split < num_splits:
                            T.reducer_update(rank_max[0], partial_max[row, split])
                    rank_max_result = T.alloc_fragment((1,), T.float32)
                    T.finalize_reducer(rank_max, rank_max_result)
                    T.copy(rank_max_result, max_ub)

                    rank_sum = T.alloc_reducer((1,), T.float32, op="sum")
                    rank_weighted_sum = T.alloc_reducer((1,), T.float32, op="sum")
                    T.reducer_init(rank_sum)
                    T.reducer_init(rank_weighted_sum)
                    for split in T.Parallel(split_extent):
                        if split < num_splits:
                            coefficient = T.exp(partial_max[row, split] - max_ub[0])
                            T.reducer_update(
                                rank_sum[0],
                                coefficient * partial_sum[row, split],
                            )
                            T.reducer_update(
                                rank_weighted_sum[0],
                                coefficient * partial_weighted_sum[row, split],
                            )
                    rank_sum_result = T.alloc_fragment((1,), T.float32)
                    rank_weighted_sum_result = T.alloc_fragment((1,), T.float32)
                    T.finalize_reducer(rank_sum, rank_sum_result)
                    T.finalize_reducer(rank_weighted_sum, rank_weighted_sum_result)
                    T.copy(rank_sum_result, sum_ub)
                    T.copy(rank_weighted_sum_result, weighted_sum_ub)

                    if T.get_thread_binding() == 0:
                        row_lse = max_ub[0] + T.log(sum_ub[0])
                        row_mean = weighted_sum_ub[0] / sum_ub[0]
                        lse[row] = row_lse
                        mean_logits[row] = row_mean
                        entropy[row] = row_lse - row_mean
                    for label_idx in T.Parallel(topk):
                        logprobs[row, label_idx] = target_logits[row, label_idx] - max_ub[0] - T.log(sum_ub[0])

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_prepare_reduce_ascend(
    topk: int,
    BLOCK_M=64,
    num_threads=128,
):
    """Rescale local statistics to the TP-global maximum and pack them."""
    M = T.dynamic("M")
    num_cores = _num_vector_cores()

    @T.prim_func
    def kernel(
        local_max: T.Tensor([M], T.float32),
        global_max: T.Tensor([M], T.float32),
        local_sum: T.Tensor([M], T.float32),
        local_weighted_sum: T.Tensor([M], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
        reduced: T.Tensor([M, topk + 2], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            for row_tile in T.Persistent([T.ceildiv(M, BLOCK_M)], num_cores, core_id):
                with T.SimtVF(threads=num_threads):
                    for i in T.Parallel(BLOCK_M):
                        row = row_tile * BLOCK_M + i
                        if row < M:
                            scale = T.exp(local_max[row] - global_max[row])
                            reduced[row, 0] = local_sum[row] * scale
                            reduced[row, 1] = local_weighted_sum[row] * scale
                    for i, label_idx in T.Parallel(BLOCK_M, topk):
                        row = row_tile * BLOCK_M + i
                        if row < M:
                            reduced[row, label_idx + 2] = target_logits[row, label_idx]

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_fwd_finalize_ascend(
    topk: int,
    BLOCK_M=64,
    num_threads=128,
):
    """Produce TP-global logprobs and entropy."""
    M = T.dynamic("M")
    num_cores = _num_vector_cores()

    @T.prim_func
    def kernel(
        global_max: T.Tensor([M], T.float32),
        reduced: T.Tensor([M, topk + 2], T.float32),
        logprobs: T.Tensor([M, topk], T.float32),
        entropy: T.Tensor([M], T.float32),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
    ):
        with T.Kernel(num_cores) as core_id:
            for row_tile in T.Persistent([T.ceildiv(M, BLOCK_M)], num_cores, core_id):
                with T.SimtVF(threads=num_threads):
                    for i in T.Parallel(BLOCK_M):
                        row = row_tile * BLOCK_M + i
                        if row < M:
                            row_lse = global_max[row] + T.log(reduced[row, 0])
                            row_mean = reduced[row, 1] / reduced[row, 0]
                            lse[row] = row_lse
                            mean_logits[row] = row_mean
                            entropy[row] = row_lse - row_mean
                    for i, label_idx in T.Parallel(BLOCK_M, topk):
                        row = row_tile * BLOCK_M + i
                        if row < M:
                            logprobs[row, label_idx] = (
                                reduced[row, label_idx + 2] - global_max[row] - T.log(reduced[row, 0])
                            )

    return kernel


@tilelang.jit(target="ascend")
def linear_cross_entropy_bwd_dlogits_ascend(
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
    """Recompute one vocabulary chunk and produce dlogits on Ascend."""
    del min_blocks_per_sm
    assert BLOCK_M % 2 == 0
    assert VOCAB_CHUNK % BLOCK_N == 0
    M = T.dynamic("M")
    half_m = BLOCK_M // 2
    num_cores = _num_cube_cores()
    num_vocab_tiles = VOCAB_CHUNK // BLOCK_N

    @T.prim_func
    def kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
        grad_logprobs: T.Tensor([M, topk], T.float32),
        grad_entropy: T.Tensor([M], T.float32),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.MixedKernel(num_cores, sids=2) as (core_id, sid):
            hidden_l1 = T.alloc_l1((BLOCK_M, BLOCK_K), dtype)
            weight_l1 = T.alloc_l1((BLOCK_N, BLOCK_K), dtype)
            logits_l0c = T.alloc_l0c((BLOCK_M, BLOCK_N), T.float32)
            logits_ub = T.alloc_shared((half_m, BLOCK_N), T.float32)
            dlogits_ub = T.alloc_shared((half_m, BLOCK_N), dtype)
            m_tiles = T.ceildiv(M, BLOCK_M)

            for m_tile, vocab_tile in T.Persistent(
                [m_tiles, num_vocab_tiles],
                num_cores,
                core_id,
                group_size=num_vocab_tiles,
            ):
                m_offset = m_tile * BLOCK_M
                local_vocab_offset = vocab_offset + vocab_tile * BLOCK_N
                for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(
                        hidden[
                            m_offset : m_offset + BLOCK_M,
                            k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                        ],
                        hidden_l1,
                    )
                    T.copy(
                        weight[
                            local_vocab_offset : local_vocab_offset + BLOCK_N,
                            k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                        ],
                        weight_l1,
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.gemm(
                        hidden_l1,
                        weight_l1,
                        logits_l0c,
                        transpose_B=True,
                        clear_accum=k_tile == 0,
                    )

                T.dual_copy(logits_l0c, logits_ub)
                with T.SimtVF(threads=num_threads):
                    for i, j in T.Parallel(half_m, BLOCK_N):
                        row = m_offset + sid * half_m + i
                        local_vocab_id = local_vocab_offset + j
                        is_valid = (row < M) & (local_vocab_id < N)
                        if is_valid:
                            logit = logits_ub[i, j] * inv_temperature
                            probability = T.exp(logit - lse[row])
                            grad_logprobs_sum = T.alloc_var(T.float32, init=0.0)
                            if HAS_LOGPROBS_GRAD:
                                for label_idx in T.unroll(topk):
                                    grad_logprobs_sum += grad_logprobs[row, label_idx]
                            grad = T.alloc_var(
                                T.float32,
                                init=-grad_logprobs_sum * probability,
                            )
                            if HAS_ENTROPY_GRAD:
                                grad -= grad_entropy[row] * probability * (logit - mean_logits[row])
                            if HAS_LOGPROBS_GRAD:
                                for label_idx in T.unroll(topk):
                                    if labels[row, label_idx] == vocab_start + local_vocab_id:
                                        grad += grad_logprobs[row, label_idx]
                            dlogits_ub[i, j] = T.cast(grad * inv_temperature, dtype)
                        else:
                            dlogits_ub[i, j] = 0

                T.copy(
                    dlogits_ub,
                    dlogits[
                        m_offset + sid * half_m : m_offset + (sid + 1) * half_m,
                        vocab_tile * BLOCK_N : (vocab_tile + 1) * BLOCK_N,
                    ],
                    l2_cache_ctrl="NOTALLOC_PW",
                )

    return kernel


linear_cross_entropy_fwd_target_logits = linear_cross_entropy_fwd_target_logits_ascend
linear_cross_entropy_fwd_mainloop = linear_cross_entropy_fwd_mainloop_ascend
linear_cross_entropy_fwd_local_reduce = linear_cross_entropy_fwd_local_reduce_ascend
linear_cross_entropy_fwd_finalize_local = linear_cross_entropy_fwd_finalize_local_ascend
linear_cross_entropy_fwd_prepare_reduce = linear_cross_entropy_fwd_prepare_reduce_ascend
linear_cross_entropy_fwd_finalize = linear_cross_entropy_fwd_finalize_ascend
linear_cross_entropy_bwd_dlogits = linear_cross_entropy_bwd_dlogits_ascend
