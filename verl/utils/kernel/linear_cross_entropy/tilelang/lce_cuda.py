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
import tilelang.language as T


@tilelang.jit
def linear_cross_entropy_fwd_target_logits(
    N: int,
    K: int,
    topk: int,
    dtype=T.bfloat16,
    num_threads=128,
):
    """Compute selected logits without carrying labels through every vocab tile."""
    M = T.dynamic("M")

    @T.prim_func
    def linear_cross_entropy_fwd_target_logits_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        target_logits: T.Tensor([M, topk], T.float32),
    ):
        with T.Kernel(M, topk, threads=num_threads) as (row, label_idx):
            partial = T.alloc_fragment([num_threads], T.float32)
            result = T.alloc_fragment([1], T.float32)
            local_label = T.alloc_var(
                T.int64,
                init=labels[row, label_idx] - vocab_start,
            )
            T.clear(partial)
            for k_block in T.Serial(T.ceildiv(K, num_threads)):
                for lane in T.Parallel(num_threads):
                    k = k_block * num_threads + lane
                    if (local_label >= 0) & (local_label < N) & (k < K):
                        partial[lane] += T.cast(hidden[row, k], T.float32) * T.cast(weight[local_label, k], T.float32)
            T.reduce_sum(partial, result)
            if (T.get_thread_binding() == 0) & (local_label >= 0) & (local_label < N):
                target_logits[row, label_idx] = result[0] * inv_temperature

    return linear_cross_entropy_fwd_target_logits_kernel


@tilelang.jit
def linear_cross_entropy_fwd_mainloop_sm100(
    N: int,
    K: int,
    dtype=T.bfloat16,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    VOCAB_PER_CTA=512,
    num_threads=256,
    num_stages=5,
):
    """Blackwell SM100 2-CTA warp-specialized TMA + tcgen05 forward mainloop."""
    assert VOCAB_PER_CTA % BLOCK_N == 0
    M = T.dynamic("M")
    num_splits = (N + VOCAB_PER_CTA - 1) // VOCAB_PER_CTA
    num_vocab_tiles = VOCAB_PER_CTA // BLOCK_N
    k_iters = (K + BLOCK_K - 1) // BLOCK_K

    @T.prim_func
    def linear_cross_entropy_fwd_mainloop_sm100_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        inv_temperature: T.float32,
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
    ):
        with T.ClusterKernel(
            T.ceildiv(M, BLOCK_M * 2) * 2,
            num_splits,
            threads=num_threads,
            cluster_dims=2,
        ) as (bx, by):
            hidden_shared = T.alloc_shared([num_stages, BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([num_stages, BLOCK_N // 2, BLOCK_K], dtype)
            logits_tmem = T.alloc_tmem([BLOCK_M, BLOCK_N], T.float32)
            buf = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            local_max = T.alloc_fragment([BLOCK_M], T.float32)
            tile_max = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)
            local_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([BLOCK_M], T.float32)
            local_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)

            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])
            if num_vocab_tiles > 1:
                tmem_empty = T.alloc_cluster_barrier([num_threads * 2])

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)
            T.use_swizzle(16)

            T.fill(local_max, -T.infinity(T.float32))
            T.clear(local_sum)
            T.clear(local_weighted_sum)

            for vocab_tile in T.Serial(num_vocab_tiles):
                vocab_offset = by * VOCAB_PER_CTA + vocab_tile * BLOCK_N
                weight_cta_offset = vocab_offset + cta_id * (BLOCK_N // 2)

                if tx < 32:
                    for k_tile in T.Serial(k_iters):
                        step = vocab_tile * k_iters + k_tile
                        stage = step % num_stages
                        parity = (step // num_stages) & 1
                        T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                        T.tma_copy(
                            hidden[
                                bx * BLOCK_M : (bx + 1) * BLOCK_M,
                                k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                            ],
                            hidden_shared[stage, :, :],
                            barrier=loaded[stage],
                        )
                        T.tma_copy(
                            weight[
                                weight_cta_offset : weight_cta_offset + (BLOCK_N // 2),
                                k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                            ],
                            weight_shared[stage, :, :],
                            barrier=loaded[stage],
                        )
                        T.mbarrier_arrive(loaded[stage], 0)
                elif cta_id == 0 and tx < 64:
                    if num_vocab_tiles > 1 and vocab_tile > 0:
                        T.mbarrier_wait_parity(tmem_empty, (vocab_tile - 1) & 1)
                    for k_tile in T.Serial(k_iters):
                        step = vocab_tile * k_iters + k_tile
                        stage = step % num_stages
                        parity = (step // num_stages) & 1
                        T.mbarrier_wait_parity(loaded[stage], parity)
                        T.tcgen05_gemm(
                            hidden_shared[stage, :, :],
                            weight_shared[stage, :, :],
                            logits_tmem,
                            transpose_B=True,
                            mbar=consumed[stage],
                            clear_accum=k_tile == 0,
                            use_2cta=True,
                        )
                    T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

                T.mbarrier_wait_parity(tmem_full, vocab_tile & 1)

                T.copy(logits_tmem, buf)
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    buf[i, j] = T.if_then_else(
                        (bx * BLOCK_M + i < M) & (vocab_offset + j < N),
                        buf[i, j] * inv_temperature,
                        -T.infinity(T.float32),
                    )
                T.reduce_max(buf, tile_max, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    new_max[i] = T.max(local_max[i], tile_max[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    buf[i, j] = T.if_then_else(
                        (bx * BLOCK_M + i < M) & (vocab_offset + j < N),
                        T.exp(buf[i, j] - new_max[i]),
                        0.0,
                    )
                T.reduce_sum(buf, tile_sum, dim=-1)

                T.copy(logits_tmem, buf)
                if num_vocab_tiles > 1 and vocab_tile + 1 < num_vocab_tiles:
                    T.mbarrier_arrive(tmem_empty, 0)
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    is_valid = (bx * BLOCK_M + i < M) & (vocab_offset + j < N)
                    scaled = buf[i, j] * inv_temperature
                    buf[i, j] = T.if_then_else(
                        is_valid,
                        scaled * T.exp(scaled - new_max[i]),
                        0.0,
                    )
                T.reduce_sum(buf, tile_weighted_sum, dim=-1)

                for i in T.Parallel(BLOCK_M):
                    rescale = T.exp(local_max[i] - new_max[i])
                    local_sum[i] = local_sum[i] * rescale + tile_sum[i]
                    local_weighted_sum[i] = local_weighted_sum[i] * rescale + tile_weighted_sum[i]
                    local_max[i] = new_max[i]

            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    partial_max[bx * BLOCK_M + i, by] = local_max[i]
                    partial_sum[bx * BLOCK_M + i, by] = local_sum[i]
                    partial_weighted_sum[bx * BLOCK_M + i, by] = local_weighted_sum[i]

    return linear_cross_entropy_fwd_mainloop_sm100_kernel


@tilelang.jit
def linear_cross_entropy_fwd_mainloop_sm90(
    N: int,
    K: int,
    dtype=T.bfloat16,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    VOCAB_PER_CTA=128,
    num_threads=256,
    num_stages=3,
):
    """Hopper SM90 warp-specialized TMA + WGMMA forward mainloop (single vocab tile per CTA)."""
    assert VOCAB_PER_CTA == BLOCK_N
    M = T.dynamic("M")
    num_splits = (N + BLOCK_N - 1) // BLOCK_N

    @T.prim_func
    def linear_cross_entropy_fwd_mainloop_sm90_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        inv_temperature: T.float32,
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), num_splits, threads=num_threads) as (bx, by):
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            probs = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            tile_max = T.alloc_fragment([BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)

            T.use_swizzle(8)
            vocab_offset = by * BLOCK_N
            T.clear(logits)

            for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                T.copy(hidden[bx * BLOCK_M, k_tile * BLOCK_K], hidden_shared)
                T.copy(weight[vocab_offset, k_tile * BLOCK_K], weight_shared)
                T.gemm(hidden_shared, weight_shared, logits, transpose_B=True)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                logits[i, j] = T.if_then_else(
                    (bx * BLOCK_M + i < M) & (vocab_offset + j < N),
                    logits[i, j] * inv_temperature,
                    -T.infinity(T.float32),
                )

            T.reduce_max(logits, tile_max, dim=-1)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                is_valid = (bx * BLOCK_M + i < M) & (vocab_offset + j < N)
                probs[i, j] = T.if_then_else(
                    is_valid,
                    T.exp(logits[i, j] - tile_max[i]),
                    0.0,
                )
                logits[i, j] = T.if_then_else(
                    is_valid,
                    logits[i, j] * probs[i, j],
                    0.0,
                )

            T.reduce_sum(probs, tile_sum, dim=-1)
            T.reduce_sum(logits, tile_weighted_sum, dim=-1)
            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    partial_max[bx * BLOCK_M + i, by] = tile_max[i]
                    partial_sum[bx * BLOCK_M + i, by] = tile_sum[i]
                    partial_weighted_sum[bx * BLOCK_M + i, by] = tile_weighted_sum[i]

    return linear_cross_entropy_fwd_mainloop_sm90_kernel


@tilelang.jit
def linear_cross_entropy_fwd_mainloop(
    N: int,
    K: int,
    dtype=T.bfloat16,
    BLOCK_M=64,
    BLOCK_N=128,
    BLOCK_K=64,
    VOCAB_PER_CTA=512,
    num_threads=128,
    num_stages=2,
    min_blocks_per_sm=1,
):
    """Compute rank-local logits and online softmax statistics (generic CUDA fallback).

    Args:
        N: Local vocabulary size.
        K: Hidden size.
    """
    assert VOCAB_PER_CTA % BLOCK_N == 0
    M = T.dynamic("M")
    num_splits = (N + VOCAB_PER_CTA - 1) // VOCAB_PER_CTA
    num_vocab_tiles = VOCAB_PER_CTA // BLOCK_N

    @T.prim_func
    def linear_cross_entropy_fwd_mainloop_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        inv_temperature: T.float32,
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), num_splits, threads=num_threads) as (bx, by):
            T.annotate_min_blocks_per_sm(min_blocks_per_sm)
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            probs = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            local_max = T.alloc_fragment([BLOCK_M], T.float32)
            tile_max = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)
            local_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([BLOCK_M], T.float32)
            local_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)

            T.fill(local_max, -T.infinity(T.float32))
            T.clear(local_sum)
            T.clear(local_weighted_sum)

            for vocab_tile in T.Serial(num_vocab_tiles):
                vocab_offset = by * VOCAB_PER_CTA + vocab_tile * BLOCK_N
                T.clear(logits)

                for k_tile in T.Pipelined(
                    T.ceildiv(K, BLOCK_K),
                    num_stages=num_stages,
                ):
                    for i, k in T.Parallel(BLOCK_M, BLOCK_K):
                        hidden_shared[i, k] = T.if_then_else(
                            (bx * BLOCK_M + i < M) & (k_tile * BLOCK_K + k < K),
                            hidden[bx * BLOCK_M + i, k_tile * BLOCK_K + k],
                            0,
                        )
                    for j, k in T.Parallel(BLOCK_N, BLOCK_K):
                        weight_shared[j, k] = T.if_then_else(
                            (vocab_offset + j < N) & (k_tile * BLOCK_K + k < K),
                            weight[vocab_offset + j, k_tile * BLOCK_K + k],
                            0,
                        )
                    T.gemm(
                        hidden_shared,
                        weight_shared,
                        logits,
                        transpose_B=True,
                    )

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    logits[i, j] = T.if_then_else(
                        (bx * BLOCK_M + i < M) & (vocab_offset + j < N),
                        logits[i, j] * inv_temperature,
                        -T.infinity(T.float32),
                    )

                T.reduce_max(logits, tile_max, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    new_max[i] = T.max(local_max[i], tile_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    is_valid = (bx * BLOCK_M + i < M) & (vocab_offset + j < N)
                    probs[i, j] = T.if_then_else(
                        is_valid,
                        T.exp(logits[i, j] - new_max[i]),
                        0.0,
                    )
                    logits[i, j] = T.if_then_else(
                        is_valid,
                        logits[i, j] * probs[i, j],
                        0.0,
                    )

                T.reduce_sum(probs, tile_sum, dim=-1)
                T.reduce_sum(logits, tile_weighted_sum, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    rescale = T.exp(local_max[i] - new_max[i])
                    local_sum[i] = local_sum[i] * rescale + tile_sum[i]
                    local_weighted_sum[i] = local_weighted_sum[i] * rescale + tile_weighted_sum[i]
                    local_max[i] = new_max[i]

            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    partial_max[bx * BLOCK_M + i, by] = local_max[i]
                    partial_sum[bx * BLOCK_M + i, by] = local_sum[i]
                    partial_weighted_sum[bx * BLOCK_M + i, by] = local_weighted_sum[i]

    return linear_cross_entropy_fwd_mainloop_kernel


@tilelang.jit
def linear_cross_entropy_fwd_local_reduce(
    num_splits: int,
    BLOCK_M=32,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge the vocabulary-split statistics produced on one rank."""
    M = T.dynamic("M")
    split_layout = T.Fragment(
        (BLOCK_M, BLOCK_S),
        forward_thread_fn=lambda i, j: (i % (num_threads // 32)) * 32 + j,
    )

    @T.prim_func
    def linear_cross_entropy_fwd_local_reduce_kernel(
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
        local_max: T.Tensor([M], T.float32),
        local_sum: T.Tensor([M], T.float32),
        local_weighted_sum: T.Tensor([M], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=num_threads) as bx:
            rank_max = T.alloc_fragment([BLOCK_M], T.float32)
            rank_sum = T.alloc_fragment([BLOCK_M], T.float32)
            rank_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            split_max_values = T.alloc_fragment([BLOCK_M, BLOCK_S], T.float32)
            split_sum_values = T.alloc_fragment([BLOCK_M, BLOCK_S], T.float32)
            split_weighted_sum_values = T.alloc_fragment([BLOCK_M, BLOCK_S], T.float32)
            split_max = T.alloc_fragment([BLOCK_M], T.float32)
            split_sum = T.alloc_fragment([BLOCK_M], T.float32)
            split_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)
            T.annotate_layout(
                {
                    split_max_values: split_layout,
                    split_sum_values: split_layout,
                    split_weighted_sum_values: split_layout,
                }
            )

            T.fill(rank_max, -T.infinity(T.float32))
            T.clear(rank_sum)
            T.clear(rank_weighted_sum)

            for split_block in T.Serial(T.ceildiv(num_splits, BLOCK_S)):
                for i, j in T.Parallel(
                    BLOCK_M,
                    BLOCK_S,
                    loop_layout=split_layout,
                ):
                    row = bx * BLOCK_M + i
                    split = split_block * BLOCK_S + j
                    is_valid = (row < M) & (split < num_splits)
                    split_max_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_max[row, split],
                        -T.infinity(T.float32),
                    )
                    split_sum_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_sum[row, split],
                        0.0,
                    )
                    split_weighted_sum_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_weighted_sum[row, split],
                        0.0,
                    )

                T.reduce_max(split_max_values, split_max, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    new_max[i] = T.max(rank_max[i], split_max[i])

                for i, j in T.Parallel(
                    BLOCK_M,
                    BLOCK_S,
                    loop_layout=split_layout,
                ):
                    coefficient = T.exp(split_max_values[i, j] - new_max[i])
                    split_sum_values[i, j] *= coefficient
                    split_weighted_sum_values[i, j] *= coefficient

                T.reduce_sum(split_sum_values, split_sum, dim=-1)
                T.reduce_sum(
                    split_weighted_sum_values,
                    split_weighted_sum,
                    dim=-1,
                )
                for i in T.Parallel(BLOCK_M):
                    rank_rescale = T.exp(rank_max[i] - new_max[i])
                    rank_sum[i] = rank_sum[i] * rank_rescale + split_sum[i]
                    rank_weighted_sum[i] = rank_weighted_sum[i] * rank_rescale + split_weighted_sum[i]
                    rank_max[i] = new_max[i]

            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    local_max[row] = rank_max[i]
                    local_sum[row] = rank_sum[i]
                    local_weighted_sum[row] = rank_weighted_sum[i]

    return linear_cross_entropy_fwd_local_reduce_kernel


@tilelang.jit
def linear_cross_entropy_fwd_finalize_local(
    num_splits: int,
    topk: int,
    BLOCK_M=32,
    BLOCK_S=32,
    num_threads=128,
):
    """Merge rank-local statistics and produce single-device outputs."""
    M = T.dynamic("M")
    split_layout = T.Fragment(
        (BLOCK_M, BLOCK_S),
        forward_thread_fn=lambda i, j: (i % (num_threads // 32)) * 32 + j,
    )

    @T.prim_func
    def linear_cross_entropy_fwd_finalize_local_kernel(
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
        logprobs: T.Tensor([M, topk], T.float32),
        entropy: T.Tensor([M], T.float32),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=num_threads) as bx:
            rank_max = T.alloc_fragment([BLOCK_M], T.float32)
            rank_sum = T.alloc_fragment([BLOCK_M], T.float32)
            rank_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            split_max_values = T.alloc_fragment(
                [BLOCK_M, BLOCK_S],
                T.float32,
            )
            split_sum_values = T.alloc_fragment(
                [BLOCK_M, BLOCK_S],
                T.float32,
            )
            split_weighted_sum_values = T.alloc_fragment(
                [BLOCK_M, BLOCK_S],
                T.float32,
            )
            split_max = T.alloc_fragment([BLOCK_M], T.float32)
            split_sum = T.alloc_fragment([BLOCK_M], T.float32)
            split_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)
            T.annotate_layout(
                {
                    split_max_values: split_layout,
                    split_sum_values: split_layout,
                    split_weighted_sum_values: split_layout,
                }
            )

            T.fill(rank_max, -T.infinity(T.float32))
            T.clear(rank_sum)
            T.clear(rank_weighted_sum)

            for split_block in T.Serial(T.ceildiv(num_splits, BLOCK_S)):
                for i, j in T.Parallel(
                    BLOCK_M,
                    BLOCK_S,
                    loop_layout=split_layout,
                ):
                    row = bx * BLOCK_M + i
                    split = split_block * BLOCK_S + j
                    is_valid = (row < M) & (split < num_splits)
                    split_max_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_max[row, split],
                        -T.infinity(T.float32),
                    )
                    split_sum_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_sum[row, split],
                        0.0,
                    )
                    split_weighted_sum_values[i, j] = T.if_then_else(
                        is_valid,
                        partial_weighted_sum[row, split],
                        0.0,
                    )

                T.reduce_max(split_max_values, split_max, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    new_max[i] = T.max(rank_max[i], split_max[i])

                for i, j in T.Parallel(
                    BLOCK_M,
                    BLOCK_S,
                    loop_layout=split_layout,
                ):
                    coefficient = T.exp(split_max_values[i, j] - new_max[i])
                    split_sum_values[i, j] *= coefficient
                    split_weighted_sum_values[i, j] *= coefficient

                T.reduce_sum(split_sum_values, split_sum, dim=-1)
                T.reduce_sum(
                    split_weighted_sum_values,
                    split_weighted_sum,
                    dim=-1,
                )
                for i in T.Parallel(BLOCK_M):
                    rank_rescale = T.exp(rank_max[i] - new_max[i])
                    rank_sum[i] = rank_sum[i] * rank_rescale + split_sum[i]
                    rank_weighted_sum[i] = rank_weighted_sum[i] * rank_rescale + split_weighted_sum[i]
                    rank_max[i] = new_max[i]

            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                if row < M:
                    row_lse = rank_max[i] + T.log(rank_sum[i])
                    row_mean = rank_weighted_sum[i] / rank_sum[i]
                    lse[row] = row_lse
                    mean_logits[row] = row_mean
                    entropy[row] = row_lse - row_mean

            for i, label_idx in T.Parallel(BLOCK_M, topk):
                row = bx * BLOCK_M + i
                if row < M:
                    logprobs[row, label_idx] = target_logits[row, label_idx] - rank_max[i] - T.log(rank_sum[i])

    return linear_cross_entropy_fwd_finalize_local_kernel


@tilelang.jit
def linear_cross_entropy_fwd_prepare_reduce(
    topk: int,
    BLOCK_M=128,
    num_threads=128,
):
    """Rescale rank-local statistics to the TP-global maximum and pack them."""
    M = T.dynamic("M")

    @T.prim_func
    def linear_cross_entropy_fwd_prepare_reduce_kernel(
        local_max: T.Tensor([M], T.float32),
        global_max: T.Tensor([M], T.float32),
        local_sum: T.Tensor([M], T.float32),
        local_weighted_sum: T.Tensor([M], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
        reduced: T.Tensor([M, topk + 2], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=num_threads) as bx:
            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    scale = T.exp(local_max[row] - global_max[row])
                    reduced[row, 0] = local_sum[row] * scale
                    reduced[row, 1] = local_weighted_sum[row] * scale

            for i, label_idx in T.Parallel(BLOCK_M, topk):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    reduced[row, label_idx + 2] = target_logits[row, label_idx]

    return linear_cross_entropy_fwd_prepare_reduce_kernel


@tilelang.jit
def linear_cross_entropy_fwd_finalize(
    topk: int,
    BLOCK_M=128,
    num_threads=128,
):
    """Produce logprobs and entropy from TP-global reduced statistics."""
    M = T.dynamic("M")

    @T.prim_func
    def linear_cross_entropy_fwd_finalize_kernel(
        global_max: T.Tensor([M], T.float32),
        reduced: T.Tensor([M, topk + 2], T.float32),
        logprobs: T.Tensor([M, topk], T.float32),
        entropy: T.Tensor([M], T.float32),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=num_threads) as bx:
            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    row_lse = global_max[row] + T.log(reduced[row, 0])
                    row_mean = reduced[row, 1] / reduced[row, 0]
                    lse[row] = row_lse
                    mean_logits[row] = row_mean
                    entropy[row] = row_lse - row_mean

            for i, label_idx in T.Parallel(BLOCK_M, topk):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    row_lse = global_max[row] + T.log(reduced[row, 0])
                    logprobs[row, label_idx] = reduced[row, label_idx + 2] - row_lse

    return linear_cross_entropy_fwd_finalize_kernel


@tilelang.jit
def linear_cross_entropy_bwd_dlogits_sm100(
    N: int,
    K: int,
    topk: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    num_threads=256,
    num_stages=4,
):
    """Blackwell SM100 warp-specialized TMA + tcgen05 backward dlogits kernel."""
    assert VOCAB_CHUNK % BLOCK_N == 0
    M = T.dynamic("M")
    k_iters = (K + BLOCK_K - 1) // BLOCK_K
    lanes_per_row = num_threads // BLOCK_M
    topk_iters = (topk + lanes_per_row - 1) // lanes_per_row
    use_small_k = topk <= 8

    @T.prim_func
    def linear_cross_entropy_bwd_dlogits_sm100_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
        grad_logprobs: T.Tensor([M, topk], T.float32),
        grad_logprobs_sum: T.Tensor([M], T.float32),
        grad_entropy: T.Tensor([M], T.float32),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, BLOCK_M),
            VOCAB_CHUNK // BLOCK_N,
            threads=num_threads,
        ) as (bx, by):
            hidden_shared = T.alloc_shared([num_stages, BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([num_stages, BLOCK_N, BLOCK_K], dtype)
            logits_tmem = T.alloc_tmem([BLOCK_M, BLOCK_N], T.float32)
            buf = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            if not use_small_k and HAS_LOGPROBS_GRAD:
                dlogits_fp32_s = T.alloc_shared([BLOCK_M, BLOCK_N], T.float32)
            dlogits_s = T.alloc_shared([BLOCK_M, BLOCK_N], dtype)

            lse_s = T.alloc_shared([BLOCK_M], T.float32)
            mean_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_ent_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_lp_sum_s = T.alloc_shared([BLOCK_M], T.float32)
            if use_small_k and HAS_LOGPROBS_GRAD:
                label_col_s = T.alloc_shared([BLOCK_M, topk], T.int64)
                grad_lp_s = T.alloc_shared([BLOCK_M, topk], T.float32)

            loaded = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            T.use_swizzle(8)

            local_vocab_offset = vocab_offset + by * BLOCK_N
            tile_global_vocab = vocab_start + local_vocab_offset

            if tx < 32:
                for k_tile in T.Serial(k_iters):
                    stage = k_tile % num_stages
                    parity = (k_tile // num_stages) & 1
                    T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                    T.tma_copy(
                        hidden[
                            bx * BLOCK_M : (bx + 1) * BLOCK_M,
                            k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                        ],
                        hidden_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        weight[
                            local_vocab_offset : local_vocab_offset + BLOCK_N,
                            k_tile * BLOCK_K : (k_tile + 1) * BLOCK_K,
                        ],
                        weight_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.mbarrier_arrive(loaded[stage])
            elif tx < 64:
                for k_tile in T.Serial(k_iters):
                    stage = k_tile % num_stages
                    parity = (k_tile // num_stages) & 1
                    T.mbarrier_wait_parity(loaded[stage], parity)
                    T.tcgen05_gemm(
                        hidden_shared[stage, :, :],
                        weight_shared[stage, :, :],
                        logits_tmem,
                        transpose_B=True,
                        mbar=consumed[stage],
                        clear_accum=k_tile == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            # Epilogue shared buffers alias input shared tiles; populate strictly after MMA completion.
            T.mbarrier_wait_parity(tmem_full, 0)

            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                lse_s[i] = T.if_then_else(row < M, lse[row], 0.0)
                if HAS_LOGPROBS_GRAD:
                    grad_lp_sum_s[i] = T.if_then_else(row < M, grad_logprobs_sum[row], 0.0)
                else:
                    grad_lp_sum_s[i] = 0.0
                if HAS_ENTROPY_GRAD:
                    mean_s[i] = T.if_then_else(row < M, mean_logits[row], 0.0)
                    grad_ent_s[i] = T.if_then_else(row < M, grad_entropy[row], 0.0)
                else:
                    mean_s[i] = 0.0
                    grad_ent_s[i] = 0.0

            if use_small_k and HAS_LOGPROBS_GRAD:
                for i, idx in T.Parallel(BLOCK_M, topk):
                    row = bx * BLOCK_M + i
                    label_col_s[i, idx] = T.if_then_else(row < M, labels[row, idx] - tile_global_vocab, -1)
                    grad_lp_s[i, idx] = T.if_then_else(row < M, grad_logprobs[row, idx], 0.0)

            T.copy(logits_tmem, buf)

            if use_small_k or not HAS_LOGPROBS_GRAD:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = T.alloc_var(T.float32, init=-grad_lp_sum_s[i] * prob)
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    if use_small_k and HAS_LOGPROBS_GRAD:
                        for idx in T.unroll(topk):
                            if j == label_col_s[i, idx]:
                                grad += grad_lp_s[i, idx]
                    dlogits_s[i, j] = T.if_then_else(
                        is_valid,
                        T.cast(grad * inv_temperature, dtype),
                        T.cast(0.0, dtype),
                    )
            else:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = -grad_lp_sum_s[i] * prob
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    dlogits_fp32_s[i, j] = T.if_then_else(is_valid, grad, 0.0)

                for kt in T.Serial(topk_iters):
                    for i, lane in T.Parallel(BLOCK_M, lanes_per_row):
                        row = bx * BLOCK_M + i
                        idx = kt * lanes_per_row + lane
                        if (row < M) & (idx < topk):
                            col = labels[row, idx] - tile_global_vocab
                            if (col >= 0) & (col < BLOCK_N) & (local_vocab_offset + col < N):
                                T.atomic_add(dlogits_fp32_s[i, col], grad_logprobs[row, idx])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    dlogits_s[i, j] = T.if_then_else(
                        is_valid,
                        T.cast(dlogits_fp32_s[i, j] * inv_temperature, dtype),
                        T.cast(0.0, dtype),
                    )

            T.copy(dlogits_s, dlogits[bx * BLOCK_M, by * BLOCK_N])

    return linear_cross_entropy_bwd_dlogits_sm100_kernel


@tilelang.jit
def linear_cross_entropy_bwd_dlogits_sm90(
    N: int,
    K: int,
    topk: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    num_threads=256,
    num_stages=3,
):
    """Hopper SM90 TMA + WGMMA backward dlogits kernel."""
    assert VOCAB_CHUNK % BLOCK_N == 0
    M = T.dynamic("M")
    lanes_per_row = num_threads // BLOCK_M
    topk_iters = (topk + lanes_per_row - 1) // lanes_per_row
    use_small_k = topk <= 8

    @T.prim_func
    def linear_cross_entropy_bwd_dlogits_sm90_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        lse: T.Tensor([M], T.float32),
        mean_logits: T.Tensor([M], T.float32),
        grad_logprobs: T.Tensor([M, topk], T.float32),
        grad_logprobs_sum: T.Tensor([M], T.float32),
        grad_entropy: T.Tensor([M], T.float32),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        vocab_offset: T.int32,
        dlogits: T.Tensor([M, VOCAB_CHUNK], dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, BLOCK_M),
            VOCAB_CHUNK // BLOCK_N,
            threads=num_threads,
        ) as (bx, by):
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            buf = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            if not use_small_k and HAS_LOGPROBS_GRAD:
                dlogits_fp32_s = T.alloc_shared([BLOCK_M, BLOCK_N], T.float32)
            dlogits_s = T.alloc_shared([BLOCK_M, BLOCK_N], dtype)

            lse_s = T.alloc_shared([BLOCK_M], T.float32)
            mean_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_ent_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_lp_sum_s = T.alloc_shared([BLOCK_M], T.float32)
            if use_small_k and HAS_LOGPROBS_GRAD:
                label_col_s = T.alloc_shared([BLOCK_M, topk], T.int64)
                grad_lp_s = T.alloc_shared([BLOCK_M, topk], T.float32)

            T.use_swizzle(8)
            local_vocab_offset = vocab_offset + by * BLOCK_N
            tile_global_vocab = vocab_start + local_vocab_offset

            T.clear(buf)
            for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                T.copy(hidden[bx * BLOCK_M, k_tile * BLOCK_K], hidden_shared)
                T.copy(weight[local_vocab_offset, k_tile * BLOCK_K], weight_shared)
                T.gemm(hidden_shared, weight_shared, buf, transpose_B=True)

            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                lse_s[i] = T.if_then_else(row < M, lse[row], 0.0)
                if HAS_LOGPROBS_GRAD:
                    grad_lp_sum_s[i] = T.if_then_else(row < M, grad_logprobs_sum[row], 0.0)
                else:
                    grad_lp_sum_s[i] = 0.0
                if HAS_ENTROPY_GRAD:
                    mean_s[i] = T.if_then_else(row < M, mean_logits[row], 0.0)
                    grad_ent_s[i] = T.if_then_else(row < M, grad_entropy[row], 0.0)
                else:
                    mean_s[i] = 0.0
                    grad_ent_s[i] = 0.0

            if use_small_k and HAS_LOGPROBS_GRAD:
                for i, idx in T.Parallel(BLOCK_M, topk):
                    row = bx * BLOCK_M + i
                    label_col_s[i, idx] = T.if_then_else(row < M, labels[row, idx] - tile_global_vocab, -1)
                    grad_lp_s[i, idx] = T.if_then_else(row < M, grad_logprobs[row, idx], 0.0)

            if use_small_k or not HAS_LOGPROBS_GRAD:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = T.alloc_var(T.float32, init=-grad_lp_sum_s[i] * prob)
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    if use_small_k and HAS_LOGPROBS_GRAD:
                        for idx in T.unroll(topk):
                            if j == label_col_s[i, idx]:
                                grad += grad_lp_s[i, idx]
                    dlogits_s[i, j] = T.if_then_else(
                        is_valid,
                        T.cast(grad * inv_temperature, dtype),
                        T.cast(0.0, dtype),
                    )
            else:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = -grad_lp_sum_s[i] * prob
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    dlogits_fp32_s[i, j] = T.if_then_else(is_valid, grad, 0.0)

                for kt in T.Serial(topk_iters):
                    for i, lane in T.Parallel(BLOCK_M, lanes_per_row):
                        row = bx * BLOCK_M + i
                        idx = kt * lanes_per_row + lane
                        if (row < M) & (idx < topk):
                            col = labels[row, idx] - tile_global_vocab
                            if (col >= 0) & (col < BLOCK_N) & (local_vocab_offset + col < N):
                                T.atomic_add(dlogits_fp32_s[i, col], grad_logprobs[row, idx])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    dlogits_s[i, j] = T.if_then_else(
                        is_valid,
                        T.cast(dlogits_fp32_s[i, j] * inv_temperature, dtype),
                        T.cast(0.0, dtype),
                    )

            T.copy(dlogits_s, dlogits[bx * BLOCK_M, by * BLOCK_N])

    return linear_cross_entropy_bwd_dlogits_sm90_kernel


@tilelang.jit
def linear_cross_entropy_bwd_dlogits(
    N: int,
    K: int,
    topk: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
    BLOCK_M=64,
    BLOCK_N=128,
    BLOCK_K=64,
    num_threads=128,
    num_stages=2,
    min_blocks_per_sm=1,
):
    """Recompute one vocabulary chunk and produce its dlogits (generic CUDA fallback)."""
    assert VOCAB_CHUNK % BLOCK_N == 0
    M = T.dynamic("M")
    lanes_per_row = max(1, num_threads // BLOCK_M)
    topk_iters = (topk + lanes_per_row - 1) // lanes_per_row
    use_small_k = topk <= 8

    @T.prim_func
    def linear_cross_entropy_bwd_dlogits_kernel(
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
        with T.Kernel(
            T.ceildiv(M, BLOCK_M),
            T.ceildiv(VOCAB_CHUNK, BLOCK_N),
            threads=num_threads,
        ) as (bx, by):
            T.annotate_min_blocks_per_sm(min_blocks_per_sm)
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            buf = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            if not use_small_k and HAS_LOGPROBS_GRAD:
                dlogits_fp32_s = T.alloc_shared([BLOCK_M, BLOCK_N], T.float32)

            lse_s = T.alloc_shared([BLOCK_M], T.float32)
            mean_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_ent_s = T.alloc_shared([BLOCK_M], T.float32)
            grad_lp_sum_s = T.alloc_shared([BLOCK_M], T.float32)
            if use_small_k and HAS_LOGPROBS_GRAD:
                label_col_s = T.alloc_shared([BLOCK_M, topk], T.int64)
                grad_lp_s = T.alloc_shared([BLOCK_M, topk], T.float32)

            local_vocab_offset = vocab_offset + by * BLOCK_N
            tile_global_vocab = vocab_start + local_vocab_offset

            T.clear(buf)
            for k_tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                for i, k in T.Parallel(BLOCK_M, BLOCK_K):
                    hidden_shared[i, k] = T.if_then_else(
                        (bx * BLOCK_M + i < M) & (k_tile * BLOCK_K + k < K),
                        hidden[bx * BLOCK_M + i, k_tile * BLOCK_K + k],
                        0,
                    )
                for j, k in T.Parallel(BLOCK_N, BLOCK_K):
                    weight_shared[j, k] = T.if_then_else(
                        (local_vocab_offset + j < N) & (k_tile * BLOCK_K + k < K),
                        weight[local_vocab_offset + j, k_tile * BLOCK_K + k],
                        0,
                    )
                T.gemm(hidden_shared, weight_shared, buf, transpose_B=True)

            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                lse_s[i] = T.if_then_else(row < M, lse[row], 0.0)
                grad_lp_sum_s[i] = 0.0
                if HAS_ENTROPY_GRAD:
                    mean_s[i] = T.if_then_else(row < M, mean_logits[row], 0.0)
                    grad_ent_s[i] = T.if_then_else(row < M, grad_entropy[row], 0.0)
                else:
                    mean_s[i] = 0.0
                    grad_ent_s[i] = 0.0

            if use_small_k and HAS_LOGPROBS_GRAD:
                for i, idx in T.Parallel(BLOCK_M, topk):
                    row = bx * BLOCK_M + i
                    label_col_s[i, idx] = T.if_then_else(row < M, labels[row, idx] - tile_global_vocab, -1)
                    grad_lp_s[i, idx] = T.if_then_else(row < M, grad_logprobs[row, idx], 0.0)
                for i in T.Parallel(BLOCK_M):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for idx in T.unroll(topk):
                        acc += grad_lp_s[i, idx]
                    grad_lp_sum_s[i] = acc
            elif HAS_LOGPROBS_GRAD:
                for kt in T.Serial(topk_iters):
                    for i, lane in T.Parallel(BLOCK_M, lanes_per_row):
                        row = bx * BLOCK_M + i
                        idx = kt * lanes_per_row + lane
                        if (row < M) & (idx < topk):
                            T.atomic_add(grad_lp_sum_s[i], grad_logprobs[row, idx])

            if use_small_k or not HAS_LOGPROBS_GRAD:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    chunk_vocab_id = by * BLOCK_N + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = T.alloc_var(T.float32, init=-grad_lp_sum_s[i] * prob)
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    if use_small_k and HAS_LOGPROBS_GRAD:
                        for idx in T.unroll(topk):
                            if j == label_col_s[i, idx]:
                                grad += grad_lp_s[i, idx]
                    if row < M:
                        dlogits[row, chunk_vocab_id] = T.if_then_else(
                            is_valid,
                            T.cast(grad * inv_temperature, dtype),
                            0,
                        )
            else:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    logit = buf[i, j] * inv_temperature
                    prob = T.if_then_else(is_valid, T.exp(logit - lse_s[i]), 0.0)
                    grad = -grad_lp_sum_s[i] * prob
                    if HAS_ENTROPY_GRAD:
                        grad -= grad_ent_s[i] * prob * (logit - mean_s[i])
                    dlogits_fp32_s[i, j] = T.if_then_else(is_valid, grad, 0.0)

                for kt in T.Serial(topk_iters):
                    for i, lane in T.Parallel(BLOCK_M, lanes_per_row):
                        row = bx * BLOCK_M + i
                        idx = kt * lanes_per_row + lane
                        if (row < M) & (idx < topk):
                            col = labels[row, idx] - tile_global_vocab
                            if (col >= 0) & (col < BLOCK_N) & (local_vocab_offset + col < N):
                                T.atomic_add(dlogits_fp32_s[i, col], grad_logprobs[row, idx])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    local_vocab_id = local_vocab_offset + j
                    chunk_vocab_id = by * BLOCK_N + j
                    is_valid = (row < M) & (local_vocab_id < N)
                    if row < M:
                        dlogits[row, chunk_vocab_id] = T.if_then_else(
                            is_valid,
                            T.cast(dlogits_fp32_s[i, j] * inv_temperature, dtype),
                            0,
                        )

    return linear_cross_entropy_bwd_dlogits_kernel
