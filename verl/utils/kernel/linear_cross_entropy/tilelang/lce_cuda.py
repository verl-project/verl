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
    """Compute rank-local logits and online softmax statistics.

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
            exp_logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            weighted_logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

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
                    exp_logits[i, j] = T.if_then_else(
                        is_valid,
                        T.exp(logits[i, j] - new_max[i]),
                        0.0,
                    )
                    weighted_logits[i, j] = T.if_then_else(
                        is_valid,
                        logits[i, j] * exp_logits[i, j],
                        0.0,
                    )

                T.reduce_sum(exp_logits, tile_sum, dim=-1)
                T.reduce_sum(weighted_logits, tile_weighted_sum, dim=-1)
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
def linear_cross_entropy_bwd_dlogits(
    N: int,
    K: int,
    topk: int,
    VOCAB_CHUNK: int,
    HAS_LOGPROBS_GRAD: bool,
    HAS_ENTROPY_GRAD: bool,
    dtype=T.bfloat16,
    BLOCK_M=32,
    BLOCK_N=128,
    BLOCK_K=32,
    num_threads=128,
    num_stages=2,
    min_blocks_per_sm=1,
):
    """Recompute one vocabulary chunk and produce its dlogits."""
    assert VOCAB_CHUNK % BLOCK_N == 0
    M = T.dynamic("M")

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
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            probabilities = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            labels_local = T.alloc_fragment([BLOCK_M, topk], T.int64)
            grad_logprobs_local = T.alloc_fragment([BLOCK_M, topk], T.float32)
            grad_logprobs_sum = T.alloc_fragment([BLOCK_M], T.float32)
            grad_entropy_local = T.alloc_fragment([BLOCK_M], T.float32)
            lse_local = T.alloc_fragment([BLOCK_M], T.float32)
            mean_logits_local = T.alloc_fragment([BLOCK_M], T.float32)

            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                lse_local[i] = T.if_then_else(row < M, lse[row], 0.0)
                if HAS_ENTROPY_GRAD:
                    mean_logits_local[i] = T.if_then_else(row < M, mean_logits[row], 0.0)
                    grad_entropy_local[i] = T.if_then_else(row < M, grad_entropy[row], 0.0)
                else:
                    mean_logits_local[i] = 0.0
                    grad_entropy_local[i] = 0.0

            for i, label_idx in T.Parallel(BLOCK_M, topk):
                row = bx * BLOCK_M + i
                labels_local[i, label_idx] = T.if_then_else(row < M, labels[row, label_idx], 0)
                if HAS_LOGPROBS_GRAD:
                    grad_logprobs_local[i, label_idx] = T.if_then_else(
                        row < M,
                        grad_logprobs[row, label_idx],
                        0.0,
                    )
                else:
                    grad_logprobs_local[i, label_idx] = 0.0

            T.clear(grad_logprobs_sum)
            for label_idx in T.unroll(topk):
                for i in T.Parallel(BLOCK_M):
                    grad_logprobs_sum[i] += grad_logprobs_local[i, label_idx]

            local_vocab_offset = vocab_offset + by * BLOCK_N
            T.clear(logits)
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
                T.gemm(hidden_shared, weight_shared, logits, transpose_B=True)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                local_vocab_id = local_vocab_offset + j
                is_valid = (row < M) & (local_vocab_id < N)
                logits[i, j] = T.if_then_else(
                    is_valid,
                    logits[i, j] * inv_temperature,
                    0.0,
                )
                probabilities[i, j] = T.if_then_else(
                    is_valid,
                    T.exp(logits[i, j] - lse_local[i]),
                    0.0,
                )
                grad = -grad_logprobs_sum[i] * probabilities[i, j]
                if HAS_ENTROPY_GRAD:
                    grad -= grad_entropy_local[i] * probabilities[i, j] * (logits[i, j] - mean_logits_local[i])
                probabilities[i, j] = grad

            if HAS_LOGPROBS_GRAD:
                for label_idx in T.unroll(topk):
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        if labels_local[i, label_idx] == vocab_start + local_vocab_offset + j:
                            probabilities[i, j] += grad_logprobs_local[i, label_idx]

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                local_vocab_id = local_vocab_offset + j
                chunk_vocab_id = by * BLOCK_N + j
                is_valid = (row < M) & (local_vocab_id < N)
                if row < M:
                    dlogits[row, chunk_vocab_id] = T.if_then_else(
                        is_valid,
                        T.cast(probabilities[i, j] * inv_temperature, dtype),
                        0,
                    )

    return linear_cross_entropy_bwd_dlogits_kernel
