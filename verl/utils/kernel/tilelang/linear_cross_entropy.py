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
import torch

from verl.utils.device import get_device_name


@tilelang.jit
def linear_cross_entropy_fwd_mainloop(
    N: int,
    K: int,
    topk: int,
    dtype=T.bfloat16,
    BLOCK_M=64,
    BLOCK_N=256,
    BLOCK_K=64,
    VOCAB_PER_CTA=512,
    num_threads=128,
    num_stages=2,
):
    """Compute rank-local logits and online softmax statistics.

    Args:
        N: Local vocabulary size.
        K: Hidden size.
        topk: Number of label logprobs to compute.
    """
    assert VOCAB_PER_CTA % BLOCK_N == 0
    M = T.dynamic("M")
    num_splits = (N + VOCAB_PER_CTA - 1) // VOCAB_PER_CTA
    num_vocab_tiles = VOCAB_PER_CTA // BLOCK_N

    @T.prim_func
    def linear_cross_entropy_fwd_mainloop_kernel(
        hidden: T.Tensor([M, K], dtype),
        weight: T.Tensor([N, K], dtype),
        labels: T.Tensor([M, topk], T.int64),
        inv_temperature: T.float32,
        vocab_start: T.int32,
        partial_max: T.Tensor([M, num_splits], T.float32),
        partial_sum: T.Tensor([M, num_splits], T.float32),
        partial_weighted_sum: T.Tensor([M, num_splits], T.float32),
        target_logits: T.Tensor([M, topk], T.float32),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), num_splits, threads=num_threads) as (bx, by):
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            exp_logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            weighted_logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)

            labels_local = T.alloc_fragment([BLOCK_M, topk], T.int64)
            target_logits_local = T.alloc_fragment([BLOCK_M, topk], T.float32)
            local_max = T.alloc_fragment([BLOCK_M], T.float32)
            tile_max = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)
            local_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([BLOCK_M], T.float32)
            local_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            tile_target_logit = T.alloc_fragment([BLOCK_M], T.float32)

            for i, label_idx in T.Parallel(BLOCK_M, topk):
                labels_local[i, label_idx] = T.if_then_else(
                    bx * BLOCK_M + i < M,
                    labels[bx * BLOCK_M + i, label_idx],
                    0,
                )
                target_logits_local[i, label_idx] = 0.0

            T.fill(local_max, -T.infinity(T.float32))
            T.clear(local_sum)
            T.clear(local_weighted_sum)

            for vocab_tile in T.Serial(num_vocab_tiles):
                vocab_offset = by * VOCAB_PER_CTA + vocab_tile * BLOCK_N
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
                            (vocab_offset + j < N) & (k_tile * BLOCK_K + k < K),
                            weight[vocab_offset + j, k_tile * BLOCK_K + k],
                            0,
                        )
                    T.gemm(hidden_shared, weight_shared, logits, transpose_B=True)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    logits[i, j] = T.if_then_else(
                        (bx * BLOCK_M + i < M) & (vocab_offset + j < N),
                        logits[i, j] * inv_temperature,
                        -T.infinity(T.float32),
                    )

                for label_idx in T.unroll(topk):
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        weighted_logits[i, j] = T.if_then_else(
                            labels_local[i, label_idx] == vocab_start + vocab_offset + j,
                            logits[i, j],
                            0.0,
                        )
                    T.reduce_sum(weighted_logits, tile_target_logit, dim=-1)
                    for i in T.Parallel(BLOCK_M):
                        target_logits_local[i, label_idx] += tile_target_logit[i]

                T.reduce_max(logits, tile_max, dim=-1)
                for i in T.Parallel(BLOCK_M):
                    new_max[i] = T.max(local_max[i], tile_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    is_valid = (bx * BLOCK_M + i < M) & (vocab_offset + j < N)
                    exp_logits[i, j] = T.if_then_else(is_valid, T.exp(logits[i, j] - new_max[i]), 0.0)
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

            split_start = vocab_start + by * VOCAB_PER_CTA
            split_end = vocab_start + T.min((by + 1) * VOCAB_PER_CTA, N)
            for i, label_idx in T.Parallel(BLOCK_M, topk):
                label = labels_local[i, label_idx]
                if (bx * BLOCK_M + i < M) & (label >= split_start) & (label < split_end):
                    target_logits[bx * BLOCK_M + i, label_idx] = target_logits_local[i, label_idx]

    return linear_cross_entropy_fwd_mainloop_kernel


@tilelang.jit
def linear_cross_entropy_fwd_local_reduce(
    num_splits: int,
    BLOCK_M=128,
    num_threads=128,
):
    """Merge the vocabulary-split statistics produced on one rank."""
    M = T.dynamic("M")

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
            split_max = T.alloc_fragment([BLOCK_M], T.float32)
            split_sum = T.alloc_fragment([BLOCK_M], T.float32)
            split_weighted_sum = T.alloc_fragment([BLOCK_M], T.float32)
            new_max = T.alloc_fragment([BLOCK_M], T.float32)

            T.fill(rank_max, -T.infinity(T.float32))
            T.clear(rank_sum)
            T.clear(rank_weighted_sum)

            for split in T.Serial(num_splits):
                for i in T.Parallel(BLOCK_M):
                    is_valid = bx * BLOCK_M + i < M
                    split_max[i] = T.if_then_else(
                        is_valid,
                        partial_max[bx * BLOCK_M + i, split],
                        -T.infinity(T.float32),
                    )
                    split_sum[i] = T.if_then_else(
                        is_valid,
                        partial_sum[bx * BLOCK_M + i, split],
                        0.0,
                    )
                    split_weighted_sum[i] = T.if_then_else(
                        is_valid,
                        partial_weighted_sum[bx * BLOCK_M + i, split],
                        0.0,
                    )
                    new_max[i] = T.max(rank_max[i], split_max[i])

                for i in T.Parallel(BLOCK_M):
                    rank_rescale = T.exp(rank_max[i] - new_max[i])
                    split_rescale = T.exp(split_max[i] - new_max[i])
                    rank_sum[i] = rank_sum[i] * rank_rescale + split_sum[i] * split_rescale
                    rank_weighted_sum[i] = rank_weighted_sum[i] * rank_rescale + split_weighted_sum[i] * split_rescale
                    rank_max[i] = new_max[i]

            for i in T.Parallel(BLOCK_M):
                if bx * BLOCK_M + i < M:
                    row = bx * BLOCK_M + i
                    local_max[row] = rank_max[i]
                    local_sum[row] = rank_sum[i]
                    local_weighted_sum[row] = rank_weighted_sum[i]

    return linear_cross_entropy_fwd_local_reduce_kernel


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
            hidden_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            weight_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            probabilities = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
            target_grad = T.alloc_fragment([BLOCK_M, BLOCK_N], T.float32)
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
                target_grad[i, j] = 0.0

            if HAS_LOGPROBS_GRAD:
                for label_idx in T.unroll(topk):
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        if labels_local[i, label_idx] == vocab_start + local_vocab_offset + j:
                            target_grad[i, j] += grad_logprobs_local[i, label_idx]

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                local_vocab_id = local_vocab_offset + j
                chunk_vocab_id = by * BLOCK_N + j
                is_valid = (row < M) & (local_vocab_id < N)
                grad = target_grad[i, j] - grad_logprobs_sum[i] * probabilities[i, j]
                if HAS_ENTROPY_GRAD:
                    grad -= grad_entropy_local[i] * probabilities[i, j] * (logits[i, j] - mean_logits_local[i])
                if row < M:
                    dlogits[row, chunk_vocab_id] = T.if_then_else(
                        is_valid,
                        T.cast(grad * inv_temperature, dtype),
                        0,
                    )

    return linear_cross_entropy_bwd_dlogits_kernel


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
        dist_process_group: torch.distributed.ProcessGroup = None,
    ):
        """Forward pass.

        Args:
            hidden: hidden states of shape (..., h).
            weight: lm head weights of shape (vocab_size/tp_size, h).
            labels: target tokens of shape (...) or (..., topk).
            temperature: temperature for logits scaling.
            dist_process_group: ColumnParallelLinear process group, megatron only.

        Returns:
            logprobs: logprobs tensor matching labels.
            entropy: entropy tensor matching hidden.shape[:-1].
        """
        if hidden.ndim < 2 or weight.ndim != 2:
            raise ValueError("hidden and weight must have shapes [..., K] and [N, K]")

        original_hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if labels.ndim == len(original_hidden_shape) - 1 and labels.numel() == hidden.shape[0]:
            single_label = True
            output_shape = labels.shape
            labels = labels.reshape(-1, 1)
        elif labels.ndim == len(original_hidden_shape) and labels[..., 0].numel() == hidden.shape[0]:
            single_label = False
            output_shape = labels.shape[:-1]
            labels = labels.reshape(-1, labels.shape[-1])
        else:
            raise ValueError("labels must contain one or more labels for every hidden row")

        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden and weight must have the same hidden size")
        if hidden.dtype != weight.dtype:
            raise ValueError("hidden and weight must have the same dtype")
        if hidden.device.type != get_device_name() or weight.device != hidden.device or labels.device != hidden.device:
            raise ValueError("hidden, weight, and labels must be on the same accelerator device")
        if labels.dtype != torch.int64:
            raise ValueError("labels must have dtype torch.int64")
        if hidden.shape[0] == 0 or weight.shape[0] == 0 or labels.shape[1] == 0:
            raise ValueError("M, N, and topk must be positive")
        if not isinstance(temperature, float) or temperature <= 0.0:
            raise ValueError("temperature must be a positive float")

        hidden = hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.contiguous()

        tp_rank = 0 if dist_process_group is None else torch.distributed.get_rank(dist_process_group)
        tp_world_size = 1 if dist_process_group is None else torch.distributed.get_world_size(dist_process_group)
        num_tokens, hidden_size = hidden.shape
        local_vocab_size = weight.shape[0]
        vocab_start = tp_rank * local_vocab_size
        topk = labels.shape[1]
        vocab_per_cta = 512
        num_splits = (local_vocab_size + vocab_per_cta - 1) // vocab_per_cta

        partial_shape = (num_tokens, num_splits)
        partial_max = torch.empty(partial_shape, dtype=torch.float32, device=hidden.device)
        partial_sum = torch.empty_like(partial_max)
        partial_weighted_sum = torch.empty_like(partial_max)
        target_logits = torch.zeros((num_tokens, topk), dtype=torch.float32, device=hidden.device)

        mainloop = linear_cross_entropy_fwd_mainloop(
            local_vocab_size,
            hidden_size,
            topk,
            dtype=hidden.dtype,
            VOCAB_PER_CTA=vocab_per_cta,
        )
        mainloop(
            hidden,
            weight,
            labels,
            1.0 / temperature,
            vocab_start,
            partial_max,
            partial_sum,
            partial_weighted_sum,
            target_logits,
        )

        local_max = torch.empty((num_tokens,), dtype=torch.float32, device=hidden.device)
        local_sum = torch.empty_like(local_max)
        local_weighted_sum = torch.empty_like(local_max)
        local_reduce = linear_cross_entropy_fwd_local_reduce(num_splits)
        local_reduce(
            partial_max,
            partial_sum,
            partial_weighted_sum,
            local_max,
            local_sum,
            local_weighted_sum,
        )

        global_max = local_max.clone()
        if tp_world_size > 1:
            torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=dist_process_group)

        reduced = torch.empty((num_tokens, topk + 2), dtype=torch.float32, device=hidden.device)
        prepare_reduce = linear_cross_entropy_fwd_prepare_reduce(topk)
        prepare_reduce(
            local_max,
            global_max,
            local_sum,
            local_weighted_sum,
            target_logits,
            reduced,
        )
        if tp_world_size > 1:
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM, group=dist_process_group)

        logprobs = torch.empty_like(target_logits)
        entropy = torch.empty((num_tokens,), dtype=torch.float32, device=hidden.device)
        lse = torch.empty_like(entropy)
        mean_logits = torch.empty_like(entropy)
        finalize = linear_cross_entropy_fwd_finalize(topk)
        finalize(
            global_max,
            reduced,
            logprobs,
            entropy,
            lse,
            mean_logits,
        )

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, weight, labels, lse, mean_logits)
        ctx.temperature = temperature
        ctx.dist_process_group = dist_process_group
        ctx.original_hidden_shape = original_hidden_shape
        ctx.output_shape = output_shape
        ctx.single_label = single_label
        logprobs = logprobs.reshape(*output_shape, topk)
        if single_label:
            logprobs = logprobs.squeeze(-1)
        return logprobs, entropy.reshape(*output_shape)

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor):
        """Backward pass.

        Args:
            grad_logprobs: gradient matching the forward logprobs, or None.
            grad_entropy: gradient matching hidden.shape[:-1], or None if entropy loss is not used.

        Returns:
            grad_hidden: gradient matching the forward hidden shape.
            grad_weight: gradient of the lm head weights tensor of shape (vocab_size/tp_size, h)
        """
        hidden, weight, labels, lse, mean_logits = ctx.saved_tensors
        temperature = ctx.temperature
        dist_process_group = ctx.dist_process_group

        num_tokens, hidden_size = hidden.shape
        local_vocab_size = weight.shape[0]
        topk = labels.shape[1]
        tp_rank = 0 if dist_process_group is None else torch.distributed.get_rank(dist_process_group)
        vocab_start = tp_rank * local_vocab_size

        has_logprobs_grad = grad_logprobs is not None
        has_entropy_grad = grad_entropy is not None
        if has_logprobs_grad:
            expected_shape = ctx.output_shape if ctx.single_label else (*ctx.output_shape, topk)
            if grad_logprobs.shape != expected_shape:
                raise ValueError(f"grad_logprobs must have shape {expected_shape}")
            grad_logprobs = grad_logprobs.reshape(num_tokens, topk).float().contiguous()
        else:
            grad_logprobs = torch.empty((num_tokens, topk), dtype=torch.float32, device=hidden.device)
        if has_entropy_grad:
            if grad_entropy.shape != ctx.output_shape:
                raise ValueError(f"grad_entropy must have shape {ctx.output_shape}")
            grad_entropy = grad_entropy.reshape(num_tokens).float().contiguous()
        else:
            # The no-entropy specialization does not read this argument.
            grad_entropy = lse

        block_n = 128
        max_vocab_chunk = 8192
        aligned_local_vocab_size = (local_vocab_size + block_n - 1) // block_n * block_n
        vocab_chunk = min(max_vocab_chunk, aligned_local_vocab_size)
        dlogits = torch.empty(
            (num_tokens, vocab_chunk),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        grad_hidden_accum = torch.zeros_like(hidden, dtype=torch.float32)
        grad_weight = torch.empty_like(weight)

        dlogits_kernel = linear_cross_entropy_bwd_dlogits(
            local_vocab_size,
            hidden_size,
            topk,
            vocab_chunk,
            has_logprobs_grad,
            has_entropy_grad,
            dtype=hidden.dtype,
            BLOCK_N=block_n,
        )
        for vocab_offset in range(0, local_vocab_size, vocab_chunk):
            dlogits_kernel(
                hidden,
                weight,
                labels,
                lse,
                mean_logits,
                grad_logprobs,
                grad_entropy,
                1.0 / temperature,
                vocab_start,
                vocab_offset,
                dlogits,
            )

            valid_vocab = min(vocab_chunk, local_vocab_size - vocab_offset)
            dlogits_valid = dlogits
            if valid_vocab != vocab_chunk:
                dlogits_valid = dlogits[:, :valid_vocab].contiguous()
            weight_chunk = weight[vocab_offset : vocab_offset + valid_vocab]

            grad_hidden_accum.add_(torch.mm(dlogits_valid, weight_chunk, out_dtype=torch.float32))
            torch.mm(
                dlogits_valid.T,
                hidden,
                out=grad_weight[vocab_offset : vocab_offset + valid_vocab],
            )

        grad_hidden = grad_hidden_accum.to(hidden.dtype).reshape(ctx.original_hidden_shape)
        return grad_hidden, grad_weight, None, None, None


linear_cross_entropy = LinearCrossEntropy.apply
