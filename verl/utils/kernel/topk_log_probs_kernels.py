# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernels for selected log probabilities over a linear vocabulary head."""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


_BLOCK_N = 128
_MAX_FORWARD_SPLITS = 128
_TARGET_BACKWARD_DLOGITS_BYTES = 32 * 1024 * 1024
_MAX_BACKWARD_DLOGITS_BYTES = 40 * 1024 * 1024
_MAX_TRITON_TOPK = 128


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up_to_block(value: int) -> int:
    return _ceil_div(value, _BLOCK_N) * _BLOCK_N


def _limit_split_to_vocab(vocab_per_split: int, vocab_size: int) -> int:
    """Avoid iterating far past the final vocabulary tile."""
    padded_vocab_size = max(_BLOCK_N, _round_up_to_block(vocab_size))
    return min(vocab_per_split, padded_vocab_size)


def _select_forward_vocab_per_split(num_tokens: int, vocab_size: int) -> int:
    """Choose forward split width from B300 token-parallelism measurements."""
    if num_tokens <= 32:
        vocab_per_split = 256
    elif num_tokens <= 1024:
        vocab_per_split = 512
    else:
        vocab_per_split = 1024
    # Bound the second-stage reduction width for unusually large local
    # vocabularies. This keeps BLOCK_SPLITS at 128 or below.
    min_for_reduction = _round_up_to_block(_ceil_div(vocab_size, _MAX_FORWARD_SPLITS))
    vocab_per_split = max(vocab_per_split, min_for_reduction)
    return _limit_split_to_vocab(vocab_per_split, vocab_size)


def _select_backward_vocab_per_split(num_tokens: int, vocab_size: int, element_size: int) -> int:
    """Reduce sequential launches while bounding the materialized dlogits tile."""
    full_dlogits_bytes = num_tokens * vocab_size * element_size
    if full_dlogits_bytes <= _MAX_BACKWARD_DLOGITS_BYTES:
        return vocab_size

    max_by_memory = _TARGET_BACKWARD_DLOGITS_BYTES // (num_tokens * element_size)
    max_by_memory = max(_BLOCK_N, max_by_memory // _BLOCK_N * _BLOCK_N)
    return min(max_by_memory, vocab_size)


if HAVE_TRITON:

    @triton.jit
    def _linear_stats_kernel(
        hidden_ptr,
        weight_ptr,
        topk_ids_ptr,
        selected_logits_ptr,
        num_tokens,
        hidden_size,
        vocab_size,
        num_splits,
        vocab_start,
        stride_hidden_m: tl.int64,
        stride_hidden_k: tl.constexpr,
        stride_weight_n: tl.int64,
        stride_weight_k: tl.constexpr,
        stride_ids_m: tl.int64,
        stride_ids_k: tl.int64,
        stride_selected_m: tl.int64,
        stride_selected_k: tl.int64,
        split_max_ptr,
        split_sum_ptr,
        rcp_temperature: tl.float32,
        TOPK: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        VOCAB_PER_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute stable softmax statistics without storing the logits matrix.

        Specialize H-axis strides to expose contiguous loads to Triton's
        layout analysis and asynchronous load pipeline. Row strides remain
        runtime values; no extra pointer-alignment assumptions are required.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(num_tokens, BLOCK_M)
        pid_m = pid % num_pid_m
        split_idx = pid // num_pid_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        split_start = split_idx * VOCAB_PER_SPLIT
        split_end = tl.minimum(split_start + VOCAB_PER_SPLIT, vocab_size)

        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)

        for n_offset in range(0, VOCAB_PER_SPLIT, BLOCK_N):
            offs_n = split_start + n_offset + tl.arange(0, BLOCK_N)
            hidden_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
            weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
            logits = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

            for k in range(0, tl.cdiv(hidden_size, BLOCK_K)):
                hidden_values = tl.load(
                    hidden_ptrs,
                    mask=(offs_m[:, None] < num_tokens) & (offs_k[None, :] < hidden_size - k * BLOCK_K),
                    other=0.0,
                )
                weight_values = tl.load(
                    weight_ptrs,
                    mask=(offs_n[:, None] < split_end) & (offs_k[None, :] < hidden_size - k * BLOCK_K),
                    other=0.0,
                )
                # IEEE input precision is required for FP32 model weights.  It
                # also keeps this path aligned with torch.mm when TP tests use
                # FP32 inputs; BF16 operands still use FP32 accumulation.
                logits = tl.dot(hidden_values, weight_values.T, logits, input_precision="ieee")
                hidden_ptrs += BLOCK_K * stride_hidden_k
                weight_ptrs += BLOCK_K * stride_weight_k

            logits *= rcp_temperature

            # Extract selected logits from the exact GEMM tile used by the
            # logsumexp calculation.  A separate row-wise dot product uses a
            # different reduction order and can drift noticeably for BF16.
            # Exactly one vocabulary tile owns each valid global token id, so
            # direct stores are race-free.  Other TP ranks retain their zero.
            tile_global_start = vocab_start + split_start + n_offset
            tile_global_end = vocab_start + tl.minimum(split_start + n_offset + BLOCK_N, split_end)
            valid_rows = offs_m < num_tokens
            offs_topk = tl.arange(0, BLOCK_TOPK)
            for topk_offset in range(0, TOPK, BLOCK_TOPK):
                topk_indices = topk_offset + offs_topk
                valid_selected = valid_rows[:, None] & (topk_indices[None, :] < TOPK)
                selected_ids = tl.load(
                    topk_ids_ptr + offs_m[:, None] * stride_ids_m + topk_indices[None, :] * stride_ids_k,
                    mask=valid_selected,
                    other=-1,
                )
                owned_by_tile = valid_selected & (selected_ids >= tile_global_start) & (selected_ids < tile_global_end)
                # Convert global ids to columns of the current register tile.
                # Non-owned entries use column zero only to keep gather indices
                # in range; their stores remain masked out below.
                selected_columns = tl.where(
                    owned_by_tile,
                    selected_ids - tile_global_start,
                    0,
                ).to(tl.int32)
                selected = tl.gather(logits, selected_columns, axis=1)
                tl.store(
                    selected_logits_ptr
                    + offs_m[:, None] * stride_selected_m
                    + topk_indices[None, :] * stride_selected_k,
                    selected,
                    mask=owned_by_tile,
                )

            logits = tl.where(offs_n[None, :] < split_end, logits, -float("inf"))

            block_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.exp(logits - new_max[:, None]), axis=1
            )
            running_max = new_max

        stats_offset = offs_m * num_splits + split_idx
        valid_rows = offs_m < num_tokens
        tl.store(split_max_ptr + stats_offset, running_max, mask=valid_rows)
        tl.store(split_sum_ptr + stats_offset, running_sum, mask=valid_rows)

    @triton.jit
    def _reduce_stats_kernel(
        split_max_ptr,
        split_sum_ptr,
        local_max_ptr,
        local_sum_ptr,
        num_tokens,
        num_splits,
        stride_local_sum_m: tl.int64,
        BLOCK_M: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_s = tl.arange(0, BLOCK_SPLITS)
        mask = (offs_m[:, None] < num_tokens) & (offs_s[None, :] < num_splits)
        offsets = offs_m[:, None] * num_splits + offs_s[None, :]
        split_max = tl.load(split_max_ptr + offsets, mask=mask, other=-float("inf"))
        split_sum = tl.load(split_sum_ptr + offsets, mask=mask, other=0.0)
        local_max = tl.max(split_max, axis=1)
        local_sum = tl.sum(split_sum * tl.exp(split_max - local_max[:, None]), axis=1)
        valid_rows = offs_m < num_tokens
        tl.store(local_max_ptr + offs_m, local_max, mask=valid_rows)
        tl.store(local_sum_ptr + offs_m * stride_local_sum_m, local_sum, mask=valid_rows)

    @triton.jit
    def _topk_dlogits_split_kernel(
        hidden_ptr,
        weight_ptr,
        topk_ids_ptr,
        grad_output_ptr,
        global_lse_ptr,
        dlogits_ptr,
        num_tokens,
        hidden_size,
        vocab_size,
        vocab_start,
        stride_hidden_m: tl.int64,
        stride_hidden_k: tl.int64,
        stride_weight_n: tl.int64,
        stride_weight_k: tl.int64,
        stride_ids_m: tl.int64,
        stride_ids_k: tl.int64,
        stride_grad_m: tl.int64,
        stride_grad_k: tl.int64,
        stride_dlogits_m: tl.int64,
        stride_dlogits_n: tl.int64,
        rcp_temperature: tl.float32,
        SPLIT_IDX: tl.constexpr,
        TOPK: tl.constexpr,
        VOCAB_PER_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(num_tokens, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        result_offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = SPLIT_IDX * VOCAB_PER_SPLIT + result_offs_n
        offs_k = tl.arange(0, BLOCK_K)
        split_end = tl.minimum((SPLIT_IDX + 1) * VOCAB_PER_SPLIT, vocab_size)

        hidden_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
        logits = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            hidden_values = tl.load(
                hidden_ptrs,
                mask=(offs_m[:, None] < num_tokens) & (offs_k[None, :] < hidden_size - k * BLOCK_K),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptrs,
                mask=(offs_n[:, None] < split_end) & (offs_k[None, :] < hidden_size - k * BLOCK_K),
                other=0.0,
            )
            logits = tl.dot(hidden_values, weight_values.T, logits, input_precision="ieee")
            hidden_ptrs += BLOCK_K * stride_hidden_k
            weight_ptrs += BLOCK_K * stride_weight_k

        logits *= rcp_temperature
        global_lse = tl.load(global_lse_ptr + offs_m, mask=offs_m < num_tokens, other=0.0)
        grad_sum = tl.zeros((BLOCK_M,), tl.float32)
        numerator_grad = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        global_vocab_ids = vocab_start + offs_n
        for k in range(0, TOPK):
            ids = tl.load(
                topk_ids_ptr + offs_m * stride_ids_m + k * stride_ids_k,
                mask=offs_m < num_tokens,
                other=-1,
            )
            grad = tl.load(
                grad_output_ptr + offs_m * stride_grad_m + k * stride_grad_k,
                mask=offs_m < num_tokens,
                other=0.0,
            )
            grad_sum += grad
            numerator_grad += tl.where(global_vocab_ids[None, :] == ids[:, None], grad[:, None], 0.0)

        probabilities = tl.exp(logits - global_lse[:, None])
        dlogits = (numerator_grad - probabilities * grad_sum[:, None]) * rcp_temperature
        valid = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < split_end)
        tl.store(
            dlogits_ptr + offs_m[:, None] * stride_dlogits_m + result_offs_n[None, :] * stride_dlogits_n,
            dlogits,
            mask=valid,
        )


def can_use_triton(hidden: torch.Tensor, topk: int) -> bool:
    return HAVE_TRITON and hidden.is_cuda and 0 < topk <= _MAX_TRITON_TOPK


def topk_log_probs_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    temperature: float,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local max and a packed ``[exp-sum, selected logits]`` buffer."""
    if not can_use_triton(hidden, topk_ids.shape[1]):
        raise RuntimeError("Triton selected-log-probability kernel is unavailable for these inputs")

    num_tokens, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    topk = topk_ids.shape[1]
    vocab_per_split = _select_forward_vocab_per_split(num_tokens, vocab_size)
    num_splits = triton.cdiv(vocab_size, vocab_per_split)
    split_max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    split_sum = torch.empty_like(split_max)
    # Column zero is reserved for local_sum. Keeping it adjacent to selected
    # logits lets the caller reduce all SUM-valued TP data in one collective.
    sum_payload = torch.zeros(
        (num_tokens, topk + 1),
        device=hidden.device,
        dtype=torch.float32,
    )
    selected_logits = sum_payload[:, 1:]

    grid = (triton.cdiv(num_tokens, 32) * num_splits,)
    _linear_stats_kernel[grid](
        hidden,
        weight,
        topk_ids,
        selected_logits,
        num_tokens,
        hidden_size,
        vocab_size,
        num_splits,
        rank * vocab_size,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        selected_logits.stride(0),
        selected_logits.stride(1),
        split_max,
        split_sum,
        1.0 / temperature,
        TOPK=topk,
        BLOCK_TOPK=16,
        VOCAB_PER_SPLIT=vocab_per_split,
        BLOCK_M=32,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=3,
    )

    local_max = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)
    local_sum = sum_payload[:, 0]
    reduce_grid = (triton.cdiv(num_tokens, 32),)
    _reduce_stats_kernel[reduce_grid](
        split_max,
        split_sum,
        local_max,
        local_sum,
        num_tokens,
        num_splits,
        local_sum.stride(0),
        BLOCK_M=32,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=4,
    )

    return local_max, sum_payload


def topk_log_probs_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    global_lse: torch.Tensor,
    grad_output: torch.Tensor,
    temperature: float,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute logits by vocab split and accumulate linear-input gradients."""
    num_tokens, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    topk = topk_ids.shape[1]
    vocab_per_split = _select_backward_vocab_per_split(num_tokens, vocab_size, hidden.element_size())
    num_splits = triton.cdiv(vocab_size, vocab_per_split)
    dlogits = torch.empty(
        (num_tokens, vocab_per_split),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    grad_hidden = torch.empty_like(hidden)
    grad_weight = torch.empty_like(weight)

    for split_idx in range(num_splits):
        split_width = min(vocab_per_split, vocab_size - split_idx * vocab_per_split)
        grid = (triton.cdiv(num_tokens, 32) * triton.cdiv(split_width, 128),)
        _topk_dlogits_split_kernel[grid](
            hidden,
            weight,
            topk_ids,
            grad_output,
            global_lse,
            dlogits,
            num_tokens,
            hidden_size,
            vocab_size,
            rank * vocab_size,
            hidden.stride(0),
            hidden.stride(1),
            weight.stride(0),
            weight.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            dlogits.stride(0),
            dlogits.stride(1),
            1.0 / temperature,
            SPLIT_IDX=split_idx,
            TOPK=topk,
            VOCAB_PER_SPLIT=vocab_per_split,
            BLOCK_M=32,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=3,
        )
        dlogits_view = dlogits[:, :split_width]
        if split_width != vocab_per_split:
            dlogits_view = dlogits_view.contiguous()
        weight_start = split_idx * vocab_per_split
        weight_end = weight_start + split_width
        if split_idx == 0:
            torch.mm(dlogits_view, weight[weight_start:weight_end], out=grad_hidden)
        else:
            grad_hidden.addmm_(dlogits_view, weight[weight_start:weight_end])
        torch.mm(dlogits_view.T, hidden, out=grad_weight[weight_start:weight_end])

    return grad_hidden, grad_weight
