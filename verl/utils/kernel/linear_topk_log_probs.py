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

"""Memory-efficient linear selected-log-probability operator.

The CUDA path uses Triton vocabulary tiles and recomputes logits during
backward. The portable fallback chunks the token dimension. Neither path
retains a full ``[num_tokens, vocab_size]`` tensor in autograd.
"""

from typing import Optional

import torch
import torch.distributed as dist

_MAX_LOGITS_CHUNK_BYTES = 64 * 1024 * 1024


def _world_info(process_group: Optional[dist.ProcessGroup]) -> tuple[int, int]:
    if process_group is None:
        return 0, 1
    return dist.get_rank(process_group), dist.get_world_size(process_group)


def _resolve_chunk_size(num_tokens: int, local_vocab_size: int, chunk_size: Optional[int]) -> int:
    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return min(num_tokens, chunk_size)
    # Logits and softmax statistics are computed in FP32. Bound the principal
    # temporary to 64 MiB; backward may hold logits and their gradient together.
    rows = _MAX_LOGITS_CHUNK_BYTES // (local_vocab_size * torch.float32.itemsize)
    return max(1, min(num_tokens, rows))


def _linear_fp32(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``hidden @ weight.T`` with FP32 output where CUDA supports it."""
    if hidden.dtype == torch.float32:
        return torch.mm(hidden, weight.T)
    if hidden.is_cuda:
        try:
            return torch.mm(hidden, weight.T, out_dtype=torch.float32)
        except TypeError:
            # Older torch versions do not expose CUDA GEMM's FP32 output.
            return torch.mm(hidden, weight.T).float()
    return torch.mm(hidden.float(), weight.float().T)


def _validate_topk_ids(topk_ids: torch.Tensor, global_vocab_size: int) -> None:
    """Validate global vocabulary ids without synchronizing CUDA with the host.

    ``Tensor.item()`` on a CUDA reduction stalls Python until all preceding GPU
    work completes.  ``torch._assert_async`` keeps the check ordered on the CUDA
    stream and reports an invalid id asynchronously.  CPU execution retains the
    eager ``ValueError`` used by the portable fallback and unit tests.
    """
    valid = torch.all((topk_ids >= 0) & (topk_ids < global_vocab_size))
    message = f"topk_ids must be in [0, {global_vocab_size})"
    if topk_ids.is_cuda:
        assert_async = getattr(torch, "_assert_async", None)
        if assert_async is not None:
            assert_async(valid, message)
        # Older torch versions without an asynchronous device assertion skip
        # this optional CUDA validation rather than introduce a host sync.
        return
    if not bool(valid):
        raise ValueError(message)


class LinearTopKLogProbs(torch.autograd.Function):
    """Autograd implementation that saves log-sum-exp instead of full logits."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        topk_ids: torch.Tensor,
        temperature: float = 1.0,
        dist_process_group: Optional[dist.ProcessGroup] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if not isinstance(temperature, float):
            raise TypeError(f"temperature must be a float, got {type(temperature)}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if hidden.dim() < 2 or weight.dim() != 2 or topk_ids.dim() < 2:
            raise ValueError(
                "expected hidden (..., hidden_size), weight (local_vocab_size, hidden_size), and topk_ids (..., topk)"
            )
        if hidden.numel() == 0:
            raise ValueError("hidden must contain at least one token and one hidden element")
        if weight.shape[0] == 0:
            raise ValueError("weight must contain at least one vocabulary row")
        if not hidden.dtype.is_floating_point or not weight.dtype.is_floating_point:
            raise TypeError(f"hidden and weight must be floating-point tensors, got {hidden.dtype} and {weight.dtype}")
        if topk_ids.shape[-1] == 0:
            raise ValueError("topk_ids must contain at least one selected token per row")
        if hidden.shape[-1] != weight.shape[-1]:
            raise ValueError(f"hidden size mismatch: hidden={hidden.shape[-1]}, weight={weight.shape[-1]}")
        if hidden.numel() // hidden.shape[-1] != topk_ids.numel() // topk_ids.shape[-1]:
            raise ValueError(f"token count mismatch between hidden {hidden.shape} and topk_ids {topk_ids.shape}")
        if hidden.device != weight.device or hidden.device != topk_ids.device:
            raise ValueError("hidden, weight, and topk_ids must be on the same device")
        if hidden.dtype != weight.dtype:
            raise ValueError(f"hidden and weight must have the same dtype, got {hidden.dtype} and {weight.dtype}")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"topk_ids must use int32 or int64, got {topk_ids.dtype}")

        original_hidden_shape = hidden.shape
        original_topk_shape = topk_ids.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        weight_2d = weight.contiguous()
        topk_ids_2d = topk_ids.reshape(hidden_2d.shape[0], -1).to(torch.long).contiguous()

        num_tokens = hidden_2d.shape[0]
        local_vocab_size = weight_2d.shape[0]
        rank, world_size = _world_info(dist_process_group)
        global_vocab_size = local_vocab_size * world_size
        _validate_topk_ids(topk_ids_2d, global_vocab_size)

        rows_per_chunk = _resolve_chunk_size(num_tokens, local_vocab_size, chunk_size)
        local_max = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)
        local_sum_exp = torch.empty_like(local_max)
        selected_logits = torch.zeros(topk_ids_2d.shape, device=hidden.device, dtype=torch.float32)
        vocab_start = rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size
        inv_temperature = 1.0 / temperature

        from . import topk_log_probs_kernels

        use_triton = topk_log_probs_kernels.can_use_triton(hidden_2d, topk_ids_2d.shape[1])
        if use_triton:
            local_max, sum_payload = topk_log_probs_kernels.topk_log_probs_forward(
                hidden_2d,
                weight_2d,
                topk_ids_2d,
                temperature,
                rank,
            )
            local_sum_exp = sum_payload[:, 0]
            selected_logits = sum_payload[:, 1:]
        else:
            for start in range(0, num_tokens, rows_per_chunk):
                end = min(start + rows_per_chunk, num_tokens)
                logits = _linear_fp32(hidden_2d[start:end], weight_2d)
                logits.mul_(inv_temperature)

                chunk_max = logits.max(dim=-1).values
                local_max[start:end] = chunk_max
                local_sum_exp[start:end] = torch.exp(logits - chunk_max.unsqueeze(-1)).sum(dim=-1)

                ids = topk_ids_2d[start:end]
                owned = (ids >= vocab_start) & (ids < vocab_end)
                local_ids = (ids - vocab_start).masked_fill(~owned, 0)
                selected = torch.gather(logits, dim=-1, index=local_ids)
                selected_logits[start:end] = selected.masked_fill(~owned, 0)

        global_max = local_max.clone()
        if world_size > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=dist_process_group)

        # Each rank's sum was normalized by its local maximum. Rescale it to
        # the global maximum before summing across vocabulary shards.
        local_sum_exp.mul_(torch.exp(local_max - global_max))
        if world_size > 1:
            if not use_triton:
                # The Triton path writes these values into one contiguous
                # payload from the start. The fallback is uncommon in OPD and
                # packs here to preserve the same two-collective TP contract.
                sum_payload = torch.cat((local_sum_exp.unsqueeze(-1), selected_logits), dim=-1)
            dist.all_reduce(sum_payload, op=dist.ReduceOp.SUM, group=dist_process_group)
            global_sum_exp = sum_payload[:, 0]
            selected_logits = sum_payload[:, 1:]
        else:
            global_sum_exp = local_sum_exp

        global_logsumexp = global_max + global_sum_exp.log()
        output = selected_logits - global_logsumexp.unsqueeze(-1)

        ctx.save_for_backward(hidden_2d, weight_2d, topk_ids_2d, global_logsumexp)
        ctx.original_hidden_shape = original_hidden_shape
        ctx.original_topk_shape = original_topk_shape
        ctx.temperature = temperature
        ctx.rank = rank
        ctx.rows_per_chunk = rows_per_chunk
        ctx.use_triton = use_triton
        return output.view(original_topk_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, topk_ids, global_logsumexp = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(topk_ids.shape).float().contiguous()
        num_tokens, hidden_size = hidden.shape
        local_vocab_size = weight.shape[0]
        vocab_start = ctx.rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size
        inv_temperature = 1.0 / ctx.temperature

        if ctx.use_triton:
            from .topk_log_probs_kernels import topk_log_probs_backward

            grad_hidden, grad_weight = topk_log_probs_backward(
                hidden,
                weight,
                topk_ids,
                global_logsumexp,
                grad_output_2d,
                ctx.temperature,
                ctx.rank,
            )
            return grad_hidden.view(ctx.original_hidden_shape), grad_weight, None, None, None, None

        grad_hidden = torch.zeros((num_tokens, hidden_size), device=hidden.device, dtype=hidden.dtype)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, num_tokens, ctx.rows_per_chunk):
            end = min(start + ctx.rows_per_chunk, num_tokens)
            hidden_chunk = hidden[start:end]
            ids = topk_ids[start:end]
            upstream = grad_output_2d[start:end]

            logits = _linear_fp32(hidden_chunk, weight)
            logits.mul_(inv_temperature)
            probabilities = torch.exp(logits - global_logsumexp[start:end].unsqueeze(-1))

            # For y_i = z_i - logsumexp(z), accumulate all selected numerator
            # gradients and subtract softmax(z) times their row-wise sum.
            grad_logits = -probabilities * upstream.sum(dim=-1, keepdim=True)
            owned = (ids >= vocab_start) & (ids < vocab_end)
            local_ids = (ids - vocab_start).masked_fill(~owned, 0)
            grad_logits.scatter_add_(dim=-1, index=local_ids, src=upstream * owned)
            grad_logits.mul_(inv_temperature)

            grad_logits_model_dtype = grad_logits.to(hidden.dtype)
            grad_hidden[start:end] = torch.mm(grad_logits_model_dtype, weight)
            grad_weight.addmm_(grad_logits_model_dtype.T, hidden_chunk)

        return grad_hidden.view(ctx.original_hidden_shape), grad_weight, None, None, None, None


def linear_topk_log_probs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    temperature: float = 1.0,
    dist_process_group: Optional[dist.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Return globally normalized student log probabilities at ``topk_ids``.

    Args:
        hidden: Hidden states with shape ``(..., hidden_size)``.
        weight: This tensor-parallel rank's output-weight shard with shape
            ``(vocab_size_per_partition, hidden_size)``.
        topk_ids: Global-vocabulary token ids with shape ``(..., topk)``.
        temperature: Scalar temperature applied to logits.
        dist_process_group: Tensor-parallel process group, or ``None`` for a
            non-sharded vocabulary.
        chunk_size: Optional number of token rows used by the non-Triton
            fallback. By default its principal FP32 logits temporary is
            limited to 64 MiB.

    Returns:
        A float32 tensor with the same shape as ``topk_ids``.
    """
    return LinearTopKLogProbs.apply(
        hidden,
        weight,
        topk_ids,
        temperature,
        dist_process_group,
        chunk_size,
    )
