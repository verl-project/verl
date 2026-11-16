# Copyright 2026 Individual Contributor
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
"""Top-k Rao-Blackwellised KL penalty against the reference policy (``kl_loss_type: topk_rb``).

Per position, with policy ``q``, reference ``p``, reference top-k ids ``T`` and tail masses ``q_t = q(not T)``,
``p_t = p(not T)``:

    KL(q || p) = sum_{v in T} q_v log(q_v / p_v) + q_t log(q_t / p_t)    # coarse: exact
               + q_t * KL(q|tail || p|tail)                              # tail: sampled

For the sampled token ``y`` with ``r = log(q_y / q_t) - log(p_y / p_t)``, the surrogate
``1[y not in T] * (sg(r) * log q_t + r^2 / 2)`` has the expected gradient of the tail term when ``y ~ q``;
detached importance weights keep it unbiased when ``y`` came from another policy. The estimator is unbiased
for the gradient of the full KL at every ``k`` and exact for ``T`` = the whole vocabulary. Tail masses are
log-sum-exps over the complement of ``T``, never ``1 - sum(top-k)``, so they stay exact when the top-k holds
nearly all of the mass. See docs/algo/topk_rb_kl.md and arXiv:2504.10637.

Logits may be vocab-parallel: as in :func:`~verl.trainer.ppo.score_centering.topk_log_probs_from_logits`,
``process_group`` is the group they are sharded over (rank ``r`` holds ids ``[r * V, (r + 1) * V)``), or None.
"""

from typing import Optional

import torch

from verl.trainer.ppo.score_centering import (
    TOPK_LOG_PROB_CHUNK_SIZE,
    _all_reduce,
    _local_ids,
    _vocab_start,
    topk_log_probs_from_logits,
)

# floor for log tail masses: keeps an empty tail (k = V) finite, far below any representable probability
_LOG_FLOOR = -1e4


def _topk_columns(token_ids: torch.Tensor, vocab_start: int, vocab_size: int) -> torch.Tensor:
    """``(n, vocab_size)`` bool mask of the local columns holding one of ``token_ids``."""
    local_ids, local_mask = _local_ids(token_ids, vocab_start, vocab_size)
    columns = torch.zeros(token_ids.shape[0], vocab_size, dtype=torch.uint8, device=token_ids.device)
    # add, not overwrite: off-shard ids are clamped onto an edge column that may hold a real top-k id
    return columns.scatter_add_(-1, local_ids, local_mask.to(torch.uint8)).bool()


class _LogTailMass(torch.autograd.Function):
    """``log`` of the softmax mass outside ``token_ids``, chunked like ``_TopKLogProbsFromLogits``.

    The tail log-sum-exp is taken relative to the tail's own maximum, so it stays exact however far below the
    row maximum the tail lies; the normalizer adds the top-k terms to it, so nothing is subtracted. Only the
    logits, the ids and per-row scalars are saved; backward recomputes per chunk with
    ``d log q_t / dz_v = 1[v not in T] q_v / q_t - q_v``.
    """

    @staticmethod
    def forward(ctx, logits, token_ids, chunk_size, process_group):
        n, vocab_size = logits.shape
        vocab_start = _vocab_start(vocab_size, process_group)
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        output = torch.empty(n, device=logits.device, dtype=torch.float32)
        log_z = torch.empty_like(output)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            maximum = work.max(dim=-1).values
            _all_reduce(maximum, torch.distributed.ReduceOp.MAX, process_group)
            local_ids, local_mask = _local_ids(token_ids[start:end], vocab_start, vocab_size)
            topk_sum = ((work.gather(-1, local_ids) - maximum[:, None]).exp() * local_mask).sum(-1)
            work.masked_fill_(_topk_columns(token_ids[start:end], vocab_start, vocab_size), -torch.inf)
            tail_max = work.max(dim=-1).values
            _all_reduce(tail_max, torch.distributed.ReduceOp.MAX, process_group)
            tail_max = torch.where(tail_max > -torch.inf, tail_max, maximum)  # empty tail: any finite shift
            sums = torch.stack([topk_sum, work.sub_(tail_max[:, None]).exp_().sum(-1)])
            _all_reduce(sums, torch.distributed.ReduceOp.SUM, process_group)
            topk_sum, tail_sum = sums
            log_z[start:end] = maximum + (topk_sum + tail_sum * (tail_max - maximum).exp()).log()
            output[start:end] = (tail_max + tail_sum.log() - log_z[start:end]).clamp(min=_LOG_FLOOR)
        ctx.save_for_backward(logits, token_ids, log_z, output)
        ctx.chunk_size, ctx.vocab_start = chunk_size, vocab_start
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, token_ids, log_z, log_tail = ctx.saved_tensors
        n, vocab_size = logits.shape
        grad_input = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            log_q = logits[start:end].to(dtype=torch.float32, copy=True).sub_(log_z[start:end, None])
            grad = (log_q - log_tail[start:end, None]).exp_()  # q_v / q_t, exact for every tail token
            grad.masked_fill_(_topk_columns(token_ids[start:end], ctx.vocab_start, vocab_size), 0.0)
            # an empty (floored) tail is a constant
            scale = grad_output[start:end].masked_fill(log_tail[start:end] <= _LOG_FLOOR, 0.0)
            grad_input[start:end] = grad.sub_(log_q.exp_()).mul_(scale[:, None])
        # every rank differentiates the same replicated output, so no all-reduce here
        return grad_input, None, None, None


def compute_topk_log_probs(
    logits: torch.Tensor,
    k: int,
    chunk_size: int = TOPK_LOG_PROB_CHUNK_SIZE,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of the reference policy for the ``topk_rb`` KL loss, without gradient.

    Args:
        logits: Reference logits, shape (N, V), any float dtype, scaled as for its log-probs. With
            ``process_group``, the local vocab shard of rank r covers global ids ``[r * V, (r + 1) * V)``.
        k: Number of tokens to keep.
        chunk_size: Number of rows processed per chunk. Default: 4096.
        process_group: Vocab-parallel group the logits are sharded over, default None (full vocab).

    Returns:
        Tuple containing:
            log_probs: Top-k log-probs, shape (N, k + 1), dtype float32; the last column is the log of the
                probability mass outside the top-k.
            ids: Global top-k token ids, shape (N, k), identical on every rank of the group.
    """
    with torch.no_grad():
        values, ids = logits.topk(min(k, logits.shape[-1]), dim=-1)
        ids = ids + _vocab_start(logits.shape[-1], process_group)
        if process_group is not None:  # global top-k from every shard's candidates
            world = torch.distributed.get_world_size(process_group)
            all_values = [torch.empty_like(values) for _ in range(world)]
            all_ids = [torch.empty_like(ids) for _ in range(world)]
            torch.distributed.all_gather(all_values, values.contiguous(), group=process_group)
            torch.distributed.all_gather(all_ids, ids.contiguous(), group=process_group)
            ids = torch.cat(all_ids, -1).gather(-1, torch.cat(all_values, -1).topk(k, dim=-1).indices)
        log_probs = topk_log_probs_from_logits(logits, ids, chunk_size, process_group)
        log_tail = _LogTailMass.apply(logits, ids, chunk_size, process_group)
    return torch.cat([log_probs, log_tail.unsqueeze(-1)], -1), ids


def compute_coarse_reverse_kl(
    logits: torch.Tensor,
    ref_topk_ids: torch.Tensor,
    ref_topk_log_probs: torch.Tensor,
    chunk_size: int = TOPK_LOG_PROB_CHUNK_SIZE,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarse KL(policy || ref) over the reference top-k plus one tail bucket, and the policy's log tail mass.

    Args:
        logits: Policy logits, shape (N, V), any float dtype, scaled as for the policy log-probs. They are saved
            for backward and must not be modified in place afterwards. Vocab-parallel as in
            :func:`compute_topk_log_probs`.
        ref_topk_ids: Reference top-k token ids, shape (N, k), from :func:`compute_topk_log_probs`.
        ref_topk_log_probs: Reference top-k log-probs with the log tail mass last, shape (N, k + 1).
        chunk_size: Number of rows processed per chunk. Default: 4096.
        process_group: Vocab-parallel group the logits are sharded over, default None (full vocab).

    Returns:
        Tuple containing:
            coarse_kl: The coarse KL, shape (N,), dtype float32, with gradient to ``logits``.
            log_q_tail: Log of the policy mass outside the reference top-k, same shape, with gradient.
    """
    ref_topk_ids = ref_topk_ids.long()
    lq = topk_log_probs_from_logits(logits, ref_topk_ids, chunk_size, process_group)
    log_q_tail = _LogTailMass.apply(logits, ref_topk_ids, chunk_size, process_group)
    lp, log_p_tail = ref_topk_log_probs[:, :-1].float(), ref_topk_log_probs[:, -1].float().clamp(min=_LOG_FLOOR)
    # q_v = 0 (a masked logit) contributes 0 log 0 = 0; select rather than multiply, so no 0 * inf in either pass
    live = lq > -torch.inf
    lq = torch.where(live, lq, 0.0)
    topk_term = torch.where(live, lq.exp() * (lq - lp), 0.0).sum(-1)
    return topk_term + log_q_tail.exp() * (log_q_tail - log_p_tail), log_q_tail


def compute_topk_rb_kl(
    coarse_kl: torch.Tensor,
    log_q_tail: torch.Tensor,
    log_p_tail: torch.Tensor,
    label_in_topk: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    old_log_prob: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token loss and KL estimate for ``kl_loss_type: topk_rb``.

    Args:
        coarse_kl: Coarse KL from :func:`compute_coarse_reverse_kl`, shape (bsz, response_length).
        log_q_tail: Log policy mass outside the reference top-k, same shape.
        log_p_tail: Log reference mass outside its top-k (last column of the reference top-k), same shape.
        label_in_topk: Whether the sampled token is in the reference top-k, same shape.
        log_prob: Policy log-prob of the sampled token, same shape.
        ref_log_prob: Reference log-prob of the sampled token, same shape.
        old_log_prob: Behaviour-policy log-prob of the sampled token; weights the sampled tail term by the
            detached ``pi_theta / pi_old``, for PPO mini-batch reuse. Default None (on-policy).
        rollout_is_weights: Detached ``pi_old / pi_rollout`` weights from rollout correction, default None.

    Returns:
        Tuple containing:
            loss: Per-token surrogate whose gradient is unbiased for the gradient of KL(policy || ref).
            value: Detached per-token unbiased estimate of KL(policy || ref), for logging.
    """
    weight = (~label_in_topk.bool()).to(log_prob.dtype)
    if old_log_prob is not None:
        weight = weight * torch.exp(log_prob.detach() - old_log_prob)
    if rollout_is_weights is not None:
        weight = weight * rollout_is_weights
    r = (log_prob - log_q_tail) - (ref_log_prob - log_p_tail)
    loss = coarse_kl + weight * (r.detach() * log_q_tail + 0.5 * r.square())
    value = coarse_kl.detach() + weight * r.detach()
    return loss, value
