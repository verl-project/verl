# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


from typing import Optional

import torch

from verl.models.mcore.util import (
    preprocess_bshd_engine,
    preprocess_thd_engine,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def vocab_parallel_log_softmax(
    vp_logits: torch.Tensor,
) -> torch.Tensor:
    """
    1. Converts logits to float (in calculate_logits_max)
    2. Finds max logit across all partitions
    3. Shifts logits by the max for stability
    4. Exponentiates the shifted logits
    5. Computes the sum of exponentiated shifted logits across all partitions
    """
    from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    # seq_len, batch_size, top_k = target_topk_logps.size()
    vp_logits, logits_max = calculate_logits_max(vp_logits)

    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())

    vp_logits = vp_logits - logits_max.unsqueeze(dim=-1)
    exp_logits = vp_logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )
    log_sum_exp_logits = sum_exp_logits.log()
    return vp_logits - log_sum_exp_logits.unsqueeze(dim=-1)


def _chunked_student_topk_stats(
    vp_logits: torch.Tensor,
    target_topk_indices_local: torch.Tensor,
    chunk_size: int,
    local_topk: int,
    vocab_start_index: int,
):
    """Chunked, low-peak-memory computation of the vocab-parallel student statistics.

    This is the memory-optimized replacement for the full ``[N, V_local]`` fp32
    log-softmax + probs that the baseline materializes. Softmax is per-token
    (normalized over the vocab dim), so we split along the TOKEN dim into chunks
    of ``chunk_size`` rows and, per chunk, do the two TP reductions (MAX then SUM
    over the local vocab shard) to obtain the *global* log-normalizer ``logZ``.
    We never materialize the full ``[N, V_local]`` fp32 log-softmax / probs; the
    largest fp32 vocab workspace is a single ``[chunk_size, V_local]`` temporary.

    Returns (all in fp32, shaped like the (b, s, ...) inputs):
      - vp_source_topk_logps  : student log p at the teacher top-k *local* indices, (b, s, topk)
      - global_log_z          : per-token global log-normalizer logZ,               (b, s)
      - local_student_topk_logps : this rank's top-k student log p,                 (b, s, local_topk)
      - local_student_topk_ids   : this rank's top-k *global* vocab ids,            (b, s, local_topk)

    NOTE: `vp_logits` is NOT modified in place (each chunk works on a fresh float
    copy), so it can be safely saved for backward and re-used to recompute p.
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    tp_group = get_tensor_model_parallel_group()

    partition_vocab_size = vp_logits.size(-1)
    prefix_shape = target_topk_indices_local.shape[:-1]  # (b, s)
    topk = target_topk_indices_local.size(-1)

    logits_2d = vp_logits.reshape(-1, partition_vocab_size)  # (N, V_local) view, never written
    tgt_idx_2d = target_topk_indices_local.reshape(-1, topk)  # (N, topk) local indices, dummy 0 off-shard
    n_tokens = logits_2d.size(0)
    device = logits_2d.device

    vp_source_topk_logps_2d = torch.empty(n_tokens, topk, dtype=torch.float32, device=device)
    global_log_z_1d = torch.empty(n_tokens, dtype=torch.float32, device=device)
    local_student_topk_logps_2d = torch.empty(n_tokens, local_topk, dtype=torch.float32, device=device)
    local_student_topk_ids_2d = torch.empty(n_tokens, local_topk, dtype=torch.long, device=device)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        zc = logits_2d[start:end].float()  # (C, V_local) fresh fp32 copy

        # Global max over the full vocab for numerical stability (same as baseline).
        local_max = zc.amax(dim=-1)  # (C,)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Global sum-exp of the shifted logits. The exp temporary is freed right after the sum.
        exp_shifted = (zc - local_max.unsqueeze(-1)).exp()  # (C, V_local)
        local_sum_exp = exp_shifted.sum(dim=-1)  # (C,)
        del exp_shifted
        torch.distributed.all_reduce(local_sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # logZ = m + log(sum_j exp(z_j - m)); log p_j = z_j - logZ.
        logZ = local_max + local_sum_exp.log()  # (C,)
        global_log_z_1d[start:end] = logZ

        rows = torch.arange(end - start, device=device).unsqueeze(-1)  # (C, 1)
        # Student log p at the teacher top-k local indices (off-shard entries use dummy idx 0;
        # they are zeroed by the shared post-processing via topk_indices_in_vocab_mask).
        vp_source_topk_logps_2d[start:end] = zc[rows, tgt_idx_2d[start:end]] - logZ.unsqueeze(-1)

        # This rank's local top-k student candidates (log p and global ids) for the overlap diagnostic.
        vals, ids = torch.topk(zc, k=local_topk, dim=-1)
        local_student_topk_logps_2d[start:end] = vals - logZ.unsqueeze(-1)
        local_student_topk_ids_2d[start:end] = ids + vocab_start_index

        del zc

    return (
        vp_source_topk_logps_2d.view(*prefix_shape, topk),
        global_log_z_1d.view(*prefix_shape),
        local_student_topk_logps_2d.view(*prefix_shape, local_topk),
        local_student_topk_ids_2d.view(*prefix_shape, local_topk),
    )


class _VocabParallelKLDivergence(torch.autograd.Function):
    """
    Adapted from:
      https://github.com/verl-project/verl-recipe/blob/ccdb8d140dfc540761a9b209b854dbd2c0011e7e/gkd/megatron/megatron_kl_loss.py.

    When ``use_chunked_topk`` is True, the forward computes the per-token global
    log-normalizer logZ in TOKEN chunks and saves only ``vp_logits`` (its own
    dtype) + ``logZ`` (fp32, [N]) + the small [N, topk] support tensors for
    backward, instead of the full fp32 ``[N, V_local]`` student probs. Backward
    then recomputes p_j = exp(z_j - logZ) in chunks (no TP collective — logZ is
    already the global normalizer). This lowers the peak of the two full fp32
    vocab tensors that the baseline materializes, at the cost of more (but
    smaller) TP collectives. The math and all returned values are identical to
    the non-chunked path up to fp reduction-order rounding.
    """

    @staticmethod
    def forward(
        ctx,
        vp_logits: torch.Tensor,
        target_topk_logps: torch.Tensor,
        target_topk_indices: torch.Tensor,
        log_prob_min_clamp: Optional[float],
        use_chunked_topk: bool = False,
        chunk_size: int = 4096,
    ):
        """
        NOTE:
          - `target_topk_*` (indices/logprobs) are in *global vocab* coordinates.
          - `vp_logits` are the *local shard* of the vocab-parallel logits on this TP rank.
          This function masks out target top-k entries that do not belong to the local shard.
        """
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from megatron.core.tensor_parallel.utils import VocabUtility

        # Find the vocab range owned by this partition
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        partition_vocab_size = vp_logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        target_topk_indices_global = target_topk_indices.clone()

        # Which target top-k indices fall into this partition's vocab range?
        topk_indices_in_vocab_mask = (target_topk_indices >= vocab_start_index) & (
            target_topk_indices < vocab_end_index
        )

        # Convert global indices -> local indices for this shard.
        # For indices not on this shard, set index=0 as a dummy (and mask them out later).
        target_topk_indices = target_topk_indices.clone()
        target_topk_logps = target_topk_logps.clone()
        target_topk_indices = target_topk_indices - vocab_start_index
        target_topk_indices[~topk_indices_in_vocab_mask] = 0

        # Target probs/logps (teacher distribution restricted to top-k), masked to this shard.
        # Note: `target_topk_mass` is computed *before* masking-out-of-shard entries, so it represents
        # the mass of the provided top-k distribution (global), independent of TP sharding.
        if log_prob_min_clamp is not None:
            target_topk_logps = target_topk_logps.clamp_min(log_prob_min_clamp)
        target_topk_logps = target_topk_logps.float()
        target_topk_probs = torch.exp(target_topk_logps)
        target_topk_mass = torch.sum(target_topk_probs, dim=-1)
        target_topk_probs[~topk_indices_in_vocab_mask] = 0
        target_topk_logps[~topk_indices_in_vocab_mask] = 0

        topk = target_topk_indices.size(-1)
        local_topk = min(topk, partition_vocab_size)

        # ---- Student-side stats: gather student log p at teacher top-k, plus local student top-k. ----
        # The chunked path never materializes the full [N, V_local] fp32 log-softmax/probs.
        if use_chunked_topk:
            (
                vp_source_topk_logps,
                global_log_z,
                local_student_topk_logps,
                local_student_topk_ids,
            ) = _chunked_student_topk_stats(
                vp_logits, target_topk_indices, chunk_size, local_topk, vocab_start_index
            )
            vp_source_probs = None
        else:
            # Compute softmax over vocab-parallel logits
            vp_source_logps = vocab_parallel_log_softmax(vp_logits).float()
            vp_source_probs = torch.exp(vp_source_logps)
            global_log_z = None

            # Gather source log probabilities at the target top-k indices (local indices)
            vp_source_logps_2d = vp_source_logps.view(-1, partition_vocab_size)  # (b*s, vocab_shard)
            arange_1d = torch.arange(
                start=0, end=vp_source_logps_2d.size(0), device=vp_source_logps_2d.device
            )  # (b*s,)
            vp_source_topk_logps_2d = vp_source_logps_2d[
                arange_1d.unsqueeze(-1), target_topk_indices.view(-1, topk)
            ]  # (b*s, topk)
            vp_source_topk_logps = vp_source_topk_logps_2d.view(target_topk_indices.shape)  # (b, s, topk)

            # Compute the student's global top-k ids from per-rank vocab-shard candidates.
            local_student_topk_logps, local_student_topk_ids = torch.topk(vp_source_logps, k=local_topk, dim=-1)
            local_student_topk_ids = local_student_topk_ids + vocab_start_index

        # `active_mask` tracks entries that should receive gradient.
        # If clamping is enabled, entries with log p_i <= clamp have zero gradient w.r.t. logits.
        active_mask = topk_indices_in_vocab_mask
        if log_prob_min_clamp is not None:
            active_mask = active_mask & (vp_source_topk_logps > log_prob_min_clamp)
            vp_source_topk_logps = vp_source_topk_logps.clamp_min(log_prob_min_clamp)
            target_active_probs = target_topk_probs.clone()
            target_active_probs[~active_mask] = 0
            target_active_mass = target_active_probs.sum(dim=-1)
            torch.distributed.all_reduce(
                target_active_mass,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )
        else:
            target_active_mass = target_topk_mass

        # For out-of-shard entries, log p is set to 0 so they contribute nothing after all-reduce.
        vp_source_topk_logps[~topk_indices_in_vocab_mask] = 0

        #   This computes the forward KL: KL(P || Q), where
        #     P = target distribution (teacher top-k probs) and
        #     Q = source distribution (student probs at those indices).
        per_token_kl_loss = torch.sum(
            target_topk_probs * (target_topk_logps - vp_source_topk_logps),
            dim=-1,
        )  # (b, s)

        torch.distributed.all_reduce(
            per_token_kl_loss,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Save for backward. The chunked path saves the (bf16/fp16) logits + logZ and recomputes
        # p_j = exp(z_j - logZ); the baseline path saves the full fp32 probs directly.
        if use_chunked_topk:
            ctx.save_for_backward(
                vp_logits, global_log_z, target_topk_probs, target_topk_indices, active_mask, target_active_mass
            )
        else:
            ctx.save_for_backward(
                vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass
            )
        ctx.use_chunked_topk = use_chunked_topk
        ctx.chunk_size = chunk_size

        # For logging: mass of student probs that lands on the teacher's top-k indices.
        vp_source_topk_probs = vp_source_topk_logps.exp() * topk_indices_in_vocab_mask  # (b, s, topk)
        per_token_topk_mass = torch.sum(vp_source_topk_probs, dim=-1)  # (b, s)
        torch.distributed.all_reduce(
            per_token_topk_mass,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Student's global top-k ids: all-gather the per-rank local top-k candidates, then top-k again.
        gathered_student_topk_logps = [torch.empty_like(local_student_topk_logps) for _ in range(world_size)]
        gathered_student_topk_ids = [torch.empty_like(local_student_topk_ids) for _ in range(world_size)]
        torch.distributed.all_gather(
            gathered_student_topk_logps, local_student_topk_logps, group=get_tensor_model_parallel_group()
        )
        torch.distributed.all_gather(
            gathered_student_topk_ids, local_student_topk_ids, group=get_tensor_model_parallel_group()
        )
        student_topk_logps = torch.cat(gathered_student_topk_logps, dim=-1)
        student_topk_ids = torch.cat(gathered_student_topk_ids, dim=-1)
        _, student_topk_positions = torch.topk(student_topk_logps, k=topk, dim=-1)
        student_topk_ids = torch.gather(student_topk_ids, dim=-1, index=student_topk_positions)

        per_target_token_kl = target_topk_probs * (target_topk_logps - vp_source_topk_logps)
        torch.distributed.all_reduce(
            per_target_token_kl,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
        overlap_mask = (target_topk_indices_global.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        overlap_token_advantage_sum = (-per_target_token_kl * overlap_mask).sum(dim=-1)
        overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
        overlap_token_advantage = torch.where(
            overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
        )

        per_token_topk_mass = per_token_topk_mass.detach()
        target_topk_mass = target_topk_mass.detach()
        overlap_count = overlap_count.detach()
        overlap_token_advantage = overlap_token_advantage.detach()
        ctx.mark_non_differentiable(per_token_topk_mass, target_topk_mass, overlap_count, overlap_token_advantage)

        return per_token_kl_loss, per_token_topk_mass, target_topk_mass, overlap_count, overlap_token_advantage

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_source_mass: torch.Tensor,
        grad_target_mass: torch.Tensor,
        grad_overlap_count: torch.Tensor,
        grad_overlap_token_advantage: torch.Tensor,
    ):
        """
        Backprop for the per-token loss:
            L = sum_{i in S} q_i * (log q_i - clamp(log p_i))

        where:
          - S are the provided target top-k indices (global top-k, then masked per shard)
          - q_i are target (teacher) probabilities at those indices
          - p_i are source (student) probabilities at those indices
          - clamp(log p_i) is applied when `log_prob_min_clamp` is set

        Let A be the subset of S that are (1) on this shard and (2) not clamped
        (i.e., log p_i > log_prob_min_clamp when clamping is enabled).
        Define m_A = sum_{i in A} q_i (aggregated across TP ranks).

        Then for any vocab index j on this shard (with p = softmax(logits)):
            dL/dz_j = m_A * p_j - q_j * 1[j in A]
        """
        if ctx.use_chunked_topk:
            vp_logits, global_log_z, target_topk_probs, target_topk_indices, active_mask, target_active_mass = (
                ctx.saved_tensors
            )
            partition_vocab_size = vp_logits.size(-1)
            logits_2d = vp_logits.reshape(-1, partition_vocab_size)  # (N, V_local) view, never written
            n_tokens = logits_2d.size(0)

            # Allocate the (unavoidable) full [N, V_local] fp32 gradient once and fill it chunk by chunk,
            # recomputing p_j = exp(z_j - logZ). This avoids holding a second full fp32 vocab tensor
            # (the baseline keeps the saved fp32 probs AND the fresh fp32 grad alive simultaneously).
            grad_input = torch.empty(n_tokens, partition_vocab_size, dtype=torch.float32, device=logits_2d.device)
            m_A_1d = target_active_mass.reshape(-1)  # (N,)
            logZ_1d = global_log_z.reshape(-1)  # (N,)
            chunk_size = ctx.chunk_size
            for start in range(0, n_tokens, chunk_size):
                end = min(start + chunk_size, n_tokens)
                p_c = (logits_2d[start:end].float() - logZ_1d[start:end].unsqueeze(-1)).exp()  # (C, V_local)
                grad_input[start:end] = p_c * m_A_1d[start:end].unsqueeze(-1)  # m_A * p_j
                del p_c
            grad_input = grad_input.view(*target_active_mass.shape, partition_vocab_size)
        else:
            vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass = (
                ctx.saved_tensors
            )
            # Scale by m_A: grad starts as m_A * p_j for all j on this shard.
            grad_input = vp_source_probs * target_active_mass.unsqueeze(-1)  # [b, s, vocab_shard]

        topk = target_topk_indices.size(-1)
        grad_input_2d = grad_input.view(-1, grad_input.size(-1))
        target_topk_probs_flat = target_topk_probs.view(-1, topk)  # (b*s, topk)
        target_topk_indices_flat = target_topk_indices.view(-1, topk)  # (b*s, topk)

        # Subtract q_j for active entries (i.e., j in A), accumulating repeats via scatter_add_.
        # Index 0 is used as a dummy for top-k entries not on this shard (their q is zeroed by mask),
        # but index 0 may also be a real token index; scatter_add_ correctly accumulates duplicates.
        sub = target_topk_probs_flat * active_mask.view(-1, topk).to(grad_input_2d.dtype)  # (b*s, topk)
        grad_input_2d.scatter_add_(dim=1, index=target_topk_indices_flat, src=-sub)

        grad_input.mul_(grad_loss.unsqueeze(dim=-1))
        return grad_input, None, None, None, None, None


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested

    # 1. split across cp groups (bsz, seqlen, topk) => (bsz, seqlen/cp_size, topk)
    if data_format == "thd":
        teacher_topk_log_probs_cp_split, *_ = preprocess_thd_engine(teacher_topk_log_probs, pre_process=True)
        teacher_topk_ids_cp_split, *_ = preprocess_thd_engine(teacher_topk_ids, pre_process=True)
    else:
        teacher_topk_log_probs_cp_split, *_ = preprocess_bshd_engine(teacher_topk_log_probs, pre_process=True)
        teacher_topk_ids_cp_split, *_ = preprocess_bshd_engine(teacher_topk_ids, pre_process=True)
    assert teacher_topk_log_probs_cp_split.shape[:2] == teacher_topk_ids_cp_split.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across tp groups
    distillation_loss_config: DistillationLossConfig = config.distillation_loss
    distillation_losses, student_mass, teacher_mass, overlap_count, overlap_token_advantage = (
        _VocabParallelKLDivergence.apply(
            student_logits,
            teacher_topk_log_probs_cp_split,
            teacher_topk_ids_cp_split,
            distillation_loss_config.log_prob_min_clamp,
            distillation_loss_config.use_chunked_topk,
            distillation_loss_config.chunked_topk_chunk_size,
        )
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
