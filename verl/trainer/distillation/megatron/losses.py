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


def _gather_teacher_log_probs_on_student_topk(
    student_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    missing_log_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather teacher log-probs on student top-k ids from returned teacher top-k entries."""
    matches = student_topk_ids.unsqueeze(-1) == teacher_topk_ids.unsqueeze(-2)
    missing = torch.full(
        student_topk_ids.shape,
        missing_log_prob,
        dtype=teacher_topk_log_probs.dtype,
        device=teacher_topk_log_probs.device,
    )
    non_match = torch.full_like(teacher_topk_log_probs.unsqueeze(-2), torch.finfo(teacher_topk_log_probs.dtype).min)
    matched = torch.where(matches, teacher_topk_log_probs.unsqueeze(-2), non_match)
    has_match = matches.any(dim=-1)
    matched = matched.max(dim=-1).values
    return torch.where(has_match, matched, missing), has_match


class _VocabParallelKLDivergence(torch.autograd.Function):
    """
    Adapted from:
      https://github.com/verl-project/verl-recipe/blob/ccdb8d140dfc540761a9b209b854dbd2c0011e7e/gkd/megatron/megatron_kl_loss.py.
    """

    @staticmethod
    def forward(
        ctx,
        vp_logits: torch.Tensor,
        target_topk_logps: torch.Tensor,
        target_topk_indices: torch.Tensor,
        log_prob_min_clamp: Optional[float],
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

        # Compute softmax over vocab-parallel logits
        vp_source_logps = vocab_parallel_log_softmax(vp_logits).float()
        vp_source_probs = torch.exp(vp_source_logps)

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

        # Gather source log probabilities at the target top-k indices (local indices)
        topk = target_topk_indices.size(-1)
        vp_source_logps_2d = vp_source_logps.view(-1, partition_vocab_size)  # (b*s, vocab_shard)
        arange_1d = torch.arange(start=0, end=vp_source_logps_2d.size(0), device=vp_source_logps_2d.device)  # (b*s,)
        vp_source_topk_logps_2d = vp_source_logps_2d[
            arange_1d.unsqueeze(-1), target_topk_indices.view(-1, topk)
        ]  # (b*s, topk)
        vp_source_topk_logps = vp_source_topk_logps_2d.view(target_topk_indices.shape)  # (b, s, topk)

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

        ctx.save_for_backward(vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass)

        # For logging: mass of student probs that lands on the teacher's top-k indices.
        vp_source_topk_probs = vp_source_topk_logps.exp() * topk_indices_in_vocab_mask  # (b, s, topk)
        per_token_topk_mass = torch.sum(vp_source_topk_probs, dim=-1)  # (b, s)
        torch.distributed.all_reduce(
            per_token_topk_mass,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Compute the student's global top-k ids from per-rank vocab-shard candidates.
        local_topk = min(topk, partition_vocab_size)
        local_student_topk_logps, local_student_topk_ids = torch.topk(vp_source_logps, k=local_topk, dim=-1)
        local_student_topk_ids = local_student_topk_ids + vocab_start_index
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
        vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass = ctx.saved_tensors

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
        return grad_input, None, None, None


class _VocabParallelReverseKLDivergenceOnStudentTopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vp_logits: torch.Tensor,
        teacher_topk_log_probs: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        student_topk: int,
        log_prob_min_clamp: float,
    ):
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from megatron.core.tensor_parallel.utils import VocabUtility

        vp_source_logps = vocab_parallel_log_softmax(vp_logits).float()
        vp_source_probs = vp_source_logps.exp()

        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        partition_vocab_size = vp_logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        local_topk = min(student_topk, partition_vocab_size)
        local_student_topk_logps, local_student_topk_ids = torch.topk(vp_source_logps, k=local_topk, dim=-1)
        local_student_topk_ids = local_student_topk_ids + vocab_start_index
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
        student_topk_logps, student_topk_positions = torch.topk(student_topk_logps, k=student_topk, dim=-1)
        student_topk_ids = torch.gather(student_topk_ids, dim=-1, index=student_topk_positions)

        teacher_logps_on_student_topk, overlap_mask = _gather_teacher_log_probs_on_student_topk(
            student_topk_ids=student_topk_ids,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            missing_log_prob=log_prob_min_clamp,
        )

        student_mass = student_topk_logps.exp().sum(dim=-1)
        teacher_mass = (teacher_logps_on_student_topk.exp() * overlap_mask).sum(dim=-1)

        student_topk_ids_in_vocab_mask = (student_topk_ids >= vocab_start_index) & (
            student_topk_ids < vocab_end_index
        )
        student_topk_local_ids = student_topk_ids - vocab_start_index
        student_topk_local_ids = student_topk_local_ids.clone()
        student_topk_local_ids[~student_topk_ids_in_vocab_mask] = 0

        active_topk_mask = student_topk_logps > log_prob_min_clamp
        active_mask = student_topk_ids_in_vocab_mask & active_topk_mask
        student_topk_logps = student_topk_logps.clamp_min(log_prob_min_clamp)
        teacher_logps_on_student_topk = teacher_logps_on_student_topk.clamp_min(log_prob_min_clamp)

        student_topk_probs = student_topk_logps.exp()
        distillation_losses = torch.sum(
            student_topk_probs * (student_topk_logps - teacher_logps_on_student_topk), dim=-1
        )

        grad_coeff = student_topk_probs * (student_topk_logps - teacher_logps_on_student_topk + 1.0)
        active_grad_coeff = grad_coeff * active_mask.to(grad_coeff.dtype)
        grad_coeff_sum = (grad_coeff * active_topk_mask.to(grad_coeff.dtype)).sum(dim=-1)

        overlap_count = overlap_mask.sum(dim=-1)
        token_advantage = teacher_logps_on_student_topk - student_topk_logps
        overlap_token_advantage_sum = (token_advantage * overlap_mask).sum(dim=-1)
        overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
        overlap_token_advantage = torch.where(
            overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
        )

        ctx.save_for_backward(vp_source_probs, student_topk_local_ids, active_grad_coeff, active_mask, grad_coeff_sum)

        student_mass = student_mass.detach()
        teacher_mass = teacher_mass.detach()
        overlap_count = overlap_count.detach()
        overlap_token_advantage = overlap_token_advantage.detach()
        ctx.mark_non_differentiable(student_mass, teacher_mass, overlap_count, overlap_token_advantage)

        return distillation_losses, student_mass, teacher_mass, overlap_count, overlap_token_advantage

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_student_mass: torch.Tensor,
        grad_teacher_mass: torch.Tensor,
        grad_overlap_count: torch.Tensor,
        grad_overlap_token_advantage: torch.Tensor,
    ):
        vp_source_probs, student_topk_local_ids, active_grad_coeff, active_mask, grad_coeff_sum = ctx.saved_tensors

        grad_input = -vp_source_probs * grad_coeff_sum.unsqueeze(-1)

        topk = student_topk_local_ids.size(-1)
        grad_input_2d = grad_input.view(-1, grad_input.size(-1))
        student_topk_local_ids_flat = student_topk_local_ids.view(-1, topk)
        active_grad_coeff_flat = active_grad_coeff.view(-1, topk) * active_mask.view(-1, topk).to(
            active_grad_coeff.dtype
        )
        grad_input_2d.scatter_add_(dim=1, index=student_topk_local_ids_flat, src=active_grad_coeff_flat)

        grad_input.mul_(grad_loss.unsqueeze(dim=-1))
        return grad_input, None, None, None, None


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
        )
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_reverse_kl_student_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute truncated reverse KL on the student's top-k support."""
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested

    if data_format == "thd":
        teacher_topk_log_probs_cp_split, *_ = preprocess_thd_engine(teacher_topk_log_probs, pre_process=True)
        teacher_topk_ids_cp_split, *_ = preprocess_thd_engine(teacher_topk_ids, pre_process=True)
    else:
        teacher_topk_log_probs_cp_split, *_ = preprocess_bshd_engine(teacher_topk_log_probs, pre_process=True)
        teacher_topk_ids_cp_split, *_ = preprocess_bshd_engine(teacher_topk_ids, pre_process=True)
    assert teacher_topk_log_probs_cp_split.shape[:2] == teacher_topk_ids_cp_split.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is None:
        raise ValueError("reverse_kl_student_topk requires distillation_loss.log_prob_min_clamp.")

    student_topk = loss_config.topk
    if student_topk is None:
        raise ValueError("reverse_kl_student_topk requires distillation_loss.topk.")
    teacher_logprob_topk = teacher_topk_ids_cp_split.shape[-1]
    if teacher_logprob_topk < student_topk:
        raise ValueError(
            "Teacher log-prob payload for reverse_kl_student_topk must be at least as wide as "
            f"distillation_loss.topk, but got {teacher_logprob_topk} < {student_topk}."
        )

    distillation_losses, student_mass, teacher_mass, overlap_count, overlap_token_advantage = (
        _VocabParallelReverseKLDivergenceOnStudentTopK.apply(
            student_logits,
            teacher_topk_log_probs_cp_split,
            teacher_topk_ids_cp_split,
            student_topk,
            loss_config.log_prob_min_clamp,
        )
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
