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


import torch
from typing import Optional
from verl.workers.config import DistillationConfig
from verl.trainer.distillation.megatron.utils import vocab_parallel_softmax

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
        exp_logits, sum_exp_logits = vocab_parallel_softmax(vp_logits)

        # Find the vocab range owned by this partition
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        partition_vocab_size = vp_logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

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
        target_topk_probs = torch.exp(target_topk_logps)
        target_topk_mass = torch.sum(target_topk_probs, dim=-1)
        target_topk_probs[~topk_indices_in_vocab_mask] = 0
        target_topk_logps[~topk_indices_in_vocab_mask] = 0

        # Compute source probabilities p = softmax(vp_logits) on this shard
        target_topk_logps_origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)
        vp_source_probs = exp_logits
        vp_source_probs = vp_source_probs.div(sum_exp_logits.unsqueeze(-1))
        vp_source_probs_2d = vp_source_probs.view(-1, partition_vocab_size)  # (b*s, vocab_shard)

        # Gather source probabilities at the target top-k indices (local indices)
        arange_1d = torch.arange(
            start=0, end=vp_source_probs_2d.size(0), device=vp_source_probs_2d.device
        )  # (b*s,)
        vp_source_topk_probs_2d = vp_source_probs_2d[
            arange_1d.unsqueeze(-1), target_topk_indices.view(-1, topk)
        ]  # (b*s, topk)
        vp_source_topk_probs = vp_source_topk_probs_2d.view(
            target_topk_logps_origin_shape
        )  # (b, s, topk)

        # Source log probabilities (student)
        vp_source_topk_logps = torch.log(1e-20 + vp_source_topk_probs)

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

        ctx.save_for_backward(
            vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass
        )

        # For logging: mass of student probs that lands on the teacher's top-k indices.
        vp_source_topk_probs = vp_source_topk_probs.detach()
        vp_source_topk_probs[~topk_indices_in_vocab_mask] = 0
        per_token_topk_mass = torch.sum(vp_source_topk_probs, dim=-1)  # (b, s)
        torch.distributed.all_reduce(
            per_token_topk_mass,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
        ctx.mark_non_differentiable(per_token_topk_mass, target_topk_mass)

        return per_token_kl_loss, per_token_topk_mass.detach(), target_topk_mass.detach()

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_source_mass: torch.Tensor,
        grad_target_mass: torch.Tensor,
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
        return grad_input, None, None, None


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities."""
    return _VocabParallelKLDivergence.apply(student_logits, teacher_topk_log_probs, teacher_topk_indices, config.log_prob_min_clamp)