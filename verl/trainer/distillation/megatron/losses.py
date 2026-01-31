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
    Source: https://github.com/verl-project/verl-recipe/blob/ccdb8d140dfc540761a9b209b854dbd2c0011e7e/gkd/megatron/megatron_kl_loss.py
    
    TODO:
    (WorkerDict pid=3705745) /data/jacob/anaconda3/envs/verlMega/lib/python3.12/site-packages/torch/autograd/graph.py:829: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at /pytorch/torch/csrc/autograd/autograd_not_implemented_fallback.cpp:62.)
    
    """
    @staticmethod
    def forward(ctx, vp_logits: torch.Tensor, target_topk_logps: torch.Tensor, target_topk_indices: torch.Tensor, log_prob_min_clamp: Optional[float]):
        """
        TODO: not that target topk is global, while vp_logits is on this shard only
        """
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from megatron.core.tensor_parallel.utils import VocabUtility


        # Compute softmax over vocab parallel logits
        exp_logits, sum_exp_logits = vocab_parallel_softmax(vp_logits)

        # Find which indices in the vocab are in this partition
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        partition_vocab_size = vp_logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        # Find which of the target top-k indices are in this partition
        topk_indices_in_vocab_mask = (target_topk_indices >= vocab_start_index) & (
            target_topk_indices < vocab_end_index
        )

        # Set all target indices not in this partition to zero; we will ignore the probabilities at these indices on this partition later
        target_topk_indices = target_topk_indices.clone()
        target_topk_logps = target_topk_logps.clone()
        target_topk_indices = target_topk_indices - vocab_start_index
        target_topk_indices[~topk_indices_in_vocab_mask] = 0

        # Set all target probabilities not in this partition to zero
        if log_prob_min_clamp is not None:
            target_topk_logps = target_topk_logps.clamp_min(log_prob_min_clamp)
        target_topk_probs = torch.exp(target_topk_logps)
        target_topk_mass = torch.sum(target_topk_probs, dim=-1)
        target_topk_probs[~topk_indices_in_vocab_mask] = 0
        target_topk_logps[~topk_indices_in_vocab_mask] = 0
        
        # Compute source probabilities
        target_topk_logps_origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)
        vp_source_probs = exp_logits
        vp_source_probs = vp_source_probs.div(sum_exp_logits.unsqueeze(-1))
        vp_source_probs_2d = vp_source_probs.view(-1, partition_vocab_size)  # (b*s, h/tp)

        # Gather source probabilities at target top-k indices
        arange_1d = torch.arange(
            start=0, end=vp_source_probs_2d.size(0), device=vp_source_probs_2d.device
        )  # (b*s, )
        vp_source_topk_probs_2d = vp_source_probs_2d[
            arange_1d.unsqueeze(-1), target_topk_indices.view(-1, topk)
        ]  # (b*s, topk)
        vp_source_topk_probs = vp_source_topk_probs_2d.view(
            target_topk_logps_origin_shape
        )  # (b, s, topk)

        # Source log probabilities
        vp_source_topk_logps = torch.log(1e-20 + vp_source_topk_probs)

        # Active mask tracks where there is gradient
        # Apply clamping; gradient of logprobs is zero for clamped probs
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

        # Set all entries in the source log probabilities that correspond to target top-k indices not in this partition to zero
        vp_source_topk_logps[~topk_indices_in_vocab_mask] = 0


        # KL(P||Q)会强制 Q 覆盖 P 的所有模式（避免漏峰）
        # KL(Q||P)会鼓励 Q 聚焦于 P 的一个模式（避免多峰)
        # 这里使用 KL(P||Q)，其中P为target，Q为source，鼓励source学习target的所有模式

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

        # Compute amount of mass in source probs at target top-k indices (for logging purposes)
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
    def backward(ctx, grad_loss: torch.Tensor, grad_source_mass: torch.Tensor, grad_target_mass: torch.Tensor):
        """
        Computes grad of L = sum_{i in S} q_i (log q_i - clamp(log p_i))
        w.r.t. logits z (where p = softmax(z)).

        Let A be the subset of S where log p_i > log_prob_min_clamp (i.e., unclamped entries).
        Define m_A = sum_{i in A} q_i.

        Then for any vocab index j on this shard:
            dL/dz_j = m_A * p_j - q_j * 1[j in A]
        """

        vp_source_probs, target_topk_probs, target_topk_indices, active_mask, target_active_mass = (
            ctx.saved_tensors
        )
        # source_probs, target_probs = ctx.saved_tensors
        # KL 散度对 vp_logits 的梯度为: (student_softmax_logits - valid_target_logits)

        # Compute amount of mass in teacher probs corresponding to tokens where student has non-zero grad 
        grad_input = vp_source_probs * target_active_mass.unsqueeze(-1)  # shape: [b, s, vp_size]

        topk = target_topk_indices.size(-1)
        grad_input_2d = grad_input.view(-1, grad_input.size(-1))
        target_topk_probs_flat = target_topk_probs.view(-1, topk)  # (b*s, topk)
        target_topk_indices_flat = target_topk_indices.view(-1, topk)  # (b*s, topk)
        
        # subtract teacher probs for active entries, accumulating if indices repeat (index 0 might repeat)
        # index 0 is a dummy index for top-k entries not on this shard where probs have been set to zero
        # index 0 may also be a valid index in the top-k entries, so scatter_add_ to sum all repeats
        sub = target_topk_probs_flat * active_mask.view(-1, topk).to(grad_input_2d.dtype)  # (b*s, topk)
        grad_input_2d.scatter_add_(dim=1, index=target_topk_indices_flat, src=-sub)

        grad_input.mul_(grad_loss.unsqueeze(dim=-1))
        return grad_input, None, None, None  # 返回给第一个输入 vp_logits 的梯度

def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities."""
    return _VocabParallelKLDivergence.apply(student_logits, teacher_topk_log_probs, teacher_topk_indices, config.log_prob_min_clamp)