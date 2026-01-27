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


import math

import torch
import torch.nn.functional as F


def clamp_log_probs(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp log probabilities to avoid numerical instability and handle inf minus inf for masked top-k probs."""
    min_log_prob = math.log(eps)
    log_p_clamped = torch.clamp(log_p, min=min_log_prob)
    log_q_clamped = torch.clamp(log_q, min=min_log_prob)
    return log_p_clamped, log_q_clamped


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_q_clamped, log_p_clamped = clamp_log_probs(log_q, log_p)
    return F.kl_div(input=log_q_clamped, target=log_p_clamped, reduction="none", log_target=True).sum(dim=-1)


def jensen_shannon_divergence(log_q: torch.Tensor, log_p: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between two distributions given their log probabilities.

    JSD(β) = β * KL(p || m) + (1 - β) * KL(q || m), where m = beta * p + (1 - beta) * q

    The gradients of JSD(β) behave similarly to forward KL and reverse KL when β is close
    to 0 and 1 respectively. See https://arxiv.org/abs/2306.13649

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        beta (float):
            JSD interpolation weight. When beta=0, behaves like forward KL.
            When beta=1, behaves like reverse KL.

    Returns:
        torch.Tensor: JSD loss per token, shape (batch_size, response_length).
    """
    q = log_q.exp()
    p = log_p.exp()
    m = beta * p + (1 - beta) * q
    log_m = m.log()
    kl1 = kl_divergence(log_q=log_m, log_p=log_p)
    kl2 = kl_divergence(log_q=log_m, log_p=log_q)
    loss = beta * kl1 + (1 - beta) * kl2
    return loss


def kullback_leibler_divergence(log_q: torch.Tensor, log_p: torch.Tensor, loss_mode: str) -> torch.Tensor:
    """
    Compute forward or reverse KL divergence between two distributions given their log probabilities.

    forward KL: KL(p || q) = sum(p * (log_p - log_q))
    reverse KL: KL(q || p) = sum(q * (log_q - log_p))

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        loss_mode (str):
            KL divergence direction: "forward" or "reverse".

    Returns:
        torch.Tensor: KL divergence loss per token, shape (batch_size, response_length).
    """
    match loss_mode:
        case "forward":
            return kl_divergence(log_q=log_q, log_p=log_p)
        case "reverse":
            return kl_divergence(log_q=log_p, log_p=log_q)
        case _:
            raise ValueError(f"Unsupported loss mode: {loss_mode}. Supported modes are: ['forward', 'reverse']")


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_indices)
    distillation_losses = kullback_leibler_divergence(
        log_q=student_topk_log_probs, log_p=teacher_topk_log_probs, loss_mode="forward"
    )
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    return distillation_losses, student_mass, teacher_mass
