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
import torch.nn.functional as F

from verl.workers.config import DistillationConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    return F.kl_div(input=log_q, target=log_p, reduction="none", log_target=True).sum(dim=-1)


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
    log_q = log_q.float()
    log_p = log_p.float()
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
    config: DistillationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_indices)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(config.log_prob_min_clamp)
    distillation_losses = kullback_leibler_divergence(
        log_q=student_topk_log_probs, log_p=teacher_topk_log_probs, loss_mode="forward"
    )
    return distillation_losses, student_mass, teacher_mass
