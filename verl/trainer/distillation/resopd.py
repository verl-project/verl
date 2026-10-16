# Copyright 2026 Individual Contributor: Penghui Yang and the ResOPD authors
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
"""ResOPD: exact coarse gradient plus a sampled within-tail residual.

This is the event-control estimator used in the paper, extracted from the
authors' research implementation. Only PyTorch is needed to use this dense-tensor
kernel; the verl adapter handles packing and masking.
"""

import torch


def resopd_loss(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_sampled_log_probs: torch.Tensor,
    teacher_sampled_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-token losses with the ResOPD backward gradient.

    Args:
        student_logits: Full student logits, shape ``[..., V]``.
        teacher_topk_log_probs: Full-normalized teacher log probabilities on a
            fixed, unique support S, shape ``[..., K]`` (not renormalized on S).
        teacher_topk_ids: Token IDs for S, same shape as teacher_topk_log_probs.
        teacher_sampled_log_probs: Teacher log probability of the actual student
            action A, shape ``[..., 1]``. This is required even when A is outside S.
        teacher_sampled_ids: The actual action A, shape ``[..., 1]``.

    Every output has shape ``[...]``. Inputs must share the token vocabulary
    and prefix alignment. The expected gradient is the full reverse-KL gradient
    when A is sampled from the current full student distribution. Mask prompt
    and padding positions in the caller. FP16/BF16 calculations use FP32;
    FP64 is retained for numerical checks.

    The forward loss is the original conditional-head/sample estimator of KL;
    its backward is a stopped-coefficient score-gradient surrogate. It is not
    a deterministic KL objective and individual forward values may be negative.
    """
    if not student_logits.is_floating_point() or student_logits.ndim < 1:
        raise ValueError("student_logits must be a floating-point tensor with a vocabulary axis")
    prefix_shape = student_logits.shape[:-1]
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError("Teacher support IDs and log probabilities must have identical shapes")
    if teacher_topk_ids.shape[:-1] != prefix_shape:
        raise ValueError("Teacher support must align with the student prefixes")
    if not 0 < teacher_topk_ids.shape[-1] <= student_logits.shape[-1]:
        raise ValueError("Teacher support size must be between one and the vocabulary size")
    if teacher_sampled_ids.shape != (*prefix_shape, 1) or teacher_sampled_log_probs.shape != (*prefix_shape, 1):
        raise ValueError("Expected one actual sampled token and teacher score per prefix")

    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_p = student_logits.to(dtype).log_softmax(dim=-1)
    support_log_p = log_p.gather(-1, teacher_topk_ids.long())
    support_log_q = teacher_topk_log_probs.detach().to(dtype)
    support_p = support_log_p.exp()
    conditional_p = support_log_p.softmax(dim=-1)
    support_ratio = support_log_p - support_log_q

    # Conditional head mean: backward is mu_head = E[r_A s_A | A in S].
    head_value = (conditional_p * support_ratio).sum(dim=-1)
    head_surrogate = ((conditional_p * support_ratio).detach() * support_log_p).sum(dim=-1)

    sampled_ids = teacher_sampled_ids.long()
    sampled_log_p = log_p.gather(-1, sampled_ids).squeeze(-1)
    sampled_ratio = sampled_log_p - teacher_sampled_log_probs.detach().to(dtype).squeeze(-1)
    sampled_surrogate = sampled_ratio.detach() * sampled_log_p
    head_hit = teacher_topk_ids.eq(sampled_ids).any(dim=-1)
    student_mass = support_p.sum(dim=-1)
    teacher_mass = support_log_q.exp().sum(dim=-1)

    # mu_B = log(tau_p/tau_q) * grad log(tau_p). Complement clamps are
    # numerical guards, matching the training implementation; event centering
    # still uses the unmodified support mass P.
    epsilon = torch.finfo(dtype).eps
    student_tail_mass = (1.0 - student_mass).clamp_min(epsilon)
    teacher_tail_mass = (1.0 - teacher_mass).clamp_min(epsilon)
    tail_log_ratio = student_tail_mass.log() - teacher_tail_mass.log()
    support_score = (support_p.detach() * support_log_p).sum(dim=-1)
    bucket_surrogate = -tail_log_ratio.detach() * support_score / student_tail_mass.detach()

    # RB(A) - (I[A in S] - P) * (mu_head - mu_B)
    # = h + tau_p * mu_B + I[A not in S] * (r_A s_A - mu_B).
    centered_event = head_hit.to(dtype) - student_mass.detach()
    surrogate = torch.where(head_hit, head_surrogate, sampled_surrogate)
    surrogate = surrogate - centered_event.detach() * (head_surrogate - bucket_surrogate)
    value = torch.where(head_hit, head_value, sampled_ratio)
    losses = value.detach() + (surrogate - surrogate.detach())

    return {
        "distillation_losses": losses,
        "student_mass": student_mass.detach(),
        "teacher_mass": teacher_mass.detach(),
        "resopd_head_hit": head_hit.to(dtype),
        "resopd_centered_event": centered_event.detach(),
        "resopd_tail_log_ratio": tail_log_ratio.detach(),
    }
