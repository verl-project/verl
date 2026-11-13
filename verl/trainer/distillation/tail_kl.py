# Copyright 2026 Bytedance Ltd. and/or its affiliates
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


def compute_tail_bucket_kl(
    student_topk_mass: torch.Tensor,
    teacher_topk_mass: torch.Tensor,
    tail_mass_eps: float,
) -> torch.Tensor:
    """Return the KL contribution of one aggregate non-top-k bucket.

    The teacher top-k IDs define a shared coarse-graining of the vocabulary:
    each selected token is its own category and every remaining token belongs
    to one tail category. ``tail_mass_eps`` only guards the student tail's
    logarithm; a zero teacher tail contributes zero by continuity.
    """
    # The epsilon guard is deliberately one-sided. The student tail appears inside
    # a logarithm, so q_tail == 0 would send the loss and its gradient to infinity;
    # clamping it to ``tail_mass_eps`` bounds both. The teacher tail is only a
    # multiplicative weight: p_tail -> 0 makes the whole term vanish by continuity
    # (x log x -> 0), so it is left unclamped -- only the input of its logarithm is
    # floored at the dtype minimum, which avoids 0 * (-inf) = nan.
    teacher_tail = (1.0 - teacher_topk_mass.float()).clamp(min=0.0, max=1.0)
    student_tail = (1.0 - student_topk_mass.float()).clamp(min=tail_mass_eps, max=1.0)
    teacher_tail_for_log = teacher_tail.clamp_min(torch.finfo(teacher_tail.dtype).tiny)
    return teacher_tail * (torch.log(teacher_tail_for_log) - torch.log(student_tail))


def compute_tail_aware_logit_gradient(
    student_probs: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_mask: torch.Tensor,
    student_topk_mass: torch.Tensor,
    teacher_topk_mass: torch.Tensor,
    tail_mass_eps: float,
) -> torch.Tensor:
    """Return the analytic gradient of coarse-grained KL with respect to logits.

    ``student_probs`` may be one vocabulary shard. Teacher entries outside that
    shard use dummy indices and must be false in ``teacher_topk_mask``.
    """
    teacher_tail = (1.0 - teacher_topk_mass).clamp(min=0.0, max=1.0)
    raw_student_tail = 1.0 - student_topk_mass
    student_tail = raw_student_tail.clamp(min=tail_mass_eps, max=1.0)
    tail_is_active = raw_student_tail > tail_mass_eps
    active_outside_scale = teacher_topk_mass - teacher_tail * student_topk_mass / student_tail
    outside_scale = torch.where(tail_is_active, active_outside_scale, teacher_topk_mass)
    grad_input = student_probs * outside_scale.unsqueeze(-1)

    source_topk_probs = torch.gather(student_probs, dim=-1, index=teacher_topk_indices)
    topk_scale = teacher_topk_mass + torch.where(tail_is_active, teacher_tail, 0.0)
    correction = (
        (topk_scale - outside_scale).unsqueeze(-1) * source_topk_probs - teacher_topk_probs
    ) * teacher_topk_mask.to(student_probs.dtype)
    grad_input.scatter_add_(dim=-1, index=teacher_topk_indices, src=correction)
    return grad_input
