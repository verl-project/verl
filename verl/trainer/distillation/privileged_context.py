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
"""Privileged-context helpers for On-Policy Self-Distillation (OPSD).

In OPSD the teacher and student share weights but see different contexts: the
student sees only the problem, while the teacher additionally sees the
ground-truth solution. The teacher does not generate -- it scores the student's
own on-policy response conditioned on the privileged solution. These two pure
helpers build the teacher's privileged input sequence and realign the teacher's
per-token top-k outputs back onto the student's ``prompt + response`` positions,
so the rest of verl's on-policy-distillation pipeline is reused unchanged.
"""

import torch


def build_privileged_sequence(
    prompt_ids: list[int],
    response_ids: list[int],
    solution_ids: list[int],
    prefix_ids: list[int],
    suffix_ids: list[int],
) -> list[int]:
    """Build the OPSD teacher's input token sequence.

    Layout: ``prompt + prefix + solution + suffix + response``. The teacher sees
    the ground-truth solution (wrapped in ``prefix`` / ``suffix`` marker tokens)
    spliced between the problem and the student's response; the student never
    sees it. The response is the suffix of the sequence, exactly as in the plain
    ``prompt + response`` teacher input, so the teacher's scores for the response
    tokens remain well defined.

    Args:
        prompt_ids: the student's prompt (problem) token ids.
        response_ids: the student's on-policy response token ids.
        solution_ids: the ground-truth solution token ids.
        prefix_ids: marker tokens placed before the solution (e.g. a
            "reference solution begin" tag). May be empty.
        suffix_ids: marker tokens placed after the solution (e.g. an "end" tag
            plus a transition prompt). May be empty.

    Returns:
        The concatenated teacher input token ids.
    """
    return prompt_ids + prefix_ids + solution_ids + suffix_ids + response_ids


def slice_privileged_teacher_to_student(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realign privileged-context teacher outputs onto the student's positions.

    The teacher's top-k outputs are computed over the privileged sequence
    (``prompt + prefix + solution + suffix + response``) and are 1:1 aligned to
    it. Only the final ``response_length`` rows -- the teacher's per-token scores
    for the response tokens under the privileged context -- are distillation
    targets. This keeps those rows and pads the ``student_prompt_length`` prompt
    rows (the downstream response mask zeroes the prompt region out anyway), so
    the returned tensors are aligned to the student's ``prompt + response`` and
    feed the existing padding / loss path unchanged.

    Args:
        teacher_ids: ``(privileged_len + response_length, k)`` teacher top-k ids.
        teacher_logprobs: ``(privileged_len + response_length, k)`` teacher
            top-k log-probs.
        student_prompt_length: length of the student's (non-privileged) prompt.
        response_length: number of response tokens.
        pad_token_id: id used to fill the padded prompt rows.

    Returns:
        ``(ids, logprobs)``, each ``(student_prompt_length + response_length, k)``
        and aligned to the student's ``prompt + response`` sequence.
    """
    k = teacher_ids.shape[-1]
    response_ids = teacher_ids[-response_length:]
    response_logprobs = teacher_logprobs[-response_length:]
    prompt_ids = torch.full(
        (student_prompt_length, k), pad_token_id, dtype=response_ids.dtype, device=response_ids.device
    )
    prompt_logprobs = torch.zeros(
        (student_prompt_length, k), dtype=response_logprobs.dtype, device=response_logprobs.device
    )
    ids = torch.cat([prompt_ids, response_ids], dim=0)
    logprobs = torch.cat([prompt_logprobs, response_logprobs], dim=0)
    return ids, logprobs
