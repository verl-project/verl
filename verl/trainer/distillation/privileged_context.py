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


def resolve_privileged_solution(sample_kwargs: dict | None, key: str) -> str | None:
    """Resolve the ground-truth solution from a (possibly nested) sample field.

    ``key`` may be dotted (e.g. ``"reward_model.ground_truth"``) to reach into
    nested dicts -- verl stores the ground truth at
    ``non_tensor_batch["reward_model"]["ground_truth"]``, not at a top-level key,
    so a flat ``.get(key)`` silently misses it and OPSD degrades to plain OPD.

    Normalizes numpy 0-d scalars (``.item()``), arrays/lists (joined with
    newlines), and dicts to a stripped string. Returns ``None`` if the field is
    absent or resolves to an empty string.
    """
    if sample_kwargs is None:
        return None
    cur = sample_kwargs
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    if hasattr(cur, "item") and getattr(cur, "size", 1) == 1:
        cur = cur.item()
    elif hasattr(cur, "tolist"):
        cur = cur.tolist()
    if isinstance(cur, list | tuple):
        cur = "\n".join(str(x) for x in cur) if len(cur) else None
    if cur is None:
        return None
    solution = str(cur).strip()
    return solution or None


def build_privileged_sequence(
    prompt_ids: list[int],
    response_ids: list[int],
    solution_ids: list[int],
    prefix_ids: list[int],
    suffix_ids: list[int],
    insert_before_token_ids: list[int] | None = None,
) -> list[int]:
    """Build the OPSD teacher's input token sequence.

    Default layout: ``prompt + prefix + solution + suffix + response`` -- the
    privileged solution (wrapped in ``prefix`` / ``suffix`` markers) is appended
    to the student prompt, and the response is the suffix exactly as in the plain
    ``prompt + response`` teacher input.

    ``prompt_ids`` is the post-``apply_chat_template`` prompt, so it ends with the
    template's assistant-turn opener. Appending the solution after it puts the
    solution inside the assistant turn. If a template needs the solution placed
    *before* a specific marker (e.g. the assistant-turn opener), pass
    ``insert_before_token_ids``: the solution block is then inserted before the
    **last** occurrence of that sub-sequence in ``prompt_ids``. If the marker is
    not found, this falls back to the default append.

    Args:
        prompt_ids: the student's prompt (problem) token ids.
        response_ids: the student's on-policy response token ids.
        solution_ids: the ground-truth solution token ids.
        prefix_ids: marker tokens placed before the solution. May be empty.
        suffix_ids: marker tokens placed after the solution. May be empty.
        insert_before_token_ids: if provided and found, insert the solution block
            before the last occurrence of this sub-sequence in ``prompt_ids``.

    Returns:
        The concatenated teacher input token ids.
    """
    block = prefix_ids + solution_ids + suffix_ids
    if insert_before_token_ids:
        m = len(insert_before_token_ids)
        for i in range(len(prompt_ids) - m, -1, -1):
            if prompt_ids[i : i + m] == insert_before_token_ids:
                return prompt_ids[:i] + block + prompt_ids[i:] + response_ids
        # marker not found -> fall through to the default append
    return prompt_ids + block + response_ids


def slice_privileged_teacher_to_student(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realign privileged-context teacher outputs onto the student's positions.

    The teacher's top-k outputs are computed over the privileged sequence and are
    1:1 aligned to it. Only the final ``response_length`` rows -- the teacher's
    per-token scores for the response tokens under the privileged context -- are
    distillation targets. This keeps those rows and pads the
    ``student_prompt_length`` prompt rows (the downstream response mask zeroes the
    prompt region out anyway), so the returned tensors are aligned to the
    student's ``prompt + response`` and feed the existing padding / loss path
    unchanged.

    Args:
        teacher_ids: ``(privileged_len + response_length, k)`` teacher top-k ids.
        teacher_logprobs: ``(privileged_len + response_length, k)`` teacher
            top-k log-probs.
        student_prompt_length: length of the student's (non-privileged) prompt.
        response_length: number of response tokens (may be 0).
        pad_token_id: id used to fill the padded prompt rows; ``None`` falls back
            to ``0`` (these rows are masked out downstream, so the value is inert).

    Returns:
        ``(ids, logprobs)``, each ``(student_prompt_length + response_length, k)``
        and aligned to the student's ``prompt + response`` sequence.
    """
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")
    if pad_token_id is None:
        pad_token_id = 0
    k = teacher_ids.shape[-1]
    # Index from an explicit start: ``teacher_ids[-0:]`` would return the whole
    # tensor, so a 0-length response must slice from the end, not from ``-0``.
    start = teacher_ids.shape[0] - response_length
    response_ids = teacher_ids[start:]
    response_logprobs = teacher_logprobs[start:]
    prompt_ids = torch.full(
        (student_prompt_length, k), int(pad_token_id), dtype=response_ids.dtype, device=response_ids.device
    )
    prompt_logprobs = torch.zeros(
        (student_prompt_length, k), dtype=response_logprobs.dtype, device=response_logprobs.device
    )
    ids = torch.cat([prompt_ids, response_ids], dim=0)
    logprobs = torch.cat([prompt_logprobs, response_logprobs], dim=0)
    return ids, logprobs
