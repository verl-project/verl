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
"""CPU tests for the OPSD privileged-context helpers.

These pin the two pure token-surgery ops that distinguish on-policy
self-distillation (OPSD) from plain OPD: building the teacher's privileged input
(problem + ground-truth solution + student response) and realigning the teacher's
per-token scores back onto the student's ``prompt + response`` positions.
"""

import torch

from verl.trainer.distillation.privileged_context import (
    build_privileged_sequence,
    slice_privileged_teacher_to_student,
)


def test_build_privileged_sequence_layout():
    seq = build_privileged_sequence(
        prompt_ids=[1, 2],
        response_ids=[9, 10],
        solution_ids=[5, 6, 7],
        prefix_ids=[100],
        suffix_ids=[200, 201],
    )
    assert seq == [1, 2, 100, 5, 6, 7, 200, 201, 9, 10]
    # the response is the suffix, exactly as in a plain prompt+response teacher input
    assert seq[-2:] == [9, 10]


def test_build_privileged_sequence_empty_markers():
    assert build_privileged_sequence([1], [9], [5], [], []) == [1, 5, 9]


def test_slice_privileged_teacher_keeps_response_pads_prompt():
    priv_len, k = 10, 2
    teacher_ids = torch.arange(priv_len * k).reshape(priv_len, k)
    teacher_logprobs = torch.randn(priv_len, k)

    ids, logprobs = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=2, response_length=2, pad_token_id=0
    )

    # aligned to student prompt(2) + response(2)
    assert ids.shape == (4, k) and logprobs.shape == (4, k)
    # response rows == the teacher's last response_length rows (privileged scores)
    assert torch.equal(ids[-2:], teacher_ids[-2:])
    assert torch.equal(logprobs[-2:], teacher_logprobs[-2:])
    # prompt rows are padded / zeroed (the response mask drops them downstream)
    assert torch.all(ids[:2] == 0)
    assert torch.all(logprobs[:2] == 0.0)


def test_slice_preserves_dtype_and_pad_value():
    teacher_ids = torch.arange(6).reshape(3, 2).long()
    teacher_logprobs = torch.randn(3, 2, dtype=torch.float32)

    ids, logprobs = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=1, response_length=2, pad_token_id=7
    )

    assert ids.dtype == torch.long and logprobs.dtype == torch.float32
    assert ids[0, 0].item() == 7
