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
"""CPU tests for the OPSD privileged-context helpers."""

import numpy as np
import pytest
import torch

from verl.trainer.distillation.privileged_context import (
    build_privileged_sequence,
    resolve_privileged_solution,
    slice_privileged_teacher_to_student,
)


def test_resolve_privileged_solution_nested_dotted_key():
    sk = {"reward_model": {"ground_truth": "72"}}
    assert resolve_privileged_solution(sk, "reward_model.ground_truth") == "72"
    # flat key still works
    assert resolve_privileged_solution({"sol": "x"}, "sol") == "x"


def test_resolve_privileged_solution_missing_returns_none():
    assert resolve_privileged_solution({"a": 1}, "reward_model.ground_truth") is None
    assert resolve_privileged_solution(None, "x") is None
    assert resolve_privileged_solution({"g": ""}, "g") is None


def test_resolve_privileged_solution_normalizes_scalar_array_list():
    assert resolve_privileged_solution({"g": np.array("72")}, "g") == "72"  # 0-d -> item
    assert resolve_privileged_solution({"g": np.array(["x"])}, "g") == "x"  # 1-elem array
    assert resolve_privileged_solution({"g": ["a", "b"]}, "g") == "a\nb"  # list -> joined


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


def test_build_privileged_sequence_insert_before_marker():
    # marker = [8] (e.g. an assistant-turn opener); solution block goes before its
    # last occurrence, the response after the original prompt tail.
    seq = build_privileged_sequence(
        prompt_ids=[1, 8, 2, 8],
        response_ids=[9],
        solution_ids=[5],
        prefix_ids=[100],
        suffix_ids=[200],
        insert_before_token_ids=[8],
    )
    assert seq == [1, 8, 2, 100, 5, 200, 8, 9]


def test_build_privileged_sequence_insert_marker_not_found_falls_back():
    seq = build_privileged_sequence([1, 2], [9], [5], [100], [200], insert_before_token_ids=[42])
    assert seq == [1, 2, 100, 5, 200, 9]  # append fallback


def test_slice_privileged_teacher_keeps_response_pads_prompt():
    priv_len, k = 10, 2
    teacher_ids = torch.arange(priv_len * k).reshape(priv_len, k)
    teacher_logprobs = torch.randn(priv_len, k)

    ids, logprobs = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=2, response_length=2, pad_token_id=0
    )

    assert ids.shape == (4, k) and logprobs.shape == (4, k)
    assert torch.equal(ids[-2:], teacher_ids[-2:])
    assert torch.equal(logprobs[-2:], teacher_logprobs[-2:])
    assert torch.all(ids[:2] == 0)
    assert torch.all(logprobs[:2] == 0.0)


def test_slice_response_length_zero_is_empty_not_whole_tensor():
    # regression: teacher_ids[-0:] would return the whole tensor; an empty
    # response must yield only the padded prompt rows.
    teacher_ids = torch.arange(12).reshape(6, 2)
    teacher_logprobs = torch.randn(6, 2)
    ids, logprobs = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=3, response_length=0, pad_token_id=0
    )
    assert ids.shape == (3, 2) and logprobs.shape == (3, 2)
    assert torch.all(ids == 0)


def test_slice_pad_token_id_none_falls_back_to_zero():
    teacher_ids = torch.arange(6).reshape(3, 2)
    teacher_logprobs = torch.randn(3, 2)
    ids, _ = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=1, response_length=2, pad_token_id=None
    )
    assert ids[0, 0].item() == 0


def test_slice_preserves_dtype_and_pad_value():
    teacher_ids = torch.arange(6).reshape(3, 2).long()
    teacher_logprobs = torch.randn(3, 2, dtype=torch.float32)
    ids, logprobs = slice_privileged_teacher_to_student(
        teacher_ids, teacher_logprobs, student_prompt_length=1, response_length=2, pad_token_id=7
    )
    assert ids.dtype == torch.long and logprobs.dtype == torch.float32
    assert ids[0, 0].item() == 7


def test_slice_negative_response_length_raises():
    with pytest.raises(ValueError):
        slice_privileged_teacher_to_student(torch.arange(6).reshape(3, 2), torch.randn(3, 2), 1, -1, 0)
