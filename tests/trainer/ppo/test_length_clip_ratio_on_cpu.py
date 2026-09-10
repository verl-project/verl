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
"""Tests that the length clip_ratio metrics measure against the configured caps.

V0 pads every sample to the configured length before stacking, so the tensor width
equals the configured cap. V1 stores jagged tensors and pads to the *batch* maximum,
so the width is data dependent and cannot be used as the truncation threshold.
"""

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics


def _make_v1_batch(prompt_lengths, response_lengths):
    """Build a batch shaped the way the V1 trainer produces it.

    ``prompts``/``responses`` are padded to the batch maximum (not to the configured
    cap) and the true lengths are carried in explicit ``prompt_length`` /
    ``response_length`` columns, mirroring ``to_padded_tensor()`` in
    ``verl/trainer/ppo/v1/trainer_base.py``.
    """
    bsz = len(response_lengths)
    padded_prompt_width = max(prompt_lengths)
    padded_response_width = max(response_lengths)

    response_mask = torch.zeros((bsz, padded_response_width), dtype=torch.int64)
    for i, length in enumerate(response_lengths):
        response_mask[i, :length] = 1

    tensors = {
        "prompts": torch.zeros((bsz, padded_prompt_width), dtype=torch.int64),
        "responses": torch.zeros((bsz, padded_response_width), dtype=torch.int64),
        "response_mask": response_mask,
        "token_level_scores": torch.zeros((bsz, padded_response_width)),
        "token_level_rewards": torch.zeros((bsz, padded_response_width)),
        "advantages": torch.zeros((bsz, padded_response_width)),
        "returns": torch.zeros((bsz, padded_response_width)),
        "prompt_length": torch.tensor(prompt_lengths, dtype=torch.float32),
        "response_length": torch.tensor(response_lengths, dtype=torch.float32),
    }
    return DataProto.from_dict(tensors=tensors)


def _make_v0_batch(prompt_lengths, response_lengths, max_prompt_length, max_response_length):
    """Build a batch shaped the way V0 produces it: padded to the configured caps.

    There are no ``prompt_length``/``response_length`` columns, so
    ``_compute_response_info`` derives the true lengths from ``attention_mask``.
    """
    bsz = len(response_lengths)

    prompt_mask = torch.zeros((bsz, max_prompt_length), dtype=torch.int64)
    response_mask = torch.zeros((bsz, max_response_length), dtype=torch.int64)
    for i, length in enumerate(prompt_lengths):
        # prompts are left padded
        prompt_mask[i, max_prompt_length - length :] = 1
    for i, length in enumerate(response_lengths):
        # responses are right padded
        response_mask[i, :length] = 1

    tensors = {
        "prompts": torch.zeros((bsz, max_prompt_length), dtype=torch.int64),
        "responses": torch.zeros((bsz, max_response_length), dtype=torch.int64),
        "attention_mask": torch.cat([prompt_mask, response_mask], dim=-1),
        "response_mask": response_mask,
        "token_level_scores": torch.zeros((bsz, max_response_length)),
        "token_level_rewards": torch.zeros((bsz, max_response_length)),
        "advantages": torch.zeros((bsz, max_response_length)),
        "returns": torch.zeros((bsz, max_response_length)),
    }
    return DataProto.from_dict(tensors=tensors)


def test_v1_clip_ratio_is_zero_when_nothing_is_truncated():
    # Caps are 8/6, nothing reaches them. V1 pads prompts to 5 and responses to 4,
    # so comparing against the tensor width would report 0.5 for both.
    batch = _make_v1_batch(prompt_lengths=[3, 5, 5, 2], response_lengths=[2, 3, 4, 4])

    metrics = compute_data_metrics(
        batch,
        use_critic=False,
        max_prompt_length=8,
        max_response_length=6,
    )

    assert metrics["response_length/clip_ratio"] == 0.0
    assert metrics["response_length_non_aborted/clip_ratio"] == 0.0
    assert metrics["prompt_length/clip_ratio"] == 0.0
    # the mean/max/min statistics are computed over true lengths and stay unchanged
    assert metrics["response_length/max"] == 4.0
    assert metrics["prompt_length/max"] == 5.0


def test_v1_clip_ratio_reports_the_truncated_fraction():
    # Responses: two of four samples hit the cap of 6, one is aborted (length 0).
    # Prompts: nothing reaches the cap of 8 even though the padded width is 4.
    batch = _make_v1_batch(prompt_lengths=[4, 4, 3, 4], response_lengths=[6, 3, 0, 6])

    metrics = compute_data_metrics(
        batch,
        use_critic=False,
        max_prompt_length=8,
        max_response_length=6,
    )

    assert metrics["response_length/clip_ratio"] == 0.5
    # non-aborted samples are [6, 3, 6] -> 2/3 truncated
    assert metrics["response_length_non_aborted/clip_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["response/aborted_ratio"] == 0.25
    assert metrics["prompt_length/clip_ratio"] == 0.0


def test_v0_shape_is_unchanged_by_the_fallback():
    # V0 pads to the configured caps, so width == cap and the fallback must reproduce
    # exactly what is reported today.
    batch = _make_v0_batch(
        prompt_lengths=[3, 5],
        response_lengths=[4, 2],
        max_prompt_length=5,
        max_response_length=4,
    )

    fallback = compute_data_metrics(batch, use_critic=False)

    assert fallback["response_length/clip_ratio"] == 0.5
    assert fallback["response_length_non_aborted/clip_ratio"] == 0.5
    assert fallback["prompt_length/clip_ratio"] == 0.5

    # passing the caps explicitly must agree with the fallback when width == cap
    explicit = compute_data_metrics(
        batch,
        use_critic=False,
        max_prompt_length=5,
        max_response_length=4,
    )
    for key in (
        "response_length/clip_ratio",
        "response_length_non_aborted/clip_ratio",
        "prompt_length/clip_ratio",
    ):
        assert explicit[key] == fallback[key]
