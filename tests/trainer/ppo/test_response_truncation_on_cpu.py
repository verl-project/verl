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
"""Response clip metrics must count backend limits below the padded width."""

import math

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics


def _batch(lengths, flags, *, v1=False):
    width = max(lengths) if v1 else 8
    prompt_lengths = torch.tensor([2, 3, 4][: len(lengths)])
    response_lengths = torch.tensor(lengths)
    rmask = (torch.arange(width)[None, :] < response_lengths[:, None]).long()
    pmask = (torch.arange(4)[None, :] < prompt_lengths[:, None]).long()
    tensors = {
        "prompts": pmask.clone(),
        "responses": rmask.clone(),
        "attention_mask": torch.cat([pmask, rmask], dim=-1),
        "response_mask": rmask,
        **{key: rmask.float() for key in ("token_level_scores", "token_level_rewards", "advantages", "returns")},
    }
    if v1:
        tensors.update(prompt_length=prompt_lengths.float(), response_length=response_lengths.float())
    non_tensors = {} if flags is None else {"response_truncated": np.array(flags, dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


@pytest.mark.parametrize("v1", [False, True])
@pytest.mark.parametrize(
    "lengths,flags,ratio,non_aborted_ratio",
    [
        ([6, 5, 4], [True, True, True], 1.0, 1.0),
        ([6, 5, 4], [False, False, False], 0.0, 0.0),
        ([6, 5, 0], [True, False, False], 1 / 3, 1 / 2),
        ([0, 0, 0], [False, False, False], 0.0, float("nan")),
        ([8, 2, 4], [False, True, None], 1 / 3, 1 / 3),
    ],
)
def test_explicit_truncation_flags(lengths, flags, ratio, non_aborted_ratio, v1):
    metrics = compute_data_metrics(_batch(lengths, flags, v1=v1), use_critic=False)
    assert metrics["response_length/clip_ratio"] == pytest.approx(ratio)
    actual = metrics["response_length_non_aborted/clip_ratio"]
    if math.isnan(non_aborted_ratio):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(non_aborted_ratio)


@pytest.mark.parametrize("flags", [None, [None, None, None]])
def test_legacy_batches_keep_width_fallback(flags):
    metrics = compute_data_metrics(_batch([8, 2, 0], flags), use_critic=False)
    assert metrics["response_length/clip_ratio"] == pytest.approx(1 / 3)
    assert metrics["response_length_non_aborted/clip_ratio"] == pytest.approx(1 / 2)


def test_truncation_flags_follow_batch_selection():
    batch = _batch([6, 5, 4], [True, False, True]).select_idxs([1, 0])
    metrics = compute_data_metrics(batch, use_critic=False)
    assert metrics["response_length/clip_ratio"] == pytest.approx(1 / 2)
