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

"""Pairwise metadata must keep batch order and include every sample."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager.pairwise_comparison import PairwiseComparisonRewardManager


@pytest.fixture
def manager():
    # Avoid constructing an API client: only batch assembly is under test.
    manager = object.__new__(PairwiseComparisonRewardManager)
    manager.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: f"#### {int(ids[0])}")

    async def process_group(data, uid, indices):
        if len(indices) == 1:
            return uid, indices, None, None, None
        return (
            uid,
            indices,
            {0: 0.9, 1: 0.1},
            [1.0, 0.0],
            {
                "win_rate": 0.5,
                "win_count": 1,
                "total_count": 2,
                "win_indices": [0],
            },
        )

    manager._process_group_async = process_group
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield manager
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _batch(uids, *, validation=False):
    size = len(uids)
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(size, 1, dtype=torch.long),
            "responses": torch.arange(10, 10 + size, dtype=torch.long).unsqueeze(-1).repeat(1, 2),
            "attention_mask": torch.ones(size, 3, dtype=torch.long),
        },
        non_tensors={"uid": np.array(uids, dtype=object)},
        meta_info={"validate": validation},
    )


def test_interleaved_groups_keep_original_sample_order(manager):
    batch = _batch(["a", "b", "a", "b"])
    result = manager(batch, return_dict=True)
    extra = result["reward_extra_info"]
    assert extra["answer_correct"] == [True, True, False, False]
    assert extra["extracted_answer"] == ["10", "11", "12", "13"]
    assert extra["tournament_score"] == [0.9, 0.9, 0.1, 0.1]
    assert extra["reward_value"] == [1.0, 1.0, 0.0, 0.0]
    torch.testing.assert_close(result["reward_tensor"][:, -1], torch.tensor(extra["reward_value"]))


def test_singleton_group_has_full_length_metadata(manager):
    batch = _batch(["a", "solo", "a"])
    extra = manager(batch, return_dict=True)["reward_extra_info"]
    assert all(len(values) == len(batch) for values in extra.values())
    assert extra["reward_value"] == [1.0, 1.0, 0.0]
    # This is the trainer's metadata attachment contract.
    batch.non_tensor_batch.update({key: np.array(values) for key, values in extra.items()})
    batch.check_consistency()


def test_validation_missing_answers_and_empty_responses_keep_rows(manager):
    batch = _batch(["a", "b", "c"], validation=True)
    batch.non_tensor_batch["reward_model"] = np.array(
        [{"ground_truth": "10"}, {}, {"ground_truth": "12"}], dtype=object
    )
    batch.batch["attention_mask"][2, 1:] = 0
    extra = manager(batch, return_dict=True)["reward_extra_info"]
    assert all(len(values) == len(batch) for values in extra.values())
    assert extra["answer_correct"] == [True, False, False]
    assert extra["extracted_answer"] == ["10", "", ""]
    assert extra["validation_total"] == [1, 1, 1]
