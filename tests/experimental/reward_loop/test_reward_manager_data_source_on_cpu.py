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

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "reasoning #### 42"


def _make_reward_batch(data_source_marker):
    non_tensors = {
        "reward_model": np.array([{"style": "rule", "ground_truth": "42"}], dtype=object),
        "extra_info": np.array([{"split": "train", "index": 7}], dtype=object),
    }
    if data_source_marker != "missing":
        non_tensors["data_source"] = np.array([data_source_marker], dtype=object)

    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 4, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
        },
        non_tensors=non_tensors,
    )


@pytest.mark.parametrize(
    ("data_source_marker", "reason"),
    [
        ("missing", "missing"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
@pytest.mark.parametrize("manager_cls", [NaiveRewardManager, RateLimitedRewardManager])
def test_reward_manager_rejects_missing_or_empty_data_source(manager_cls, data_source_marker, reason):
    called = False

    def compute_score(**kwargs):
        nonlocal called
        called = True
        return 1.0

    manager = manager_cls(config=None, tokenizer=FakeTokenizer(), compute_score=compute_score)
    batch = _make_reward_batch(data_source_marker)

    with pytest.raises(ValueError) as exc_info:
        manager.loop.run_until_complete(manager.run_single(batch))

    message = str(exc_info.value)
    assert f"Reward data source is {reason}" in message
    assert "openai/gsm8k" in message
    assert "examples/data_preprocess/gsm8k.py" in message
    assert "reward.custom_reward_function.path" in message
    assert "split='train'" in message
    assert "index=7" in message
    assert not called
