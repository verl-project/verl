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

import pytest
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded response"


def _make_data_proto() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "decoded response"}],
        },
    )


def _make_config() -> DictConfig:
    return DictConfig({"trainer": {"experiment_name": "gdpo-test"}})


@pytest.fixture(autouse=True)
def reset_gdpo_reward_manager_state():
    GDPORewardManager._class_initialized = False
    yield
    GDPORewardManager._class_initialized = False


@pytest.mark.asyncio
async def test_gdpo_reward_manager_forwards_arbitrary_reward_components():
    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        assert data_source == "test"
        assert solution_str == ground_truth
        assert extra_info["experiment_name"] == "gdpo-test"
        return {
            "score": 1.0,
            "format_reward": 0.5,
            "accuracy_reward": 1.0,
            "length_reward": -0.25,
        }

    manager = GDPORewardManager(config=_make_config(), tokenizer=DummyTokenizer(), compute_score=compute_score)
    result = await manager.run_single(_make_data_proto())

    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"] == {
        "score": 1.0,
        "format_reward": 0.5,
        "accuracy_reward": 1.0,
        "length_reward": -0.25,
    }


@pytest.mark.asyncio
async def test_gdpo_reward_manager_requires_score_for_dict_results():
    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        return {"format_reward": 1.0}

    manager = GDPORewardManager(config=_make_config(), tokenizer=DummyTokenizer(), compute_score=compute_score)
    with pytest.raises(KeyError, match="must include a scalar 'score'"):
        await manager.run_single(_make_data_proto())

