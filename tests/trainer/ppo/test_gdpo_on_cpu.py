# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU coverage for GDPO advantage and reward components."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_gdpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.utils.reward_score import rlla


def _gdpo_inputs():
    token_rewards = torch.zeros((4, 2), dtype=torch.float32)
    response_mask = torch.ones((4, 2), dtype=torch.float32)
    index = np.asarray(["a", "a", "b", "b"], dtype=object)
    non_tensor_batch = {
        "accuracy_reward": np.asarray([1.0, 0.0, 0.0, 1.0]),
        "format_reward": np.asarray([0.0, 1.0, 0.0, 1.0]),
    }
    batch = {
        "prompts": torch.zeros((4, 2), dtype=torch.long),
        "attention_mask": torch.ones((4, 4), dtype=torch.long),
    }
    config = {"gdpo_reward_keys": ["accuracy_reward", "format_reward"]}
    return token_rewards, response_mask, index, non_tensor_batch, batch, config


def test_gdpo_normalizes_each_reward_dimension_before_sum() -> None:
    inputs = _gdpo_inputs()
    advantages, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=inputs[0],
        response_mask=inputs[1],
        index=inputs[2],
        non_tensor_batch=inputs[3],
        batch=inputs[4],
        config=inputs[5],
    )

    torch.testing.assert_close(advantages, returns)
    torch.testing.assert_close(advantages[0], torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[1], torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.all(advantages[2] < 0)
    assert torch.all(advantages[3] > 0)
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_gdpo_requires_declared_reward_components() -> None:
    inputs = _gdpo_inputs()
    with pytest.raises(AssertionError, match="format_reward"):
        compute_gdpo_outcome_advantage(
            token_level_rewards=inputs[0],
            response_mask=inputs[1],
            index=inputs[2],
            non_tensor_batch={"accuracy_reward": inputs[3]["accuracy_reward"]},
            batch=inputs[4],
            config=inputs[5],
        )


def test_gdpo_registries() -> None:
    assert get_adv_estimator_fn("gdpo") is compute_gdpo_outcome_advantage
    assert get_adv_estimator_fn(AdvantageEstimator.GDPO) is compute_gdpo_outcome_advantage
    assert get_reward_manager_cls("gdpo").__name__ == "GDPORewardManager"


def test_rlla_reward_returns_total_and_decoupled_components(monkeypatch) -> None:
    monkeypatch.setattr(random, "randint", lambda *_args: 2)
    answer = '<think>Call the tool.</think>\n<tool_call>\n{"name":"lookup","parameters":{"id":1}}\n</tool_call>'
    result = rlla.compute_score(
        data_source="rlla",
        solution_str=answer,
        ground_truth=answer,
        extra_info={"experiment_name": "qwen2_5_1_5b_gdpo"},
    )

    assert result == {
        "score": 4.0,
        "format_reward": 1.0,
        "accuracy_reward": 3.0,
    }
