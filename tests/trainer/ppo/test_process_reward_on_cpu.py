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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.config.algorithm import AlgoConfig, ProcessRewardConfig
from verl.trainer.ppo.process_reward import (
    BaseProcessScorer,
    ProcessRewardRunner,
    generated_turn_spans,
    shape_process_advantages,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


def test_process_config_is_disabled_by_default_and_supports_dataclass_conversion():
    assert not ProcessRewardConfig().enabled
    config = omega_conf_to_dataclass(
        OmegaConf.create({"process_reward": {"enabled": True, "mode": "gvpo", "weight": 0.2}}), AlgoConfig
    )
    assert isinstance(config.process_reward, ProcessRewardConfig)
    assert config.process_reward.enabled and config.process_reward.mode == "gvpo"
    assert config.process_reward.weight == 0.2


class TurnScorer(BaseProcessScorer):
    async def evaluate(self, output, span, context):
        return context["turn_rewards"][output.extra_fields["generated_turn_spans"].index(span)]


class DenseScorer(BaseProcessScorer):
    async def evaluate(self, output, span, context):
        return torch.arange(span[1] - span[0], dtype=torch.float32)


class UnexpectedScorer(BaseProcessScorer):
    async def evaluate(self, output, span, context):
        raise AssertionError("turn_rewards must bypass configured scorers")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("rewards", [[-1.0, 0.5], [0.0, 0.0]])
async def test_standard_turn_rewards_take_precedence(configured, rewards):
    output = SimpleNamespace(
        response_ids=list(range(6)),
        response_mask=[1, 1, 0, 1, 1, 0],
        extra_fields={"generated_turn_spans": [(0, 2), (3, 5)], "turn_rewards": rewards},
    )
    runner = ProcessRewardRunner([{"path": f"{__name__}.UnexpectedScorer", "weight": 99}] if configured else [])
    scores = await runner.score(output, {})
    expected = torch.tensor([rewards[0], rewards[0], 0, rewards[1], rewards[1], 0])
    torch.testing.assert_close(scores, expected)
    shaped = shape_process_advantages(
        torch.zeros(6), scores, torch.tensor(output.response_mask), mode="additive", weight=0.1
    )
    torch.testing.assert_close(shaped, expected * 0.1)
    assert output.extra_fields["turn_rewards"] == rewards


@pytest.mark.asyncio
@pytest.mark.parametrize("rewards", [None, 1.0, [], [1.0], [[1.0, 2.0]], [float("nan"), 0], [0, float("inf")]])
async def test_invalid_standard_turn_rewards_rejected_without_fallback(rewards):
    output = SimpleNamespace(
        response_ids=[1, 2],
        response_mask=[1, 1],
        extra_fields={"generated_turn_spans": [(0, 1), (1, 2)], "turn_rewards": rewards},
    )
    runner = ProcessRewardRunner([{"path": f"{__name__}.UnexpectedScorer"}])
    with pytest.raises(ValueError, match="turn_rewards"):
        await runner.score(output, {})


@pytest.mark.asyncio
async def test_missing_turn_rewards_and_scorers_rejected():
    output = SimpleNamespace(response_ids=[1], response_mask=[1], extra_fields={})
    with pytest.raises(ValueError, match="turn_rewards.*scorer"):
        await ProcessRewardRunner([]).score(output, {})


@pytest.mark.asyncio
async def test_empty_trajectory_accepts_empty_turn_rewards():
    output = SimpleNamespace(response_ids=[], response_mask=[], extra_fields={"turn_rewards": []})
    assert (await ProcessRewardRunner([]).score(output, {})).numel() == 0


@pytest.mark.asyncio
async def test_process_rewards_align_to_generated_turns():
    output = SimpleNamespace(
        response_ids=list(range(7)),
        response_mask=[1, 1, 0, 0, 1, 1, 0],
        extra_fields={"generated_turn_spans": [(0, 2), (4, 6)]},
    )
    runner = ProcessRewardRunner(
        [
            {"path": f"{__name__}.TurnScorer", "weight": 1.0},
            {"path": f"{__name__}.DenseScorer", "weight": 0.5},
        ]
    )
    scores = await runner.score(output, {"turn_rewards": [0.25, 0.75]})
    torch.testing.assert_close(scores, torch.tensor([0.25, 0.75, 0.0, 0.0, 0.75, 1.25, 0.0]))


def test_invalid_turn_boundaries_rejected():
    with pytest.raises(ValueError, match="non-generated"):
        generated_turn_spans([1, 0, 1], [(0, 2), (2, 3)])
    with pytest.raises(ValueError, match="cover"):
        generated_turn_spans([1, 0, 1], [(0, 1)])
    with pytest.raises(ValueError, match="ordered"):
        generated_turn_spans([1, 0, 1], [(2, 3), (0, 1)])


def test_advantage_shaping_preserves_mask():
    advantages = torch.tensor([[2.0, -1.0, 0.0, -2.0]])
    mask = torch.tensor([[1, 1, 0, 1]])
    scores = torch.tensor([[0.5, 0.75, 0.0, 0.0]])

    additive = shape_process_advantages(advantages, scores, mask, mode="additive", weight=2.0)
    weighted = shape_process_advantages(advantages, scores, mask, mode="process_weighted")

    torch.testing.assert_close(additive, torch.tensor([[3.0, 0.5, 0.0, -2.0]]))
    torch.testing.assert_close(weighted, torch.tensor([[1.0, 0.75, 0.0, -2.0]]))


def test_weighted_mode_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        shape_process_advantages(
            torch.ones(1, 2), torch.tensor([[0.0, 1.2]]), torch.ones(1, 2), mode="process_weighted"
        )


def test_process_scores_survive_v1_jagged_batch_transport():
    fields = [
        {"response_mask": torch.tensor([1, 0, 1]), "process_scores": torch.tensor([0.3, 0.0, 0.7])},
        {"response_mask": torch.tensor([1]), "process_scores": torch.tensor([0.5])},
    ]
    padded = list_of_dict_to_tensordict(fields).to_padded_tensor()
    torch.testing.assert_close(padded["process_scores"], torch.tensor([[0.3, 0.0, 0.7], [0.5, 0.0, 0.0]]))


def test_gvpo_failure_branches_and_unchanged_success_tokens():
    advantages = torch.tensor([[2.0, 0.0, -2.0, 5e-7, -5e-7, 1e-6, -1e-6, 2.0, -2.0, 9.0]])
    process = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    result = shape_process_advantages(advantages, process, mask, mode="gvpo")
    torch.testing.assert_close(result, torch.tensor([[0.0, -0.4, -2.4, -0.4, -0.4, -0.4, -0.4, 2.0, -2.0, 0.0]]))
    assert advantages[0, 0] == 2.0  # Input is not mutated.


def test_gvpo_uses_failure_sign_instead_of_penalty_magnitude():
    adv = torch.tensor([[1.0, 0.0, -1.0]])
    mask = torch.ones_like(adv)
    a = shape_process_advantages(adv, -torch.ones_like(adv), mask, mode="gvpo", weight=0.1)
    b = shape_process_advantages(adv, -10 * torch.ones_like(adv), mask, mode="gvpo", weight=1.0)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, torch.tensor([[0.0, -0.4, -1.2]]))


@pytest.mark.parametrize("field", ["negative_scale", "zero_penalty", "zero_eps"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_gvpo_invalid_parameters(field, value):
    with pytest.raises(ValueError, match="GVPO"):
        shape_process_advantages(torch.ones(1, 1), -torch.ones(1, 1), torch.ones(1, 1), mode="gvpo", **{field: value})
