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
from torch import nn

from verl.workers.config.optimizer import (
    WEIGHT_DECAY_SCALE_KEY,
    FSDPOptimizerConfig,
    build_optimizer,
)


class _CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4)
        self.layer_norm = nn.LayerNorm(4)
        self.rms = _CustomRMSNorm(4)
        self.bn = nn.BatchNorm1d(4)
        self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)


def _parameter_names_by_weight_decay_scale(model, optimizer):
    name_by_parameter_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        group.get(WEIGHT_DECAY_SCALE_KEY, 1.0): {name_by_parameter_id[id(parameter)] for parameter in group["params"]}
        for group in optimizer.param_groups
    }


def test_standard_weight_decay_policy_groups_parameters():
    model = _ToyModel()
    optimizer = build_optimizer(
        model,
        FSDPOptimizerConfig(lr=0.1, weight_decay=0.2),
    )

    names_by_scale = _parameter_names_by_weight_decay_scale(model, optimizer)
    assert names_by_scale[1.0] == {"embedding.weight", "linear.weight"}
    assert names_by_scale[0.0] == {
        "linear.bias",
        "layer_norm.weight",
        "layer_norm.bias",
        "rms.weight",
        "bn.weight",
        "bn.bias",
    }
    assert "frozen" not in names_by_scale[1.0] | names_by_scale[0.0]
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_standard_policy_applies_weight_decay_only_to_decay_group():
    model = _ToyModel()
    optimizer = build_optimizer(
        model,
        FSDPOptimizerConfig(lr=0.1, weight_decay=0.2),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    names_by_scale = _parameter_names_by_weight_decay_scale(model, optimizer)
    for name, parameter in model.named_parameters():
        if name in names_by_scale[1.0]:
            assert not torch.equal(parameter, before[name])
        else:
            assert torch.equal(parameter, before[name])


def test_optimizer_override_sets_effective_decay_group_value():
    model = _ToyModel()
    optimizer = build_optimizer(
        model,
        FSDPOptimizerConfig(
            lr=0.1,
            weight_decay=0.2,
            override_optimizer_config={"weight_decay": 0.3},
        ),
    )

    weight_decay_by_scale = {group[WEIGHT_DECAY_SCALE_KEY]: group["weight_decay"] for group in optimizer.param_groups}
    assert weight_decay_by_scale == {1.0: pytest.approx(0.3), 0.0: pytest.approx(0.0)}


def test_standard_optimizer_state_dict_round_trip():
    model = _ToyModel()
    config = FSDPOptimizerConfig(lr=0.1, weight_decay=0.2)
    optimizer = build_optimizer(model, config)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    restored_model = _ToyModel()
    restored_optimizer = build_optimizer(restored_model, config)
    restored_optimizer.load_state_dict(optimizer.state_dict())

    assert [group[WEIGHT_DECAY_SCALE_KEY] for group in restored_optimizer.param_groups] == [1.0, 0.0]
    assert [group["weight_decay"] for group in restored_optimizer.param_groups] == pytest.approx([0.2, 0.0])


def test_legacy_all_policy_preserves_single_parameter_group():
    model = _ToyModel()
    optimizer = build_optimizer(
        model.parameters(),
        FSDPOptimizerConfig(lr=0.1, weight_decay=0.2, weight_decay_policy="all"),
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert WEIGHT_DECAY_SCALE_KEY not in optimizer.param_groups[0]
    assert set(optimizer.param_groups[0]["params"]) == set(model.parameters())


def test_standard_policy_requires_model_for_parameter_classification():
    model = _ToyModel()

    with pytest.raises(ValueError, match="requires the model itself"):
        build_optimizer(model.parameters(), FSDPOptimizerConfig(lr=0.1))
