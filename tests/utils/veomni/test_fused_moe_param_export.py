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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.workers.engine.veomni import transformer_impl
from verl.workers.engine.veomni.utils import FUSED_MOE_PARAM_MODELS, gather_fused_moe_param

EXPERT_PARAM_NAME = "model.layers.0.mlp.experts.gate_up_proj"


def make_engine_with_expert_param(tensor):
    module = SimpleNamespace(
        config=SimpleNamespace(model_type="gpt_oss", num_local_experts=4),
        state_dict=Mock(return_value={EXPERT_PARAM_NAME: tensor}),
    )
    return SimpleNamespace(
        module=module,
        _uses_fsdp2_cpu_offload_policy=True,
        _is_offload_param=False,
    )


def test_gpt_oss_uses_fused_moe_export():
    assert "gpt_oss" in FUSED_MOE_PARAM_MODELS


def test_gpt_oss_export_keeps_fused_param_without_ep(monkeypatch):
    tensor = torch.arange(8).reshape(2, 4)
    engine = make_engine_with_expert_param(tensor)
    monkeypatch.setattr(
        transformer_impl.parallel_state,
        "get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=False),
    )

    params, peft_config = transformer_impl.VeOmniEngine.get_per_tensor_param(engine)

    assert list(params) == [(EXPERT_PARAM_NAME, tensor)]
    assert peft_config is None


def test_gpt_oss_export_reconstructs_fused_param_with_ep(monkeypatch):
    local_tensor = torch.arange(8).reshape(2, 4)
    full_tensor = torch.arange(16).reshape(4, 4)
    engine = make_engine_with_expert_param(local_tensor)
    ep_group = Mock()
    monkeypatch.setattr(
        transformer_impl.parallel_state,
        "get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=True, ep_size=2, ep_group=ep_group),
    )
    gather = Mock(return_value=full_tensor)
    monkeypatch.setattr(transformer_impl, "gather_fused_moe_param", gather)

    params, _ = transformer_impl.VeOmniEngine.get_per_tensor_param(engine)

    assert list(params) == [(EXPERT_PARAM_NAME, full_tensor)]
    gather.assert_called_once_with(
        local_tensor,
        ep_size=2,
        ep_group=ep_group,
        expected_num_experts=4,
    )


def test_gather_fused_moe_param_reconstructs_expert_dimension(monkeypatch):
    local_tensor = torch.arange(8).reshape(2, 4)
    ep_group = Mock()

    def fake_all_gather_into_tensor(output, tensor, group):
        assert group is ep_group
        torch.cat((tensor, tensor + 8), dim=0, out=output)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)

    output = gather_fused_moe_param(
        local_tensor,
        ep_size=2,
        ep_group=ep_group,
        expected_num_experts=4,
    )

    assert torch.equal(output, torch.cat((local_tensor, local_tensor + 8), dim=0))


def test_gather_fused_moe_param_rejects_inconsistent_expert_count():
    with pytest.raises(ValueError, match=r"local experts \(2\) \* EP size \(2\) = 4, expected 8"):
        gather_fused_moe_param(
            torch.empty(2, 4),
            ep_size=2,
            ep_group=Mock(),
            expected_num_experts=8,
        )
