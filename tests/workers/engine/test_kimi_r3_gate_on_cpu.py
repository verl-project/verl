# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Standard token-axis rollout IDs reach FSDP without model-row metadata."""

from types import SimpleNamespace

import pytest
import torch
from fsdp_turbo.models.kimi.kimi_r3_utils import prepare_kimi_routing_replay
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.workers.config.engine import EngineRouterReplayConfig, FSDPEngineConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def test_fsdp_routing_replay_config_rejects_record_only_mode():
    with pytest.raises(ValueError, match="only 'disabled' and 'R3'"):
        FSDPEngineConfig(router_replay=EngineRouterReplayConfig(mode="R2"))


@pytest.mark.parametrize("remove_padding", [False, True])
@pytest.mark.parametrize("has_response", [False, True])
def test_standard_agent_loop_ids_reach_native_replay_on_the_same_token_axis(remove_padding, has_response):
    samples = [([11, 1, 1, 12], [13]), ([21, 1], [23, 24])]
    exports, token_ids = [], []
    for prompts, responses in samples:
        tokens = torch.tensor(prompts + responses)
        routes = torch.tensor([[[0, 1], [2, 3]]]).expand(len(tokens) - 1, -1, -1).numpy()
        exported = AgentLoopOutput(
            prompt_ids=prompts,
            response_ids=responses,
            response_mask=[int(has_response)] * len(responses),
            routed_experts=routes,
            metrics=AgentLoopMetrics(),
        ).as_dict()
        assert exported["routed_experts"].shape[0] == len(tokens)
        assert not any("semantics" in key for key in exported)
        exports.append(exported)
        token_ids.append(tokens)
    batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(token_ids, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(
                [torch.arange(len(t)) for t in token_ids], layout=torch.jagged
            ),
            "routed_experts": torch.nested.as_nested_tensor(
                [e["routed_experts"] for e in exports], layout=torch.jagged
            ),
            "response_mask": torch.tensor([[int(has_response), 0], [int(has_response), int(has_response)]]),
            "temperature": torch.ones(2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, enable_routing_replay=True, use_remove_padding=remove_padding)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = None
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.module = torch.nn.Linear(2, 2)
    inputs, _ = engine.prepare_model_inputs(batch)
    assert inputs["routed_experts"].shape[:2] == inputs["input_ids"].shape
    mask = inputs["router_replay_mask"]
    expected = torch.tensor([[has_response] * 4 + [False], [has_response] * 3 + [False, False]])
    if remove_padding:
        expected = torch.cat((expected[0], expected[1, :4])).reshape(1, -1)
    torch.testing.assert_close(mask, expected)
    attention = inputs["attention_mask"]
    if attention is None:
        attention = torch.ones_like(inputs["input_ids"])
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2, num_experts=4, num_experts_per_token=2))
    ids, actual_mask = prepare_kimi_routing_replay(inputs["routed_experts"], mask, attention, config)
    torch.testing.assert_close(actual_mask, expected)
    torch.testing.assert_close(ids[actual_mask], torch.tensor([[0, 1], [2, 3]]).expand(int(expected.sum()), -1, -1))
    assert not bool(ids[~actual_mask].any()) or not has_response
