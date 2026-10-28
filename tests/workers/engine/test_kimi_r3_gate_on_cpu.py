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

from types import SimpleNamespace

import pytest
import torch
from fsdp_turbo.models.kimi.kimi_r3_utils import align_kimi_full_model_routes, prepare_kimi_routing_replay
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl.trainer.ppo.padding_utils import build_padding_routed_experts
from verl.workers.config.engine import EngineRouterReplayConfig, FSDPEngineConfig


@pytest.mark.parametrize("mode", ["disabled", "R3"])
def test_fsdp_routing_replay_config_accepts_disabled_and_r3(mode):
    config = FSDPEngineConfig(router_replay=EngineRouterReplayConfig(mode=mode))
    assert config.router_replay.mode == mode


def test_fsdp_routing_replay_config_rejects_record_only_mode():
    with pytest.raises(ValueError, match="only 'disabled' and 'R3'"):
        FSDPEngineConfig(router_replay=EngineRouterReplayConfig(mode="R2"))


def test_model_row_alignment_preserves_ids_and_mask():
    routes = torch.arange(12).reshape(1, 3, 2, 2)
    aligned_ids, aligned_mask = align_kimi_full_model_routes(
        routes, torch.ones(1, 3, dtype=torch.bool), torch.tensor([[False, True, True, True, True]])
    )
    torch.testing.assert_close(aligned_ids[:, 1:4], routes)
    assert aligned_mask.tolist() == [[False, True, True, True, False]]
    assert not bool(aligned_ids[:, (0, 4)].any())


def test_padding_reuses_a_source_causal_route_row():
    source_ids = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [0, 3]]], dtype=torch.uint8)
    padding_ids = build_padding_routed_experts(source_ids, seq_len=1)
    torch.testing.assert_close(padding_ids[0], source_ids[0])
    aligned_ids, aligned_mask = align_kimi_full_model_routes(
        padding_ids.unsqueeze(0), torch.tensor([[True]]), torch.tensor([[True, True]])
    )
    assert aligned_mask.tolist() == [[True, False]]
    torch.testing.assert_close(aligned_ids[:, :1], padding_ids.unsqueeze(0))


@pytest.mark.parametrize("nested", [False, True])
def test_prepare_model_rows_only_ids_and_mask(nested):
    samples = [torch.tensor([[[0, 1], [2, 3]]]).expand(3, -1, -1), torch.tensor([[[1, 2], [0, 3]]]).expand(2, -1, -1)]
    routes = torch.nested.as_nested_tensor(samples, layout=torch.jagged) if nested else samples[0].unsqueeze(0)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]) if nested else torch.ones(1, 4)
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2, num_experts=4, num_experts_per_token=2))
    ids, mask = prepare_kimi_routing_replay(routes, None, attention_mask, config, "model_rows")
    torch.testing.assert_close(ids[0, :3], samples[0])
    assert mask.tolist() == (
        [[True, True, True, False], [True, True, False, False]] if nested else [[True, True, True, False]]
    )


def test_agent_loop_export_reaches_fsdp_replay_without_renaming_metadata():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2, num_experts=4, num_experts_per_token=2))
    ids = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [0, 3]]], dtype=torch.uint8)
    exported = AgentLoopOutput(
        prompt_ids=[11, 12],
        response_ids=[13],
        response_mask=[1],
        routed_experts=ids,
        routed_experts_source_semantics="model_rows",
        metrics=AgentLoopMetrics(),
    ).as_dict()
    # Batch the actual producer fields unchanged, including its metadata key.
    batch = TensorDict(
        {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
            for key, value in exported.items()
            if key.startswith("routed_experts")
        },
        batch_size=[1],
    )
    from fsdp_turbo.models.kimi.kimi_k3_cp import kimi_k3_model_forward

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    batch["input_ids"] = torch.nested.as_nested_tensor([torch.tensor([11, 12, 13])], layout=torch.jagged)
    batch["position_ids"] = torch.nested.as_nested_tensor([torch.arange(3)], layout=torch.jagged)
    batch["temperature"] = torch.ones(1)
    tu.assign_non_tensor(batch, enable_routing_replay=True, use_remove_padding=False)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = None
    engine.module = SimpleNamespace(forward=kimi_k3_model_forward)
    model_inputs, _ = engine.prepare_model_inputs(batch)
    assert model_inputs["output_raw_logits"] is True
    actual, mask = prepare_kimi_routing_replay(
        model_inputs["routed_experts"],
        None,
        model_inputs["attention_mask"],
        config,
        model_inputs["routed_experts_transport_semantics"],
    )
    torch.testing.assert_close(actual[0, :2], ids.long())
    assert mask.tolist() == [[True, True, False]]
    with pytest.raises(ValueError, match="model_rows transport semantics"):
        prepare_kimi_routing_replay(batch["routed_experts"], None, torch.ones(1, 3), config, None)
    del batch["routed_experts"]
    with pytest.raises(ValueError, match="requires routed_experts"):
        engine.prepare_model_inputs(batch)
    engine.module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="does not accept routing replay inputs"):
        engine.prepare_model_inputs(batch)
