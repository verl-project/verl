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

"""Standard ID-only RouterReplay uses current gate probabilities and gradients."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture
def functions():
    external = {
        f"megatron.core.transformer.{name}": Mock()
        for name in ("moe.moe_utils", "moe.token_dispatcher", "moe.router", "transformer_config")
    }
    path = Path(__file__).resolve().parents[3] / "verl/utils/megatron/router_replay_patch.py"
    spec = importlib.util.spec_from_file_location("router_replay_cpu", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, external):
        spec.loader.exec_module(module)
    return vars(module)


def test_backward_replay_is_fifo_and_masks_select_native_routes(functions):
    replay = functions["RouterReplay"]()
    for ids, mask in (([[2, 0]], [True]), ([[0, 2]], [False]), ([[0, 1]], [True])):
        replay.set_target_indices(torch.tensor(ids), torch.tensor(mask))
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_BACKWARD)
    scores = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)

    def native_topk(value, k, **kwargs):
        return value.topk(k)

    for expected_ids in ([[2, 0]], [[1, 2]], [[0, 1]], [[1, 2]]):
        probs, ids = replay.get_replay_topk(scores, 2, default_compute_topk=native_topk)
        assert ids.tolist() == expected_ids
        torch.testing.assert_close(probs, scores.gather(1, torch.tensor(expected_ids)))
    replay.clear_indices()
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_FORWARD)
    probs, ids = replay.get_replay_topk(scores, 2, default_compute_topk=native_topk)
    torch.testing.assert_close(ids, scores.topk(2).indices)
    torch.testing.assert_close(probs, scores.topk(2).values)


@pytest.mark.parametrize("score_function,use_pre_softmax", [("softmax", False), ("softmax", True), ("sigmoid", False)])
def test_replayed_ids_use_current_gate_and_change_after_optimizer_steps(functions, score_function, use_pre_softmax):
    replay = functions["RouterReplay"]()
    ids = torch.tensor([[0, 2]])
    replay.set_target_indices(ids, torch.tensor([True]))
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_FORWARD)
    logits = torch.nn.Parameter(torch.tensor([[0.2, 0.3, -0.4]]))
    optimizer = torch.optim.SGD([logits], lr=0.5)
    outputs = []
    for _ in range(3):
        optimizer.zero_grad()
        probs, route_map = functions["_patched_topk_routing_with_score_function"](
            logits, 2, use_pre_softmax, None, None, score_function, None, False, replay, 1.5
        )
        assert route_map.tolist() == [[True, False, True]]
        if score_function == "sigmoid":
            current = logits.sigmoid().gather(1, ids)
            expected = current / current.sum(-1, keepdim=True)
        elif use_pre_softmax:
            expected = logits.softmax(-1).gather(1, ids)
        else:
            expected = logits.gather(1, ids).softmax(-1)
        torch.testing.assert_close(probs.gather(1, ids), expected * 1.5)
        outputs.append(probs.detach().clone())
        loss = probs[0, 0]
        actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad((expected * 1.5)[0, 0], logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        logits.grad = actual_grad
        optimizer.step()
    assert not torch.allclose(outputs[0], outputs[-1])
