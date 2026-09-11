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
"""``AgentLoopOutput.as_dict`` keeps int16 routing and pads the tail with the sentinel."""

from __future__ import annotations

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.utils.routed_experts import ROUTER_REPLAY_UNRECORDED

LAYERS, TOPK = 4, 2


def _output(routed_experts, prompt_len=3, response_len=4):
    return AgentLoopOutput(
        prompt_ids=list(range(1, prompt_len + 1)),
        response_ids=list(range(1, response_len + 1)),
        response_mask=[1] * response_len,
        metrics=AgentLoopMetrics(),
        extra_fields={},
        routed_experts=routed_experts,
    )


def test_as_dict_casts_to_int16_and_pads_unrecorded_tail():
    n_real = 5
    experts = np.arange(n_real * LAYERS * TOPK, dtype=np.int32).reshape(n_real, LAYERS, TOPK) % 16
    routed = _output(experts).as_dict()["routed_experts"]

    assert routed.dtype == torch.int16
    assert routed.shape == (7, LAYERS, TOPK)
    assert torch.equal(routed[:n_real], torch.as_tensor(experts, dtype=torch.int16))
    assert bool((routed[n_real:] == ROUTER_REPLAY_UNRECORDED).all())


def test_as_dict_accepts_readonly_frombuffer():
    n_real = 4
    payload = np.arange(n_real * LAYERS * TOPK, dtype=np.int32).tobytes()
    experts = np.frombuffer(payload, dtype=np.int32).reshape(n_real, LAYERS, TOPK)
    assert not experts.flags.writeable

    routed = _output(experts).as_dict()["routed_experts"]
    assert routed.dtype == torch.int16
    routed[0, 0, 0] = 123
    assert int(np.frombuffer(payload, dtype=np.int32)[0]) == 0


def test_as_dict_truncates_overlong_capture():
    prompt_len, response_len = 3, 4
    n_real = prompt_len + response_len + 5
    experts = np.arange(n_real * LAYERS * TOPK, dtype=np.int16).reshape(n_real, LAYERS, TOPK) % 16
    routed = _output(experts, prompt_len, response_len).as_dict()["routed_experts"]
    assert routed.shape == (prompt_len + response_len, LAYERS, TOPK)
    assert torch.equal(routed, torch.as_tensor(experts[: prompt_len + response_len], dtype=torch.int16))
