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

import asyncio

import torch

from verl.workers.rollout.vllm_rollout.vllm_rollout import _iter_vllm_compatible_moe_params


def _collect(weights, model_type):
    async def collect():
        return [item async for item in _iter_vllm_compatible_moe_params(weights, model_type)]

    return asyncio.run(collect())


def test_qwen_moe_packed_weights_are_expanded_per_expert():
    gate_up = torch.randn(2, 6, 8)
    down = torch.randn(2, 8, 3)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_moe",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (3, 8),
        (3, 8),
        (3, 8),
        (3, 8),
        (8, 3),
        (8, 3),
    ]


def test_gpt_oss_packed_weights_are_not_expanded():
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 3, 8)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    converted = _collect(weights, "gpt_oss")

    assert [name for name, _ in converted] == [name for name, _ in weights]
    assert converted[0][1] is gate_up
    assert converted[1][1] is down
    assert converted[0][1].shape == (2, 8, 6)
    assert converted[1][1].shape == (2, 3, 8)
