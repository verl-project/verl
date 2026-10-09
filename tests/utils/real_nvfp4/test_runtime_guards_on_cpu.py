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

"""BF16 transport checks for native NVFP4 refit."""

import pytest
import torch

from verl.utils.real_nvfp4.bf16_transport import attest_real_nvfp4_bf16_transport


def _weights():
    return [
        (f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight", torch.ones(2, 16))
        for layer in range(2)
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]


def _attest(weights):
    return list(attest_real_nvfp4_bf16_transport(iter(weights), {"num_hidden_layers": 2, "num_experts": 2}))


def test_expert_coverage_accepts_complete_reordered_stream():
    weights = list(reversed(_weights()))
    weights.insert(0, ("model.embed_tokens.weight", torch.ones(2, 16)))
    assert len(_attest(weights)) == 13


def test_expert_coverage_rejects_duplicate_replacing_missing_projection():
    weights = _weights()
    weights[1] = weights[0]
    with pytest.raises(RuntimeError, match="duplicate expert weight"):
        _attest(weights)


def test_expert_coverage_rejects_missing_projection():
    with pytest.raises(RuntimeError, match="missing 1 expert weights"):
        _attest(_weights()[:-1])


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "other.layers.0.mlp.experts.0.gate_proj.weight",
    ],
)
def test_expert_coverage_rejects_wrong_layer_expert_or_prefix(name):
    weights = _weights()
    weights[0] = name, weights[0][1]
    with pytest.raises(RuntimeError, match="unexpected expert weight"):
        _attest(weights)


def test_transport_rejects_prepacked_tensors():
    weights = _weights() + [("model.layers.0.mlp.experts.0.gate_proj.weight_scale", torch.ones(1))]
    with pytest.raises(RuntimeError, match="packed tensor"):
        _attest(weights)


def test_transport_rejects_non_floating_expert_weights():
    weights = _weights()
    weights[0] = weights[0][0], torch.ones(2, 8, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="must be floating point"):
        _attest(weights)
