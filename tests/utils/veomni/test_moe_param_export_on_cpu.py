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

import importlib.util
from pathlib import Path

import pytest
import torch

_UTILS_PATH = Path(__file__).parents[3] / "verl" / "workers" / "engine" / "veomni" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("_veomni_moe_export_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)

MOE_PARAM_HANDERS = _UTILS.MOE_PARAM_HANDERS
enumerate_hf_slots = _UTILS.enumerate_hf_slots
get_moe_param_handler = _UTILS.get_moe_param_handler
passthrough_moe_param_handler = _UTILS.passthrough_moe_param_handler
streamed_gpt_oss_moe_param_handler = _UTILS.streamed_gpt_oss_moe_param_handler


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("model.layers.0.mlp.experts.gate_up_proj", (4, 8, 12)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias", (4, 12)),
        ("model.layers.0.mlp.experts.down_proj", (4, 6, 8)),
        ("model.layers.0.mlp.experts.down_proj_bias", (4, 8)),
    ],
)
def test_gpt_oss_non_ep_keeps_packed_expert_params(name, shape):
    tensor = torch.randn(shape)
    handler = get_moe_param_handler("gpt_oss", ep_enabled=False)

    assert MOE_PARAM_HANDERS["gpt_oss"] is passthrough_moe_param_handler
    exported = list(handler(name, tensor, expert_id_base=0))

    assert len(exported) == 1
    assert exported[0][0] == name
    assert exported[0][1] is tensor


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("model.layers.0.mlp.experts.gate_up_proj", (2, 8, 12)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias", (2, 12)),
        ("model.layers.0.mlp.experts.down_proj", (2, 6, 8)),
        ("model.layers.0.mlp.experts.down_proj_bias", (2, 8)),
    ],
)
def test_gpt_oss_ep_streams_global_experts_in_checkpoint_layout(name, shape, monkeypatch):
    tensor = torch.randn(shape)
    monkeypatch.setattr(_UTILS, "get_device_id", lambda: tensor.device)
    handler = get_moe_param_handler("gpt_oss", ep_enabled=True)

    assert handler is streamed_gpt_oss_moe_param_handler
    exported = list(handler(name, tensor, expert_id_base=4))

    assert [exported_name for exported_name, _ in exported] == [
        name.replace("mlp.experts.", "mlp.experts.4."),
        name.replace("mlp.experts.", "mlp.experts.5."),
    ]
    assert all(not exported_name.endswith(".weight") for exported_name, _ in exported)
    torch.testing.assert_close(exported[0][1], tensor[0])
    torch.testing.assert_close(exported[1][1], tensor[1])

    slots = enumerate_hf_slots(handler, name, shape, tensor.dtype, device=tensor.device)
    assert slots == [
        (name.replace("mlp.experts.", "mlp.experts.0."), shape[1:]),
        (name.replace("mlp.experts.", "mlp.experts.1."), shape[1:]),
    ]
