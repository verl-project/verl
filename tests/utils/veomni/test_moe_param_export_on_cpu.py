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
get_moe_param_handler = _UTILS.get_moe_param_handler
passthrough_moe_param_handler = _UTILS.passthrough_moe_param_handler


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
