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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.workers.engine.veomni.utils import default_moe_param_handler, get_moe_param_handler, veomni_shard_export


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

    exported = list(handler(name, tensor, expert_id_base=0))

    assert len(exported) == 1
    assert exported[0][0] == name
    assert exported[0][1] is tensor


def test_gpt_oss_ep_keeps_existing_export_handler():
    assert get_moe_param_handler("gpt_oss", ep_enabled=True) is default_moe_param_handler


def test_gpt_oss_delta_sharded_is_explicitly_unsupported():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="gpt_oss")

    with pytest.raises(NotImplementedError, match="GPT-OSS does not support delta_sharded"):
        veomni_shard_export(model)
