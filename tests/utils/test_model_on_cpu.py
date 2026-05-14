# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace  # Or use a mock object library

import pytest
import torch

from verl.utils.model import split_fused_moe_experts, update_model_config


# Parametrize with different override scenarios
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
def test_update_model_config(override_kwargs):
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    mock_config = SimpleNamespace(
        param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me"
    )
    # Apply the updates using the parametrized override_kwargs
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        override_nested = override_kwargs["nested_params"]
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    else:  # Case 1: Plain override (nested params untouched)
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"


def test_split_fused_moe_experts_gate_up_stacked_rows():
    """transformers>=5.0.0: gate_up_proj shape ``(E, 2*inter, hidden)``."""
    e, inter, h = 2, 3, 4
    w = torch.arange(e * 2 * inter * h, dtype=torch.float32).reshape(e, 2 * inter, h)
    out = dict(
        split_fused_moe_experts(
            [("model.layers.0.mlp.experts.gate_up_proj", w)],
            hidden_size=h,
            moe_intermediate_size=inter,
        )
    )
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, h)
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (inter, h)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], w[0, :inter])
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.up_proj.weight"], w[0, inter:])


def test_split_fused_moe_experts_gate_up_hidden_last():
    """transformers<5.0.0: gate_up_proj shape ``(E, hidden, 2*inter)``."""
    h, two_inter = 4, 6
    inter = two_inter // 2
    g0 = torch.ones(h, inter)
    u0 = torch.ones(h, inter) * 2
    gu0 = torch.cat([g0, u0], dim=1)
    gu1 = torch.cat([g0 * 3, u0 * 4], dim=1)
    w = torch.stack([gu0, gu1], dim=0)
    out = dict(
        split_fused_moe_experts(
            [("model.layers.0.mlp.experts.gate_up_proj", w)],
            hidden_size=h,
            moe_intermediate_size=inter,
        )
    )
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, h)
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (inter, h)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.gate_proj.weight"], g0.T)
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.up_proj.weight"], u0.T)


def test_split_fused_moe_experts_gate_up_stacked_rows_fallback_without_config():
    """Without ``hidden_size`` / ``moe_intermediate_size``, keep stacked-rows behavior."""
    e, inter, h = 2, 3, 4
    w = torch.arange(e * 2 * inter * h, dtype=torch.float32).reshape(e, 2 * inter, h)
    out = dict(split_fused_moe_experts([("model.layers.0.mlp.experts.gate_up_proj", w)]))
    assert out["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter, h)


def test_split_fused_moe_experts_down_proj_hidden_first():
    """transformers>=5.0.0: down_proj shape ``(E, hidden, inter)``."""
    e, h, mi = 2, 8, 3
    w = torch.arange(e * h * mi, dtype=torch.float32).reshape(e, h, mi)
    out = dict(
        split_fused_moe_experts(
            [("model.layers.0.mlp.experts.down_proj", w)],
            hidden_size=h,
            moe_intermediate_size=mi,
        )
    )
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.down_proj.weight"], w[0])


def test_split_fused_moe_experts_down_proj_inter_first():
    """transformers<5.0.0: down_proj shape ``(E, inter, hidden)``."""
    e, h, mi = 2, 8, 3
    w = torch.arange(e * mi * h, dtype=torch.float32).reshape(e, mi, h)
    out = dict(
        split_fused_moe_experts(
            [("model.layers.0.mlp.experts.down_proj", w)],
            hidden_size=h,
            moe_intermediate_size=mi,
        )
    )
    torch.testing.assert_close(out["model.layers.0.mlp.experts.0.down_proj.weight"], w[0].T)
