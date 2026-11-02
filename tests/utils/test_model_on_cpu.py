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

from verl.utils.model import extract_multi_modal_inputs, update_model_config


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


def test_vision_token_types_keep_one_row_per_sample():
    """A text-only sample without vision_token_types still gets its own all -1 row, in batch order."""
    batch = [
        {},
        {"vision_token_types": torch.tensor([[-1, 0, 1, 3]])},
        None,
    ]
    merged = extract_multi_modal_inputs(batch)
    assert merged["vision_token_types"].tolist() == [[-1, -1, -1, -1], [-1, 0, 1, 3], [-1, -1, -1, -1]]
