# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com and/or its affiliates
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
from pprint import pp

from omegaconf import OmegaConf

from verl.workers.config.model import HFModelConfig


def test_target_modules_accepts_list_via_omegaconf():
    """
    Test that target_modules field accepts both string and list values
    when merging OmegaConf configs (simulates CLI override behavior).

    The purpose is to ensure we can pass
    actor_rollout_ref.model.target_modules='["k_proj","o_proj","down_proj","q_proj"]'
    """
    model_path = "~/models/Qwen/Qwen2.5-0.5B"  # Just a path string, not loaded

    # Create structured config from the dataclass defaults
    # This is what omega_conf_to_dataclass does internally
    cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

    pp("{cfg_from_dataclass=}")

    # Simulate CLI override with target_modules as a list
    cli_config = OmegaConf.create(
        {
            "path": model_path,
            "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
        }
    )

    pp("{cli_config=}")

    # This merge should NOT raise ValidationError
    # Before the fix (target_modules: str), this would fail with:
    # "Cannot convert 'ListConfig' to string"
    merged = OmegaConf.merge(cfg_from_dataclass, cli_config)

    # Verify the list was merged correctly
    assert list(merged.target_modules) == ["k_proj", "o_proj", "q_proj", "v_proj"]


def test_target_modules_accepts_string_via_omegaconf():
    """Test that target_modules still accepts string values."""
    cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

    cli_config = OmegaConf.create(
        {
            "path": "~/models/some-model",
            "target_modules": "all-linear",
        }
    )

    merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
    assert merged.target_modules == "all-linear"
