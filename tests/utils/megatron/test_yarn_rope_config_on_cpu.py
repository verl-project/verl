# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from types import SimpleNamespace


def _load_rope_config_module():
    module_path = Path(__file__).parents[3] / "verl" / "utils" / "megatron" / "rope_config.py"
    spec = importlib.util.spec_from_file_location("rope_config", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_yarn_runtime_config_from_legacy_rope_scaling():
    hf_config = SimpleNamespace(
        rope_theta=1000000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )

    rope_config = _load_rope_config_module()

    assert rope_config.get_hf_yarn_rope_config(hf_config, {}) == {
        "position_embedding_type": "yarn",
        "rotary_base": 1000000.0,
        "yarn_rotary_scaling_factor": 4.0,
        "yarn_original_max_position_embeddings": 32768,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 0.0,
        "yarn_correction_range_round_to_int": True,
    }


def test_prefers_nested_rope_parameters_over_rope_scaling():
    hf_config = SimpleNamespace(
        rope_theta=10000.0,
        rope_scaling={"type": "linear", "factor": 2.0},
        rope_parameters={
            "short": {"rope_type": "default", "rope_theta": 10000.0},
            "long": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "original_max_position_embeddings": 65536,
                "beta_fast": 48.0,
                "beta_slow": 2.0,
            },
        },
    )

    rope_config = _load_rope_config_module()

    yarn_config = rope_config.get_hf_yarn_rope_config(hf_config, {})

    assert yarn_config["position_embedding_type"] == "yarn"
    assert yarn_config["rotary_base"] == 500000.0
    assert yarn_config["yarn_rotary_scaling_factor"] == 8.0
    assert yarn_config["yarn_original_max_position_embeddings"] == 65536
    assert yarn_config["yarn_beta_fast"] == 48.0
    assert yarn_config["yarn_beta_slow"] == 2.0


def test_explicit_override_values_take_precedence_without_mutating_input():
    hf_config = SimpleNamespace(
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )
    override_transformer_config = {
        "attention_backend": "flash",
        "rotary_base": 123456.0,
        "yarn_rotary_scaling_factor": 16.0,
        "yarn_mscale": 0.5,
    }

    rope_config = _load_rope_config_module()

    yarn_config = rope_config.get_hf_yarn_rope_config(hf_config, override_transformer_config)

    assert yarn_config["rotary_base"] == 123456.0
    assert yarn_config["yarn_rotary_scaling_factor"] == 16.0
    assert yarn_config["yarn_mscale"] == 0.5
    assert rope_config.without_yarn_runtime_config(override_transformer_config) == {"attention_backend": "flash"}
    assert override_transformer_config == {
        "attention_backend": "flash",
        "rotary_base": 123456.0,
        "yarn_rotary_scaling_factor": 16.0,
        "yarn_mscale": 0.5,
    }


def test_non_yarn_rope_config_is_noop():
    hf_config = SimpleNamespace(
        rope_theta=10000.0,
        rope_scaling={"type": "linear", "factor": 2.0},
    )

    rope_config = _load_rope_config_module()

    assert rope_config.get_hf_yarn_rope_config(hf_config, {}) == {}
