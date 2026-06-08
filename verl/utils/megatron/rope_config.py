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

from typing import Any

YARN_RUNTIME_CONFIG_KEYS = {
    "position_embedding_type",
    "rotary_base",
    "yarn_rotary_scaling_factor",
    "yarn_original_max_position_embeddings",
    "yarn_beta_fast",
    "yarn_beta_slow",
    "yarn_mscale",
    "yarn_mscale_all_dim",
    "yarn_correction_range_round_to_int",
}


def _get_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_hf_value(hf_config: Any, key: str, default: Any = None) -> Any:
    value = getattr(hf_config, key, default)
    if value is not default:
        return value

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, key, default)
    return default


def _get_rope_config(hf_config: Any) -> Any:
    rope_config = _get_hf_value(hf_config, "rope_parameters", None)
    if rope_config is None:
        rope_config = _get_hf_value(hf_config, "rope_scaling", None)

    if isinstance(rope_config, dict) and "rope_type" not in rope_config and "type" not in rope_config:
        for value in rope_config.values():
            rope_type = _get_value(value, "rope_type", _get_value(value, "type"))
            if rope_type == "yarn":
                return value
    return rope_config


def get_hf_yarn_rope_config(hf_config: Any, override_transformer_config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract YaRN fields that GPTModel reads after Megatron config construction."""
    rope_config = _get_rope_config(hf_config)
    yarn_config: dict[str, Any] = {}

    if rope_config is not None:
        rope_type = _get_value(rope_config, "rope_type", _get_value(rope_config, "type"))
        if rope_type == "yarn":
            yarn_config = {
                "position_embedding_type": "yarn",
                "rotary_base": _get_value(rope_config, "rope_theta", _get_hf_value(hf_config, "rope_theta", None)),
                "yarn_rotary_scaling_factor": _get_value(rope_config, "factor"),
                "yarn_original_max_position_embeddings": _get_value(
                    rope_config, "original_max_position_embeddings"
                ),
                "yarn_beta_fast": _get_value(rope_config, "beta_fast", 32.0),
                "yarn_beta_slow": _get_value(rope_config, "beta_slow", 1.0),
                "yarn_mscale": _get_value(rope_config, "mscale", 1.0),
                "yarn_mscale_all_dim": _get_value(rope_config, "mscale_all_dim", 0.0),
                "yarn_correction_range_round_to_int": _get_value(
                    rope_config, "correction_range_round_to_int", True
                ),
            }
            yarn_config = {key: value for key, value in yarn_config.items() if value is not None}

    if override_transformer_config:
        for key in YARN_RUNTIME_CONFIG_KEYS:
            if key in override_transformer_config:
                yarn_config[key] = override_transformer_config[key]
    return yarn_config


def without_yarn_runtime_config(override_transformer_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in override_transformer_config.items() if key not in YARN_RUNTIME_CONFIG_KEYS}
