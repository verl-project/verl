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

import dataclasses

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from verl.utils import omega_conf_to_dataclass
from verl.workers.config.rollout import MultiTurnConfig


def test_invalid_tool_call_limit_defaults_to_disabled() -> None:
    assert MultiTurnConfig().max_consecutive_invalid_tool_calls is None


def test_invalid_tool_call_limit_config_is_immutable() -> None:
    config = MultiTurnConfig(max_consecutive_invalid_tool_calls=5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_consecutive_invalid_tool_calls = 6


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "5"])
def test_invalid_tool_call_limit_rejects_non_positive_or_non_integer_values(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_consecutive_invalid_tool_calls must be a positive integer or null",
    ):
        MultiTurnConfig(max_consecutive_invalid_tool_calls=value)


@pytest.mark.parametrize("value", [1, 5, 100])
def test_invalid_tool_call_limit_accepts_positive_integers(value: int) -> None:
    assert MultiTurnConfig(max_consecutive_invalid_tool_calls=value).max_consecutive_invalid_tool_calls == value


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5])
def test_invalid_tool_call_limit_is_validated_after_omegaconf_conversion(value: object) -> None:
    config = OmegaConf.create({"max_consecutive_invalid_tool_calls": value})
    with pytest.raises(
        (ValueError, ValidationError),
        match="positive integer or null|could not be converted to Integer",
    ):
        omega_conf_to_dataclass(config, MultiTurnConfig)


@pytest.mark.parametrize("value", [None, 1, 5])
def test_invalid_tool_call_limit_survives_omegaconf_conversion(value: int | None) -> None:
    config = OmegaConf.create({"max_consecutive_invalid_tool_calls": value})
    converted = omega_conf_to_dataclass(config, MultiTurnConfig)
    assert converted.max_consecutive_invalid_tool_calls == value


def test_invalid_tool_call_limit_string_is_coerced_before_dataclass_validation() -> None:
    config = OmegaConf.create({"max_consecutive_invalid_tool_calls": "5"})
    converted = omega_conf_to_dataclass(config, MultiTurnConfig)
    assert converted.max_consecutive_invalid_tool_calls == 5
