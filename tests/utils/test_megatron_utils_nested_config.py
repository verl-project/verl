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

from types import SimpleNamespace

import pytest

from verl.utils.megatron_utils import get_hf_config_attr, get_hf_rope_theta


def test_get_hf_rope_theta_from_nested_omni_text_config():
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(rope_theta=1_000_000.0),
        )
    )

    assert get_hf_rope_theta(config) == 1_000_000.0


def test_get_hf_rope_theta_from_nested_transformers_v5_parameters():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            language_config=SimpleNamespace(rope_parameters={"full_attention": {"rope_theta": 10_000.0}}),
        )
    )

    assert get_hf_rope_theta(config) == 10_000.0


def test_get_hf_config_attr_handles_nested_configs_and_cycles():
    config = SimpleNamespace()
    config.thinker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096))
    config.model_config = config

    assert get_hf_config_attr(config, "hidden_size") == 4096


def test_nested_config_lookup_rejects_missing_attributes():
    config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(AttributeError, match="has no nested hidden_size"):
        get_hf_config_attr(config, "hidden_size")
