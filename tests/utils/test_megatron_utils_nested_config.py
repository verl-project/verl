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
    text_config = SimpleNamespace(rope_theta=1_000_000.0)

    class OmniConfig:
        sub_configs = {"thinker_config": object, "code2wav_config": object}

        def __init__(self):
            self.thinker_config = SimpleNamespace(text_config=text_config)
            self.code2wav_config = SimpleNamespace(rope_theta=10_000.0)

        def get_text_config(self):
            return text_config

    assert get_hf_rope_theta(OmniConfig()) == 1_000_000.0


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


def test_get_hf_config_attr_uses_standard_sub_configs_and_text_accessor():
    text_config = SimpleNamespace(hidden_size=8192)

    class CompositeConfig:
        sub_configs = {"decoder_config": object}

        def __init__(self):
            self.hidden_size = 1024
            self.decoder_config = SimpleNamespace(hidden_size=2048)

        def get_text_config(self):
            return text_config

    assert get_hf_config_attr(CompositeConfig(), "hidden_size") == 8192


def test_get_hf_config_attr_prefers_direct_text_config_over_root():
    config = SimpleNamespace(hidden_size=1024, text_config=SimpleNamespace(hidden_size=4096))

    assert get_hf_config_attr(config, "hidden_size") == 4096


def test_nested_config_lookup_does_not_fall_back_to_multimodal_sibling():
    class CompositeConfig:
        sub_configs = {"audio_config": object, "code2wav_config": object}

        def __init__(self):
            self.audio_config = SimpleNamespace(hidden_size=1024)
            self.code2wav_config = SimpleNamespace(rope_theta=10_000.0)

    with pytest.raises(AttributeError, match="has no nested hidden_size"):
        get_hf_config_attr(CompositeConfig(), "hidden_size")
    with pytest.raises(AttributeError, match="has no rope_theta"):
        get_hf_rope_theta(CompositeConfig())


def test_get_hf_config_attr_preserves_falsey_text_values():
    config = SimpleNamespace(text_config=SimpleNamespace(flag=False, count=0))

    assert get_hf_config_attr(config, "flag") is False
    assert get_hf_config_attr(config, "count") == 0


def test_nested_config_lookup_rejects_missing_attributes():
    config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(AttributeError, match="has no nested hidden_size"):
        get_hf_config_attr(config, "hidden_size")
