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

import pytest

from verl.base_config import BaseConfig
from verl.trainer.config.config import ModuleConfig
from verl.workers.config.reward import RewardConfig, RewardManagerConfig


class TestRewardManagerConfig:
    """Test suite for RewardManagerConfig, a direct BaseConfig subclass."""

    def test_default_instantiation(self):
        """RewardManagerConfig() must construct without error.

        Regression test: __post_init__ previously called super().__post_init__(),
        but BaseConfig defines no __post_init__, so default construction raised
        AttributeError: 'super' object has no attribute '__post_init__'.
        """
        config = RewardManagerConfig()
        assert isinstance(config, BaseConfig)
        assert config.source == "register"
        assert config.name == "naive"

    def test_importlib_source_requires_module_path(self):
        """When source is importlib, module.path must be set."""
        config = RewardManagerConfig(source="importlib", module=ModuleConfig(path="my.module"))
        assert config.source == "importlib"

        with pytest.raises(AssertionError, match="module.path should be set"):
            RewardManagerConfig(source="importlib", module=ModuleConfig())

    def test_reward_config_builds_reward_manager(self):
        """RewardConfig builds RewardManagerConfig via its default_factory."""
        config = RewardConfig()
        assert isinstance(config.reward_manager, RewardManagerConfig)

    def test_dict_interface(self):
        """RewardManagerConfig provides the dict-like interface from BaseConfig."""
        config = RewardManagerConfig()
        assert "source" in config
        assert config["source"] == "register"
        assert config.get("nonexistent_key", "default") == "default"
