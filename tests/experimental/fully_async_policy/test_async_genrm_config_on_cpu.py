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

"""Unit tests for async GenRM config validation logic (no GPU required).

Tests the config parsing and assertion logic added to FullyAsyncRollouter
to support GenRM/DisRM in fully async training mode.
"""

import unittest

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_reward_model


def _make_config(reward_model_enable=False, enable_resource_pool=False):
    """Create a minimal OmegaConf config for testing reward model settings."""
    return OmegaConf.create(
        {
            "reward": {
                "reward_model": {
                    "enable": reward_model_enable,
                    "enable_resource_pool": enable_resource_pool,
                    "n_gpus_per_node": 2,
                    "nnodes": 1,
                    "model_path": "dummy/model",
                    "rollout": {
                        "name": "vllm",
                        "tensor_model_parallel_size": 1,
                        "gpu_memory_utilization": 0.5,
                        "skip_tokenizer_init": False,
                    },
                },
                "custom_reward_function": {
                    "path": None,
                    "name": None,
                },
            },
        }
    )


class TestNeedRewardModel(unittest.TestCase):
    """Test that need_reward_model correctly reads config."""

    def test_rm_disabled(self):
        config = _make_config(reward_model_enable=False)
        assert need_reward_model(config) is False

    def test_rm_enabled(self):
        config = _make_config(reward_model_enable=True)
        assert need_reward_model(config) is True


class TestAsyncRollouterRMAssert(unittest.TestCase):
    """Test the assertion logic that enforces standalone mode for async RM.

    This replicates the validation logic from FullyAsyncRollouter.__init__
    without instantiating the full class (which requires Ray, worker groups, etc.).
    """

    @staticmethod
    def _validate_async_rm_config(config):
        """Replicate the RM validation logic from FullyAsyncRollouter.__init__."""
        use_rm = need_reward_model(config)
        if use_rm:
            assert config.reward.reward_model.enable_resource_pool, (
                "GenRM/DisRM in fully async mode requires standalone mode (enable_resource_pool=True). "
                "Colocate mode is not supported because async rollout never pauses."
            )
        return use_rm

    def test_rm_disabled_passes(self):
        """use_rm=False should pass regardless of enable_resource_pool."""
        config = _make_config(reward_model_enable=False, enable_resource_pool=False)
        use_rm = self._validate_async_rm_config(config)
        assert use_rm is False

    def test_rm_enabled_standalone_passes(self):
        """use_rm=True + enable_resource_pool=True (standalone) should pass."""
        config = _make_config(reward_model_enable=True, enable_resource_pool=True)
        use_rm = self._validate_async_rm_config(config)
        assert use_rm is True

    def test_rm_enabled_colocate_fails(self):
        """use_rm=True + enable_resource_pool=False (colocate) should assert."""
        config = _make_config(reward_model_enable=True, enable_resource_pool=False)
        with pytest.raises(AssertionError, match="standalone mode"):
            self._validate_async_rm_config(config)


if __name__ == "__main__":
    unittest.main()
