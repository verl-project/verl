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

"""CPU unit tests for fully async policy utilities and GenRM config validation.

Tests GenRM/DisRM config validation and rollout metric aggregation.
"""

import unittest

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    RolloutSample,
    assemble_batch_from_rollout_samples,
)
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


def test_compute_score_metrics_are_preserved_and_aggregated():
    compute_score_times = [0.25, 0.75]
    full_batch = DataProto(
        batch=TensorDict(
            {"response_mask": torch.ones(2, 1, dtype=torch.long)},
            batch_size=[2],
        ),
        non_tensor_batch={
            "min_global_steps": np.array([1, 1]),
            "max_global_steps": np.array([1, 2]),
        },
        meta_info={
            "metrics": [
                {"generate_sequences": 1.0, "tool_calls": 0.0, "compute_score": value} for value in compute_score_times
            ]
        },
    )
    rollout_sample = RolloutSample(full_batch, sample_id="sample-0", epoch=0, rollout_status={})

    result = assemble_batch_from_rollout_samples([rollout_sample], tokenizer=None, config=None)

    assert result.non_tensor_batch["compute_score_times"] == pytest.approx(compute_score_times)
    assert result.meta_info["timing_s/agent_loop/compute_score/min"] == pytest.approx(0.25)
    assert result.meta_info["timing_s/agent_loop/compute_score/mean"] == pytest.approx(0.5)
    assert result.meta_info["timing_s/agent_loop/compute_score/max"] == pytest.approx(0.75)

    aggregator = MetricsAggregator(total_gpus=1)
    aggregator.add_step_metrics(result.meta_info, sample_count=2)
    aggregator.add_step_metrics(
        {
            "timing_s/agent_loop/compute_score/min": 0.5,
            "timing_s/agent_loop/compute_score/mean": 1.0,
            "timing_s/agent_loop/compute_score/max": 1.5,
        },
        sample_count=2,
    )
    aggregated = aggregator.get_aggregated_metrics()
    assert aggregated["timing_s/agent_loop/compute_score/min"] == pytest.approx(0.25)
    assert aggregated["timing_s/agent_loop/compute_score/mean"] == pytest.approx(0.75)
    assert aggregated["timing_s/agent_loop/compute_score/max"] == pytest.approx(1.5)


if __name__ == "__main__":
    unittest.main()
