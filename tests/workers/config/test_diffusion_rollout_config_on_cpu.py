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

import pytest

from verl.workers.config.rollout import DiffusionRolloutConfig, DiffusionSamplingConfig


class TestDiffusionSamplingConfigCPU:
    def test_default_values(self):
        config = DiffusionSamplingConfig()
        assert config.do_sample is True
        assert config.n == 1
        assert config.noise_level == 0.0
        assert config.num_inference_steps == 40
        assert config.seed == 42


class TestDiffusionRolloutConfigCPU:
    def test_default_values(self):
        config = DiffusionRolloutConfig(name="vllm_omni")
        assert config.name == "vllm_omni"
        assert config.mode == "async"
        assert config.val_kwargs == DiffusionSamplingConfig()
        assert config.sde_type == "sde"

    def test_sync_mode_raises(self):
        with pytest.raises(ValueError, match="Rollout mode 'sync' has been removed"):
            DiffusionRolloutConfig(name="vllm_omni", mode="sync")

    def test_pipeline_parallel_not_supported_for_vllm_omni(self):
        with pytest.raises(NotImplementedError, match="not implemented pipeline_model_parallel_size > 1"):
            DiffusionRolloutConfig(name="vllm_omni", pipeline_model_parallel_size=2)
