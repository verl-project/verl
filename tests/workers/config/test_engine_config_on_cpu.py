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

import os

import pytest
from omegaconf import OmegaConf

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig


class TestMcoreEngineConfig:
    def test_default_values(self):
        config = McoreEngineConfig()
        assert config.tensor_model_parallel_size == 1
        assert config.sequence_parallel is False  # Should be auto-corrected
        assert config.seed == 42

    def test_post_init_validation(self):
        # Test TP size 1 forces sequence_parallel=False
        config = McoreEngineConfig(tensor_model_parallel_size=1)
        assert config.sequence_parallel is False

        # Test TP >1 keeps sequence_parallel=True
        config = McoreEngineConfig(tensor_model_parallel_size=2)
        assert config.sequence_parallel is True

    def test_mutable_fields(self):
        config = McoreEngineConfig()
        config.sequence_parallel = True  # Should be mutable
        with pytest.raises(AttributeError):
            config.tensor_model_parallel_size = 2  # Frozen field

    @pytest.mark.parametrize("offload_field", ["param_offload", "optimizer_offload"])
    def test_offload_flags(self, offload_field):
        config = McoreEngineConfig(**{offload_field: True})
        assert getattr(config, offload_field) is True


class TestFSDPEngineConfigCPU:
    def test_default_values(self):
        config = FSDPEngineConfig()
        assert config.param_offload is False
        assert config.optimizer_offload is False
        assert config.fsdp_size == -1
        assert config.use_no_sync_for_gradient_accumulation is True

    def test_gradient_accumulation_sync_can_be_restored_per_micro_batch(self):
        config = FSDPEngineConfig(use_no_sync_for_gradient_accumulation=False)
        assert config.use_no_sync_for_gradient_accumulation is False

    @pytest.mark.parametrize(
        "offload_params",
        [{"param_offload": True}, {"optimizer_offload": True}, {"param_offload": True, "optimizer_offload": True}],
    )
    def test_offload_combinations(self, offload_params):
        config = FSDPEngineConfig(**offload_params)
        assert config.param_offload == offload_params.get("param_offload", False)
        assert config.optimizer_offload == offload_params.get("optimizer_offload", False)

    def test_wrap_policy_configuration(self):
        test_policy = {"layer_class": "TransformerBlock"}
        config = FSDPEngineConfig(wrap_policy=test_policy)
        assert config.wrap_policy == test_policy

    def test_prefix_packing_default_and_round_trip(self):
        assert FSDPEngineConfig().use_prefix_packing is False
        assert FSDPEngineConfig(use_prefix_packing=True).use_prefix_packing is True

    def test_prefix_packing_yaml_default_matches_dataclass(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "config", "engine", "fsdp.yaml"
        )
        yaml_cfg = OmegaConf.load(yaml_path)
        assert yaml_cfg.use_prefix_packing == FSDPEngineConfig().use_prefix_packing
