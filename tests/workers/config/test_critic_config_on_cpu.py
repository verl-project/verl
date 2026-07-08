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

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import ProfilerConfig
from verl.workers.config import (
    CriticConfig,
    FSDPCriticConfig,
    FSDPOptimizerConfig,
    McoreCriticConfig,
    McoreOptimizerConfig,
    OptimizerConfig,
)


@pytest.mark.skip(reason="This test is flaky when we actively load model config")
class TestCriticConfig:
    """Test suite for critic configuration dataclasses."""

    @pytest.fixture
    def config_dir(self):
        """Get the path to the config directory."""
        return Path(__file__).parent.parent.parent.parent / "verl" / "trainer" / "config" / "critic"

    def test_megatron_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that McoreCriticConfig can be instantiated from megatron_critic.yaml."""
        yaml_path = config_dir / "megatron_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="megatron_critic", overrides=["ppo_micro_batch_size_per_gpu=1"])

        megatron_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(megatron_config_obj, McoreCriticConfig)
        assert isinstance(megatron_config_obj, CriticConfig)

        expected_attrs = [
            "strategy",
            "rollout_n",
            "optim",
            "model",
            "ppo_mini_batch_size",
            "ppo_max_token_len_per_gpu",
            "cliprange_value",
            "get",
            "nccl_timeout",
            "megatron",
            "load_weight",
        ]
        for attr in expected_attrs:
            assert hasattr(megatron_config_obj, attr), f"Missing attribute: {attr}"

        assert callable(megatron_config_obj.get)
        assert megatron_config_obj.strategy == "megatron"

    def test_fsdp_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that FSDPCriticConfig can be instantiated from dp_critic.yaml."""
        yaml_path = config_dir / "dp_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="dp_critic", overrides=["ppo_micro_batch_size_per_gpu=1"])

        fsdp_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(fsdp_config_obj, FSDPCriticConfig)
        assert isinstance(fsdp_config_obj, CriticConfig)

        expected_attrs = [
            "strategy",
            "rollout_n",
            "optim",
            "model",
            "ppo_mini_batch_size",
            "ppo_max_token_len_per_gpu",
            "cliprange_value",
            "get",
            "forward_micro_batch_size",
            "forward_micro_batch_size_per_gpu",
            "ulysses_sequence_parallel_size",
            "grad_clip",
        ]
        for attr in expected_attrs:
            assert hasattr(fsdp_config_obj, attr), f"Missing attribute: {attr}"

        assert callable(fsdp_config_obj.get)
        assert fsdp_config_obj.strategy == "fsdp"

    def test_config_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        megatron_config = McoreCriticConfig(ppo_micro_batch_size_per_gpu=1, optim=McoreOptimizerConfig(lr=0.1))
        assert isinstance(megatron_config, CriticConfig)
        assert isinstance(megatron_config, McoreCriticConfig)

        fsdp_config = FSDPCriticConfig(ppo_micro_batch_size_per_gpu=1, optim=FSDPOptimizerConfig(lr=0.1))
        assert isinstance(fsdp_config, CriticConfig)
        assert isinstance(fsdp_config, FSDPCriticConfig)

        critic_config = CriticConfig(ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=OptimizerConfig(lr=0.1))
        assert isinstance(critic_config, CriticConfig)
        assert not isinstance(critic_config, McoreCriticConfig)
        assert not isinstance(critic_config, FSDPCriticConfig)

    def test_config_dict_interface(self):
        """Test that configs provide dict-like interface from BaseConfig."""
        optim = OptimizerConfig(lr=0.1)
        config = CriticConfig(ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=optim)

        assert "strategy" in config
        assert config["strategy"] == "fsdp2"

        assert config.get("strategy") == "fsdp2"
        assert config.get("nonexistent_key", "default") == "default"

        keys = list(config)
        assert "strategy" in keys
        assert "rollout_n" in keys

        assert len(config) > 0

    def test_frozen_fields_immutability(self):
        """Test that frozen fields raise exceptions when modified after creation."""
        critic_config = CriticConfig(ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=OptimizerConfig(lr=0.1))
        frozen_fields = ["rollout_n", "strategy", "cliprange_value"]

        for field_name in frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(critic_config, field_name, "modified_value")

        megatron_config = McoreCriticConfig(ppo_micro_batch_size_per_gpu=1, optim=McoreOptimizerConfig(lr=0.1))
        megatron_frozen_fields = ["nccl_timeout", "load_weight", "data_loader_seed"]

        for field_name in megatron_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(megatron_config, field_name, "modified_value")

        fsdp_config = FSDPCriticConfig(ppo_micro_batch_size_per_gpu=1, optim=FSDPOptimizerConfig(lr=0.1))
        fsdp_frozen_fields = ["ulysses_sequence_parallel_size", "grad_clip"]

        for field_name in fsdp_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(fsdp_config, field_name, "modified_value")

    def test_batch_size_fields_modifiable(self):
        """Test that batch size fields can be modified after creation."""
        optim = OptimizerConfig(lr=0.1)
        critic_config = CriticConfig(ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=optim)

        critic_config.ppo_mini_batch_size = 8
        critic_config.ppo_micro_batch_size = 4
        critic_config.ppo_micro_batch_size_per_gpu = 2

        assert critic_config.ppo_mini_batch_size == 8
        assert critic_config.ppo_micro_batch_size == 4
        assert critic_config.ppo_micro_batch_size_per_gpu == 2

        fsdp_config = FSDPCriticConfig(ppo_micro_batch_size_per_gpu=1, optim=FSDPOptimizerConfig(lr=0.1))

        fsdp_config.forward_micro_batch_size = 16
        fsdp_config.forward_micro_batch_size_per_gpu = 8

        assert fsdp_config.forward_micro_batch_size == 16
        assert fsdp_config.forward_micro_batch_size_per_gpu == 8

    def test_profiler_config_type_validation(self):
        """Test that profiler field has correct type and validation."""
        optim = OptimizerConfig(lr=0.1)
        critic_config = CriticConfig(ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=optim)
        assert isinstance(critic_config.profiler, ProfilerConfig)
        assert critic_config.profiler.all_ranks is False
        assert critic_config.profiler.ranks == []

        custom_profiler = ProfilerConfig(all_ranks=True, ranks=[0, 1])
        critic_config_custom = CriticConfig(
            profiler=custom_profiler, ppo_micro_batch_size_per_gpu=1, strategy="fsdp2", optim=optim
        )
        assert isinstance(critic_config_custom.profiler, ProfilerConfig)
        assert critic_config_custom.profiler.all_ranks is True
        assert critic_config_custom.profiler.ranks == [0, 1]

        profiler1 = ProfilerConfig(enable=True, ranks=[0, 1])
        profiler2 = ProfilerConfig(all_ranks=True, ranks=[1, 2])

        union_result = profiler1.union(profiler2)
        assert union_result.enable is True
        assert union_result.all_ranks is True
        assert set(union_result.ranks) == {0, 1, 2}

        intersect_result = profiler1.intersect(profiler2)
        assert intersect_result.all_ranks is False
        assert intersect_result.ranks == [1]

    def test_critic_config_validation_logic(self):
        """Test the __post_init__ validation logic for CriticConfig."""
        optim = OptimizerConfig(lr=0.1)
        valid_config = CriticConfig(
            strategy="fsdp2", ppo_micro_batch_size_per_gpu=2, use_dynamic_bsz=False, optim=optim
        )
        assert valid_config.ppo_micro_batch_size_per_gpu == 2

        valid_config2 = CriticConfig(
            strategy="fsdp2",
            ppo_micro_batch_size_per_gpu=None,
            ppo_micro_batch_size=4,
            ppo_mini_batch_size=8,
            use_dynamic_bsz=False,
            optim=optim,
        )
        assert valid_config2.ppo_micro_batch_size == 4

        dynamic_config = CriticConfig(
            strategy="fsdp2", ppo_micro_batch_size_per_gpu=2, use_dynamic_bsz=True, optim=optim
        )
        assert dynamic_config.use_dynamic_bsz is True

        with pytest.raises(ValueError, match="You have set both.*micro_batch_size.*AND.*micro_batch_size_per_gpu"):
            CriticConfig(
                strategy="fsdp2",
                ppo_micro_batch_size=4,
                ppo_micro_batch_size_per_gpu=2,
                use_dynamic_bsz=False,
                optim=optim,
            )

        with pytest.raises(
            ValueError, match="Please set at least one of.*micro_batch_size.*or.*micro_batch_size_per_gpu"
        ):
            CriticConfig(
                strategy="fsdp2",
                ppo_micro_batch_size=None,
                ppo_micro_batch_size_per_gpu=None,
                use_dynamic_bsz=False,
                optim=optim,
            )

    def test_micro_batch_size_divisibility_validation(self):
        """Test micro batch size divisibility validation in __post_init__."""
        optim = OptimizerConfig(lr=0.1)
        valid_config = CriticConfig(
            strategy="fsdp2", ppo_micro_batch_size_per_gpu=2, ppo_mini_batch_size=8, use_dynamic_bsz=False, optim=optim
        )
        assert valid_config.ppo_mini_batch_size == 8
        assert valid_config.ppo_micro_batch_size_per_gpu == 2

        valid_config_with_mbs = CriticConfig(
            strategy="fsdp2", ppo_mini_batch_size=8, ppo_micro_batch_size=4, use_dynamic_bsz=False, optim=optim
        )
        assert valid_config_with_mbs.ppo_mini_batch_size == 8
        assert valid_config_with_mbs.ppo_micro_batch_size == 4

        with pytest.raises(ValueError, match="ppo_mini_batch_size.*must be divisible by.*ppo_micro_batch_size"):
            CriticConfig(
                strategy="fsdp2", ppo_mini_batch_size=7, ppo_micro_batch_size=4, use_dynamic_bsz=False, optim=optim
            )

        dynamic_config = CriticConfig(
            strategy="fsdp2", ppo_mini_batch_size=7, ppo_micro_batch_size=4, use_dynamic_bsz=True, optim=optim
        )
        assert dynamic_config.use_dynamic_bsz is True

    def test_fsdp_sequence_parallelism_validation(self):
        """Test FSDP sequence parallelism validation in FSDPCriticConfig.__post_init__."""
        valid_config = FSDPCriticConfig(
            ppo_micro_batch_size_per_gpu=2,
            ulysses_sequence_parallel_size=2,
            model={"use_remove_padding": True},
            optim=FSDPOptimizerConfig(lr=0.1),
        )
        assert valid_config.ulysses_sequence_parallel_size == 2

        with pytest.raises(
            ValueError, match="When using sequence parallelism for critic, you must enable.*use_remove_padding"
        ):
            FSDPCriticConfig(
                ppo_micro_batch_size_per_gpu=2,
                ulysses_sequence_parallel_size=2,
                model={"use_remove_padding": False},
                optim=FSDPOptimizerConfig(lr=0.1),
            )

        valid_config_no_sp = FSDPCriticConfig(
            ppo_micro_batch_size_per_gpu=2,
            ulysses_sequence_parallel_size=1,
            model={"use_remove_padding": False},
            optim=FSDPOptimizerConfig(lr=0.1),
        )
        assert valid_config_no_sp.ulysses_sequence_parallel_size == 1
