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

from verl.workers.config.optimizer import FSDPOptimizerConfig, McoreOptimizerConfig


class TestFSDPOptimizerConfigCPU:
    def test_default_configuration(self):
        config = FSDPOptimizerConfig(lr=0.1)
        assert config.min_lr_ratio is None
        assert config.lr_scheduler_type == "constant"
        assert config.num_cycles == 0.5

    @pytest.mark.parametrize("lr_scheduler_type", ["constant", "cosine"])
    def test_valid_lr_scheduler_types(self, lr_scheduler_type):
        config = FSDPOptimizerConfig(lr_scheduler_type=lr_scheduler_type, lr=0.1)
        assert config.lr_scheduler_type == lr_scheduler_type

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_valid_warmup_style_types(self, warmup_style):
        config = FSDPOptimizerConfig(warmup_style=warmup_style, lr=0.1)
        assert config.lr_scheduler_type == warmup_style

    def test_invalid_lr_scheduler_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(lr_scheduler_type="invalid_style", lr=0.1)

    def test_invalid_warmup_style_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(warmup_style="invalid_style", lr=0.1)

    @pytest.mark.parametrize("num_cycles", [0.1, 1.0, 2.5])
    def test_num_cycles_configuration(self, num_cycles):
        config = FSDPOptimizerConfig(num_cycles=num_cycles, lr=0.1)
        assert config.num_cycles == num_cycles


class TestMcoreOptimizerConfigMuonCPU:
    """The Muon (emerging optimizer) fields are pure config plumbing, so they are exercised on CPU."""

    def test_default_optimizer_is_adam(self):
        config = McoreOptimizerConfig(lr=0.1)
        assert config.optimizer == "adam"

    def test_muon_defaults_track_megatron(self):
        config = McoreOptimizerConfig(lr=0.1)
        # Defaults mirror Megatron-Core's OptimizerConfig Muon defaults.
        assert config.use_layer_wise_distributed_optimizer is False
        assert config.muon_momentum == 0.95
        assert config.muon_nesterov is False
        assert config.muon_split_qkv is True
        assert config.muon_scale_mode == "spectral"
        assert config.muon_coefficient_type == "quintic"
        assert config.muon_num_ns_steps == 5
        assert config.muon_tp_mode == "blockwise"
        assert config.muon_fp32_matmul_prec == "medium"
        assert config.muon_extra_scale_factor == 1.0
        assert config.muon_scalar_optimizer == "adam"

    def test_muon_overrides_are_carried(self):
        config = McoreOptimizerConfig(
            lr=0.1,
            optimizer="muon",
            muon_momentum=0.9,
            muon_num_ns_steps=6,
            muon_tp_mode="allgather",
            muon_scalar_optimizer="lion",
            use_layer_wise_distributed_optimizer=True,
        )
        assert config.optimizer == "muon"
        assert config.muon_momentum == 0.9
        assert config.muon_num_ns_steps == 6
        assert config.muon_tp_mode == "allgather"
        assert config.muon_scalar_optimizer == "lion"
        assert config.use_layer_wise_distributed_optimizer is True
