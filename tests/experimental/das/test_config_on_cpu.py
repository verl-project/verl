# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""``rollout.das`` config: dataclass validation, JSON hop, and RolloutConfig coercion."""

import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DASConfig, RolloutConfig
from verl.workers.config.das import CLS_LONG, CLS_MEDIUM, CLS_SHORT

ROLLOUT_YAML = "verl/trainer/config/rollout/rollout.yaml"


def test_defaults_are_off_and_sane():
    cfg = DASConfig()
    assert cfg.enable is False
    assert cfg.budget_for_class(CLS_LONG) == 24
    assert cfg.budget_for_class(CLS_MEDIUM) == 8
    assert cfg.budget_for_class(CLS_SHORT) == 0


def test_class_for_length_boundaries():
    cfg = DASConfig(short_max_tokens=256, long_min_tokens=1024)
    assert cfg.class_for_length(0) == CLS_SHORT
    assert cfg.class_for_length(255) == CLS_SHORT
    assert cfg.class_for_length(256) == CLS_MEDIUM
    assert cfg.class_for_length(1023) == CLS_MEDIUM
    assert cfg.class_for_length(1024) == CLS_LONG


@pytest.mark.parametrize(
    "bad",
    [
        {"window_iterations": 0},
        {"fresh_iterations": -1},
        {"poll_interval_s": 0},
        {"budget_medium": -1},
        {"short_max_tokens": 2048, "long_min_tokens": 1024},
        {"max_problems": 0},
        {"push_batch_size": 0},
        {"service_name": ""},
    ],
)
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        DASConfig(**bad)


def test_json_round_trip_excludes_target():
    cfg = DASConfig(enable=True, budget_long=12, service_name="x")
    payload = cfg.to_json()
    assert "_target_" not in payload
    assert DASConfig.from_json(payload) == cfg


def test_yaml_block_matches_dataclass_defaults():
    cfg = OmegaConf.load(ROLLOUT_YAML)
    yaml_das = OmegaConf.to_container(cfg.das, resolve=True)
    assert DASConfig(**yaml_das) == DASConfig()


def test_rollout_config_coerces_dict_and_dictconfig():
    r = RolloutConfig(name="vllm", das={"enable": False, "budget_long": 12})
    assert isinstance(r.das, DASConfig)
    assert r.das.budget_long == 12

    cfg = OmegaConf.load(ROLLOUT_YAML)
    cfg.name = "vllm"
    cfg.das.enable = True
    cfg.engine_kwargs = {"vllm": {"speculative_config": {"method": "suffix", "num_speculative_tokens": 24}}}
    r = omega_conf_to_dataclass(cfg)
    assert isinstance(r.das, DASConfig)
    assert r.das.enable is True


def test_enable_requires_vllm_and_single_dp():
    with pytest.raises(ValueError, match="rollout.name='vllm'"):
        RolloutConfig(name="sglang", das={"enable": True})
    with pytest.raises(ValueError, match="data_parallel_size"):
        RolloutConfig(name="vllm", data_parallel_size=2, das={"enable": True})


def test_enable_without_suffix_spec_config_warns():
    with pytest.warns(UserWarning, match="method is not 'suffix'"):
        RolloutConfig(name="vllm", das={"enable": True})
    with pytest.warns(UserWarning, match="method is not 'suffix'"):
        RolloutConfig(
            name="vllm",
            das={"enable": True},
            engine_kwargs={"vllm": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4}}},
        )
