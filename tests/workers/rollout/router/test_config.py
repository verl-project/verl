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

"""Tests for llm_router config dataclasses and parsing.

Per §5 of detailed_config.md, each module has 5 test categories:
① Input/output normal cases
② Input/output abnormal cases
③ Hydra parsing normal cases
④ Hydra parsing abnormal cases
⑤ Other cases
"""

from __future__ import annotations

import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from verl.workers.rollout.router.kvcaware.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from verl.workers.rollout.router.kvcaware.types import SlowCut

# Default collector_names for strategy construction (required field)
_CN = ["vllm_zmq"]


# ============================================================
# 5.1 StrategyConfig / KVCAwareStrategyConfig
# ============================================================

# -- ① Input/output normal cases --


pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestStrategyNormalInput:
    """S01-S12: normal input/output cases"""

    def test_s01_weight_1_0(self):
        """
        Feature: weight field parses normally
        Description: construct KVCAwareStrategyConfig with weight=1.0
        Expectation: cfg.weight == 1.0
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.weight == 1.0

    def test_s02_weight_0_7(self):
        """
        Feature: weight field accepts decimal values
        Description: construct KVCAwareStrategyConfig with weight=0.7
        Expectation: cfg.weight == 0.7
        """
        cfg = KVCAwareStrategyConfig(weight=0.7, collector_names=_CN)
        assert cfg.weight == 0.7

    def test_s03_alpha_0_7(self):
        """
        Feature: alpha field parses normally
        Description: construct KVCAwareStrategyConfig with alpha=0.7
        Expectation: cfg.alpha == 0.7
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=0.7, collector_names=_CN)
        assert cfg.alpha == 0.7

    def test_s04_alpha_0_pure_load(self):
        """
        Feature: alpha=0.0 yields pure-load scoring
        Description: construct KVCAwareStrategyConfig with alpha=0.0
        Expectation: cfg.alpha == 0.0
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=0.0, collector_names=_CN)
        assert cfg.alpha == 0.0

    def test_s05_alpha_1_pure_cache(self):
        """
        Feature: alpha=1.0 yields pure-cache scoring
        Description: construct KVCAwareStrategyConfig with alpha=1.0
        Expectation: cfg.alpha == 1.0
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=1.0, collector_names=_CN)
        assert cfg.alpha == 1.0

    def test_s06_alpha_default_0_7(self):
        """
        Feature: alpha uses default when omitted
        Description: construct KVCAwareStrategyConfig without alpha
        Expectation: cfg.alpha == 0.7
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.alpha == 0.7

    def test_s07_load_threshold_0_5(self):
        """
        Feature: load_threshold field parses normally
        Description: construct KVCAwareStrategyConfig with load_threshold=0.5
        Expectation: cfg.load_threshold == 0.5
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, load_threshold=0.5, collector_names=_CN)
        assert cfg.load_threshold == 0.5

    def test_s08_load_threshold_default_0_9(self):
        """
        Feature: load_threshold uses default when omitted
        Description: construct KVCAwareStrategyConfig without load_threshold
        Expectation: cfg.load_threshold == 0.9 (load ceiling; overload when load > threshold)
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.load_threshold == 0.9

    def test_s09_layer_weights_dict(self):
        """
        Feature: layer_weights custom dict parses normally (three tiers: gpu/cpu/ssd)
        Description: construct strategy config with layer_weights={"gpu":0.7,"cpu":0.2,"ssd":0.1}
        Expectation: cfg.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
        """
        cfg = KVCAwareStrategyConfig(
            weight=1.0, layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}, collector_names=_CN
        )
        assert cfg.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}

    def test_s10_layer_weights_default(self):
        """
        Feature: layer_weights uses default when omitted
        Description: construct KVCAwareStrategyConfig without layer_weights
        Expectation: cfg.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}

    def test_s10b_memory_overload_filter_default_true(self):
        """
        Feature: memory_overload_filter uses default when omitted
        Description: construct KVCAwareStrategyConfig without memory_overload_filter
        Expectation: cfg.memory_overload_filter is True
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.memory_overload_filter is True

    def test_s10c_memory_overload_filter_false(self):
        """
        Feature: memory_overload_filter=False parses normally
        Description: construct strategy config with memory_overload_filter=False
        Expectation: cfg.memory_overload_filter is False
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, memory_overload_filter=False, collector_names=_CN)
        assert cfg.memory_overload_filter is False

    def test_s10d_slow_cut_default_prefix_load_aware(self):
        """
        Feature: slow_cut uses default when omitted
        Description: construct KVCAwareStrategyConfig without slow_cut
        Expectation: cfg.slow_cut == SlowCut.PREFIX_LOAD_AWARE
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=_CN)
        assert cfg.slow_cut == SlowCut.PREFIX_LOAD_AWARE

    def test_s10e_slow_cut_str_coerced(self):
        """
        Feature: slow_cut accepts a YAML string and coerces to SlowCut
        Description: construct strategy config with slow_cut="least-inflight"
        Expectation: cfg.slow_cut == SlowCut.LEAST_INFLIGHT
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, slow_cut="least-inflight", collector_names=_CN)
        assert cfg.slow_cut == SlowCut.LEAST_INFLIGHT

    def test_s11_collector_names_bind(self):
        """
        Feature: collector_names binds multiple collectors
        Description: construct strategy config with collector_names=["vllm_zmq","mooncake_prometheus"]
        Expectation: cfg.collector_names has length 2 and matches
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=["vllm_zmq", "mooncake_prometheus"])
        assert cfg.collector_names == ["vllm_zmq", "mooncake_prometheus"]
        assert len(cfg.collector_names) == 2

    def test_s12_multi_strategy_weights_sum_to_1(self):
        """
        Feature: sum of multi-strategy weights is ~1.0
        Description: construct two strategies with weights 0.6 and 0.4
        Expectation: sum of weights == pytest.approx(1.0)
        """
        s1 = KVCAwareStrategyConfig(weight=0.6, collector_names=["vllm_zmq"])
        s2 = KVCAwareStrategyConfig(weight=0.4, collector_names=["mooncake_prometheus"])
        assert s1.weight + s2.weight == pytest.approx(1.0)

    def test_s12b_collector_names_missing_required(self):
        """
        Feature: construction fails when collector_names is missing
        Description: construct strategy config without the required collector_names
        Expectation: raises TypeError
        """
        with pytest.raises(TypeError):
            KVCAwareStrategyConfig(weight=1.0)


# -- ② Input/output abnormal cases --


class TestStrategyAbnormalInput:
    """S13-S23: abnormal input/output cases"""

    def test_s13_weight_zero(self):
        """
        Feature: weight=0 triggers validation error
        Description: construct strategy config with weight=0.0
        Expectation: raises ConfigError matching "weight"
        """
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=0.0, collector_names=_CN)

    def test_s14_weight_above_1(self):
        """
        Feature: weight above upper bound triggers validation error
        Description: construct strategy config with weight=1.5
        Expectation: raises ConfigError matching "weight"
        """
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=1.5, collector_names=_CN)

    def test_s15_weight_negative(self):
        """
        Feature: negative weight triggers validation error
        Description: construct strategy config with weight=-1.0
        Expectation: raises ConfigError matching "weight"
        """
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=-1.0, collector_names=_CN)

    def test_s16_weight_string_type_error(self):
        """
        Feature: construction fails on wrong weight type
        Description: construct strategy config with weight="0.7"
        Expectation: raises TypeError
        """
        with pytest.raises(TypeError):
            KVCAwareStrategyConfig(weight="0.7", collector_names=_CN)

    def test_s17_weight_missing_required(self):
        """
        Feature: construction fails when weight is missing
        Description: construct strategy config without the required weight
        Expectation: raises TypeError
        """
        with pytest.raises(TypeError):
            KVCAwareStrategyConfig(collector_names=_CN)

    def test_s18_load_threshold_zero(self):
        """
        Feature: load_threshold=0 triggers validation error
        Description: construct strategy config with load_threshold=0
        Expectation: raises ConfigError matching "load_threshold"
        """
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=0, collector_names=_CN)

    def test_s19_load_threshold_negative(self):
        """
        Feature: negative load_threshold triggers validation error
        Description: construct strategy config with load_threshold=-1
        Expectation: raises ConfigError matching "load_threshold"
        """
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=-1, collector_names=_CN)

    def test_s19b_load_threshold_one_or_above(self):
        """
        Feature: load_threshold >= 1 triggers validation error
        Description: construct strategy config with load_threshold=1.0 and load_threshold=80
        Expectation: raises ConfigError matching "load_threshold"
        """
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=1.0, collector_names=_CN)
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=80, collector_names=_CN)

    def test_s20_layer_weights_non_gpu_cpu_ssd_key(self):
        """
        Feature: layer_weights with illegal key triggers validation error
        Description: construct strategy config with a "disk" key in layer_weights
        Expectation: raises ConfigError matching "layer_weights"
        """
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"gpu": 0.7, "cpu": 0.2, "disk": 0.1}, collector_names=_CN)

    def test_s21_layer_weights_missing_key(self):
        """
        Feature: layer_weights missing a required tier triggers validation error
        Description: construct strategy config with layer_weights missing "ssd"
        Expectation: raises ConfigError matching "layer_weights"
        """
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"gpu": 0.7, "cpu": 0.3}, collector_names=_CN)

    def test_s21b_layer_weights_sum_below_one(self):
        """
        Feature: layer_weights values summing below 1.0 trigger validation error
        Description: construct strategy config with weights summing to 0.8
        Expectation: raises ConfigError matching "layer_weights"
        """
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"gpu": 0.5, "cpu": 0.2, "ssd": 0.1}, collector_names=_CN)

    def test_s21c_layer_weights_sum_above_one(self):
        """
        Feature: layer_weights values summing above 1.0 trigger validation error
        Description: construct strategy config with weights summing to 1.1
        Expectation: raises ConfigError matching "layer_weights"
        """
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.2}, collector_names=_CN)

    def test_s22_collector_names_not_list_type(self):
        """
        Feature: non-list collector_names triggers validation error
        Description: construct strategy config with collector_names="vllm_zmq"
        Expectation: raises ConfigError matching "collector_names must be a list"
        """
        with pytest.raises(ConfigError, match="collector_names must be a list"):
            KVCAwareStrategyConfig(weight=1.0, collector_names="vllm_zmq")

    def test_s22b_memory_overload_filter_non_bool(self):
        """
        Feature: non-bool memory_overload_filter triggers validation error
        Description: construct strategy config with memory_overload_filter="yes"
        Expectation: raises ConfigError matching "memory_overload_filter"
        """
        with pytest.raises(ConfigError, match="memory_overload_filter"):
            KVCAwareStrategyConfig(weight=1.0, memory_overload_filter="yes", collector_names=_CN)

    def test_s22c_slow_cut_invalid_value(self):
        """
        Feature: unknown slow_cut value triggers validation error
        Description: construct strategy config with slow_cut="random"
        Expectation: raises ConfigError matching "slow_cut"
        """
        with pytest.raises(ConfigError, match="slow_cut"):
            KVCAwareStrategyConfig(weight=1.0, slow_cut="random", collector_names=_CN)

    def test_s23_multi_strategy_weights_not_sum_to_1(self):
        """
        Feature: multi-strategy weights not summing to 1.0 triggers validation error
        Description: from_config with two strategies each weighted 0.4
        Expectation: raises ConfigError matching "weight"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.4,
                        "collector_names": ["vllm_zmq"],
                    },
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.4,
                        "collector_names": ["mooncake_prometheus"],
                    },
                ],
            }
        )
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareConfig.from_config(kwargs)

    def test_s23b_strategies_empty_list(self):
        """
        Feature: empty strategies list triggers validation error
        Description: from_config with strategies=[]
        Expectation: raises ConfigError matching "strategies"
        """
        kwargs = OmegaConf.create({"strategies": []})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_config(kwargs)

    def test_s23c_strategies_not_list(self):
        """
        Feature: illegal strategies type triggers validation error
        Description: from_config with strategies="kvc_aware"
        Expectation: raises ConfigError matching "strategies"
        """
        kwargs = OmegaConf.create({"strategies": "kvc_aware"})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_config(kwargs)

    def test_s23d_strategy_item_not_dict(self):
        """
        Feature: non-dict strategy item triggers validation error
        Description: from_config with strategies=["kvc_aware"]
        Expectation: raises ConfigError matching "strategies"
        """
        kwargs = OmegaConf.create({"strategies": ["kvc_aware"]})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_config(kwargs)


# -- ③ Hydra parsing normal cases --


class TestStrategyHydraNormal:
    """S24-S26: Hydra parsing normal cases"""

    def test_s24_strategy_instantiate(self):
        """
        Feature: strategy instantiates via _target_
        Description: instantiate a strategy OmegaConf entry with _target_
        Expectation: result is a KVCAwareStrategyConfig instance
        """
        entry = OmegaConf.create(
            {
                "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                "weight": 1.0,
                "collector_names": ["vllm_zmq"],
            }
        )
        from hydra.utils import instantiate

        result = instantiate(entry)
        assert isinstance(result, KVCAwareStrategyConfig)

    def test_s25_strategy_inherit_base(self):
        """
        Feature: strategy instance inherits StrategyConfig base
        Description: instantiate a strategy entry then assert base type
        Expectation: result is a StrategyConfig instance and weight==1.0
        """
        entry = OmegaConf.create(
            {
                "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                "weight": 1.0,
                "collector_names": ["vllm_zmq"],
            }
        )
        from hydra.utils import instantiate

        result = instantiate(entry)
        assert isinstance(result, StrategyConfig)
        assert result.weight == 1.0

    def test_s26_strategy_defaults_filled(self):
        """
        Feature: defaults are filled after strategy instantiation
        Description: instantiate a strategy entry with only weight/collector_names
        Expectation: alpha/load_threshold/layer_weights equal defaults
        """
        entry = OmegaConf.create(
            {
                "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                "weight": 1.0,
                "collector_names": ["vllm_zmq"],
            }
        )
        from hydra.utils import instantiate

        result = instantiate(entry)
        assert result.alpha == 0.7
        assert result.load_threshold == 0.9
        assert result.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
        assert result.collector_names == ["vllm_zmq"]


# -- ④ Hydra parsing abnormal cases --


class TestStrategyHydraAbnormal:
    """S27-S30: Hydra parsing abnormal cases"""

    def test_s27_target_module_not_exist(self):
        """
        Feature: instantiation fails when _target_ module does not exist
        Description: instantiate an entry whose _target_ points to nonexistent.Module.Class
        Expectation: raises InstantiationException/ImportError/ConfigError
        """
        entry = OmegaConf.create({"_target_": "nonexistent.Module.Class"})
        from hydra.utils import instantiate

        with pytest.raises((InstantiationException, ImportError, ConfigError)):
            instantiate(entry)

    def test_s28_target_class_not_exist(self):
        """
        Feature: instantiation fails when _target_ class does not exist
        Description: instantiate an entry whose _target_ points to config.NonExistClass
        Expectation: raises InstantiationException/AttributeError/ConfigError
        """
        entry = OmegaConf.create(
            {
                "_target_": "verl.workers.rollout.router.kvcaware.config.NonExistClass",
            }
        )
        from hydra.utils import instantiate

        with pytest.raises((InstantiationException, AttributeError, ConfigError)):
            instantiate(entry)

    def test_s29_target_missing(self):
        """
        Feature: strategy item missing _target_ triggers validation error
        Description: from_config with a strategy item that has no _target_
        Expectation: raises ConfigError matching "_target_"
        """
        kwargs = OmegaConf.create({"strategies": [{"weight": 1.0}]})
        with pytest.raises(ConfigError, match="_target_"):
            KVCAwareConfig.from_config(kwargs)

    def test_s30_target_not_strategy_subclass(self):
        """
        Feature: _target_ pointing to a non-strategy subclass triggers validation error
        Description: from_config with a strategy item whose _target_ is CacheStoreConfig
        Expectation: raises ConfigError matching "StrategyConfig"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.CacheStoreConfig",
                        "kv_cache_store_type": "list",
                        "ttl": 30,
                    },
                ],
            }
        )
        with pytest.raises(ConfigError, match="StrategyConfig"):
            KVCAwareConfig.from_config(kwargs)


# -- ⑤ Other cases --


class TestStrategyOther:
    """S31-S32: other cases"""

    def test_s31_empty_config_no_strategy(self):
        """
        Feature: empty config triggers strategies-required error
        Description: from_config with an empty OmegaConf dict
        Expectation: raises ConfigError matching "strategies"
        """
        kwargs = OmegaConf.create({})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_config(kwargs)

    def test_s32_strategies_null_error(self):
        """
        Feature: null strategies triggers required error
        Description: from_config with strategies=None
        Expectation: raises ConfigError matching "strategies"
        """
        kwargs = OmegaConf.create({"strategies": None})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_config(kwargs)


# ============================================================
# 5.2 CollectorConfig
# ============================================================

# -- ① Input/output normal cases --


class TestMetricsNormalInput:
    """M01-M09: normal input/output cases"""

    # -- CollectorConfig: http_polling dict --

    def test_m01_http_polling_default(self):
        """
        Feature: CollectorConfig default http_polling values
        Description: construct CollectorConfig with no args
        Expectation: http_polling == {"polling_interval": 5.0, "http_timeout": 10.0}
        """
        cfg = CollectorConfig()
        assert cfg.http_polling == {"polling_interval": 5.0, "http_timeout": 10.0}

    def test_m02_http_polling_custom(self):
        """
        Feature: http_polling accepts custom values
        Description: construct CollectorConfig with custom http_polling
        Expectation: http_polling equals custom values
        """
        cfg = CollectorConfig(http_polling={"polling_interval": 3.0, "http_timeout": 15.0})
        assert cfg.http_polling == {"polling_interval": 3.0, "http_timeout": 15.0}

    # -- CollectorConfig: long_connection dict --

    def test_m03_long_connection_default(self):
        """
        Feature: CollectorConfig default long_connection values
        Description: construct CollectorConfig with no args
        Expectation: long_connection equals the four default params
        """
        cfg = CollectorConfig()
        assert cfg.long_connection == {
            "base_retry_delay": 1.0,
            "max_retry_delay": 30.0,
            "max_retry_attempts": 5,
            "retry_backoff_factor": 2.0,
        }

    def test_m04_long_connection_custom(self):
        """
        Feature: long_connection accepts custom values
        Description: construct CollectorConfig with custom long_connection
        Expectation: base_retry_delay/max_retry_delay equal custom values
        """
        cfg = CollectorConfig(
            long_connection={
                "base_retry_delay": 2.0,
                "max_retry_delay": 60.0,
                "max_retry_attempts": 10,
                "retry_backoff_factor": 3.0,
            }
        )
        assert cfg.long_connection["base_retry_delay"] == 2.0
        assert cfg.long_connection["max_retry_delay"] == 60.0

    # -- CollectorConfig base class is empty (placeholder) --

    def test_m08_collector_config_fields(self):
        """
        Feature: CollectorConfig exposes connection-type fields
        Description: inspect the dataclass field set of CollectorConfig
        Expectation: field set is {http_polling, long_connection}
        """
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(CollectorConfig)}
        assert field_names == {"http_polling", "long_connection"}

    def test_m09_collector_config_is_dataclass(self):
        """
        Feature: CollectorConfig is a dataclass
        Description: check the dataclass type of CollectorConfig
        Expectation: is_dataclass returns True
        """
        import dataclasses

        assert dataclasses.is_dataclass(CollectorConfig)


# -- ② Input/output abnormal cases --


class TestMetricsAbnormalInput:
    """M10-M17: abnormal input/output cases"""

    def test_m10_polling_interval_zero(self):
        """
        Feature: polling_interval=0 triggers validation error
        Description: construct CollectorConfig with polling_interval=0
        Expectation: raises ConfigError matching "polling_interval"
        """
        with pytest.raises(ConfigError, match="polling_interval"):
            CollectorConfig(http_polling={"polling_interval": 0, "http_timeout": 10})

    def test_m11_polling_interval_negative(self):
        """
        Feature: negative polling_interval triggers validation error
        Description: construct CollectorConfig with polling_interval=-1
        Expectation: raises ConfigError matching "polling_interval"
        """
        with pytest.raises(ConfigError, match="polling_interval"):
            CollectorConfig(http_polling={"polling_interval": -1, "http_timeout": 10})

    def test_m12_http_timeout_zero(self):
        """
        Feature: http_timeout=0 triggers validation error
        Description: construct CollectorConfig with http_timeout=0
        Expectation: raises ConfigError matching "http_timeout"
        """
        with pytest.raises(ConfigError, match="http_timeout"):
            CollectorConfig(http_polling={"polling_interval": 5, "http_timeout": 0})

    def test_m13_base_retry_delay_negative(self):
        """
        Feature: negative base_retry_delay triggers validation error
        Description: construct CollectorConfig with base_retry_delay=-1
        Expectation: raises ConfigError matching "base_retry_delay"
        """
        with pytest.raises(ConfigError, match="base_retry_delay"):
            CollectorConfig(long_connection={"base_retry_delay": -1})

    def test_m14_max_retry_delay_less_than_base(self):
        """
        Feature: max_retry_delay below base triggers validation error
        Description: construct CollectorConfig with max_retry_delay=3 below base=5
        Expectation: raises ConfigError matching "max_retry_delay"
        """
        with pytest.raises(ConfigError, match="max_retry_delay"):
            CollectorConfig(
                long_connection={
                    "base_retry_delay": 5,
                    "max_retry_delay": 3,
                }
            )

    def test_m15_max_retry_attempts_zero(self):
        """
        Feature: max_retry_attempts=0 triggers validation error
        Description: construct CollectorConfig with max_retry_attempts=0
        Expectation: raises ConfigError matching "max_retry_attempts"
        """
        with pytest.raises(ConfigError, match="max_retry_attempts"):
            CollectorConfig(long_connection={"max_retry_attempts": 0})

    def test_m16_retry_backoff_factor_negative(self):
        """
        Feature: negative retry_backoff_factor triggers validation error
        Description: construct CollectorConfig with retry_backoff_factor=-1
        Expectation: raises ConfigError matching "retry_backoff_factor"
        """
        with pytest.raises(ConfigError, match="retry_backoff_factor"):
            CollectorConfig(long_connection={"retry_backoff_factor": -1})


# -- ③ Hydra parsing normal cases --
# (No collector sub-config instantiate — concrete subclasses removed)


# -- ⑤ Other cases --


class TestMetricsOther:
    """M18-M20: other cases"""

    def test_m18_collector_defaults_with_strategy(self):
        """
        Feature: collector takes defaults when only strategies given
        Description: from_config with a config containing only strategies
        Expectation: http_polling/long_connection equal defaults
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert result.collector.http_polling == {"polling_interval": 5.0, "http_timeout": 10.0}
        assert result.collector.long_connection == {
            "base_retry_delay": 1.0,
            "max_retry_delay": 30.0,
            "max_retry_attempts": 5,
            "retry_backoff_factor": 2.0,
        }

    def test_m19_collector_null_default(self):
        """
        Feature: collector takes defaults when null
        Description: from_config with collector=None
        Expectation: result.collector is a CollectorConfig instance
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": None,
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.collector, CollectorConfig)

    def test_m20_http_polling_override(self):
        """
        Feature: collector.http_polling overrides defaults
        Description: from_config with config overriding polling_interval
        Expectation: http_polling["polling_interval"] == 3.0
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": {
                    "http_polling": {"polling_interval": 3},
                },
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert result.collector.http_polling["polling_interval"] == 3.0


# ============================================================
# 5.3 CacheStoreConfig
# ============================================================

# -- ① Input/output normal cases --


class TestCacheStoreNormalInput:
    """C01-C05: normal input/output cases"""

    def test_c01_kv_cache_store_type_list(self):
        """
        Feature: kv_cache_store_type=list parses normally
        Description: construct CacheStoreConfig with kv_cache_store_type="list"
        Expectation: cfg.kv_cache_store_type == "list"
        """
        cfg = CacheStoreConfig(kv_cache_store_type="list")
        assert cfg.kv_cache_store_type == "list"

    def test_c02_kv_cache_store_type_radix_tree(self):
        """
        Feature: kv_cache_store_type=radix_tree parses normally
        Description: construct CacheStoreConfig with kv_cache_store_type="radix_tree"
        Expectation: cfg.kv_cache_store_type == "radix_tree"
        """
        cfg = CacheStoreConfig(kv_cache_store_type="radix_tree")
        assert cfg.kv_cache_store_type == "radix_tree"

    def test_c03_kv_cache_store_type_default(self):
        """
        Feature: kv_cache_store_type uses default when omitted
        Description: construct CacheStoreConfig with no args
        Expectation: cfg.kv_cache_store_type == "list"
        """
        cfg = CacheStoreConfig()
        assert cfg.kv_cache_store_type == "list"

    def test_c04_ttl_30(self):
        """
        Feature: ttl field parses normally
        Description: construct CacheStoreConfig with ttl=30.0
        Expectation: cfg.ttl == 30.0
        """
        cfg = CacheStoreConfig(ttl=30.0)
        assert cfg.ttl == 30.0

    def test_c05_ttl_default(self):
        """
        Feature: ttl uses default when omitted
        Description: construct CacheStoreConfig with no args
        Expectation: cfg.ttl == 30.0
        """
        cfg = CacheStoreConfig()
        assert cfg.ttl == 30.0


# -- ② Input/output abnormal cases --


class TestCacheStoreAbnormalInput:
    """C06-C09: abnormal input/output cases"""

    def test_c06_kv_cache_store_type_unknown(self):
        """
        Feature: illegal kv_cache_store_type triggers validation error
        Description: construct CacheStoreConfig with kv_cache_store_type="unknown"
        Expectation: raises ConfigError matching "kv_cache_store_type"
        """
        with pytest.raises(ConfigError, match="kv_cache_store_type"):
            CacheStoreConfig(kv_cache_store_type="unknown")

    def test_c07_ttl_zero(self):
        """
        Feature: ttl=0 triggers validation error
        Description: construct CacheStoreConfig with ttl=0
        Expectation: raises ConfigError matching "ttl"
        """
        with pytest.raises(ConfigError, match="ttl"):
            CacheStoreConfig(ttl=0)

    def test_c08_ttl_negative(self):
        """
        Feature: negative ttl triggers validation error
        Description: construct CacheStoreConfig with ttl=-1
        Expectation: raises ConfigError matching "ttl"
        """
        with pytest.raises(ConfigError, match="ttl"):
            CacheStoreConfig(ttl=-1)

    def test_c09_cache_store_not_dict(self):
        """
        Feature: non-dict cache_store triggers validation error
        Description: from_config with cache_store="list"
        Expectation: raises ConfigError matching "cache_store"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "cache_store": "list",
            }
        )
        with pytest.raises(ConfigError, match="cache_store"):
            KVCAwareConfig.from_config(kwargs)


# -- ⑤ Other cases --


class TestCacheStoreOther:
    """C10-C11: other cases"""

    def test_c10_cache_store_defaults_with_strategy(self):
        """
        Feature: cache_store takes defaults when only strategies given
        Description: from_config with a config containing only strategies
        Expectation: kv_cache_store_type/ttl equal defaults
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert result.cache_store.kv_cache_store_type == "list"
        assert result.cache_store.ttl == 30.0

    def test_c11_cache_store_null_default(self):
        """
        Feature: cache_store takes defaults when null
        Description: from_config with cache_store=None
        Expectation: result.cache_store is a CacheStoreConfig instance
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "cache_store": None,
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.cache_store, CacheStoreConfig)


# ============================================================
# 5.4 KVCAwareConfig top-level
# ============================================================

# -- ① Input/output normal cases --


class TestKVCAwareNormalInput:
    """K01: normal input/output cases"""

    def test_k01_full_config_normal_parse(self):
        """
        Feature: full config parses normally
        Description: from_config with a full config of strategies/collector/cache_store
        Expectation: result is a KVCAwareConfig and field types are correct
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "alpha": 0.7,
                        "collector_names": ["vllm_zmq", "mooncake_prometheus"],
                    },
                ],
                "collector": {
                    "http_polling": {"polling_interval": 5, "http_timeout": 10},
                    "long_connection": {
                        "base_retry_delay": 1.0,
                        "max_retry_delay": 30.0,
                        "max_retry_attempts": 5,
                        "retry_backoff_factor": 2.0,
                    },
                },
                "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result, KVCAwareConfig)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert isinstance(result.collector, CollectorConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)


# -- ② Input/output abnormal cases --


class TestKVCAwareAbnormalInput:
    """K02-K04: abnormal input/output cases"""

    def test_k02_multi_error_aggregation(self):
        """
        Feature: from_config raises on multiple errors
        Description: from_config with empty strategies/illegal collector/illegal cache_store
        Expectation: raises ConfigError and message contains relevant field names
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [],
                "collector": {"http_polling": {"polling_interval": -1}},
                "cache_store": {"ttl": 0},
            }
        )
        with pytest.raises(ConfigError) as exc_info:
            KVCAwareConfig.from_config(kwargs)
        error_msg = str(exc_info.value)
        assert "strategies" in error_msg or "polling_interval" in error_msg or "ttl" in error_msg

    def test_k03_collector_not_dict(self):
        """
        Feature: non-dict collector triggers validation error
        Description: from_config with collector="vllm"
        Expectation: raises ConfigError matching "collector"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": "vllm",
            }
        )
        with pytest.raises(ConfigError, match="collector"):
            KVCAwareConfig.from_config(kwargs)


# -- ③ Hydra parsing normal cases --


class TestKVCAwareHydraNormal:
    """K05-K08: Hydra parsing normal cases"""

    def test_k05_omega_conf_merge_to_kvc_aware_config(self):
        """
        Feature: OmegaConf input parses into KVCAwareConfig normally
        Description: from_config with an OmegaConf containing all three sections
        Expectation: result is a KVCAwareConfig instance
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": {
                    "http_polling": {"polling_interval": 5, "http_timeout": 10},
                },
                "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result, KVCAwareConfig)

    def test_k06_collector_auto_recursive_parse(self):
        """
        Feature: collector auto-recursive parsing
        Description: from_config with config overriding http_polling
        Expectation: result.collector is CollectorConfig and polling_interval==3.0
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": {"http_polling": {"polling_interval": 3}},
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.collector, CollectorConfig)
        assert result.collector.http_polling["polling_interval"] == 3.0

    def test_k07_cache_store_auto_recursive_parse(self):
        """
        Feature: cache_store auto-recursive parsing
        Description: from_config with config overriding cache_store
        Expectation: result.cache_store fields equal custom values
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "cache_store": {"kv_cache_store_type": "radix_tree", "ttl": 60},
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.cache_store, CacheStoreConfig)
        assert result.cache_store.kv_cache_store_type == "radix_tree"
        assert result.cache_store.ttl == 60.0

    def test_k08_strategies_dict_to_list_conversion(self):
        """
        Feature: strategies dict auto-converts to list
        Description: from_config with strategies in dict form
        Expectation: result.strategies is a list and element types/values are correct
        """
        kwargs = OmegaConf.create(
            {
                "strategies": {
                    "kvc_aware": {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                },
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 1
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert result.strategies[0].weight == 1.0


# -- ④ Hydra parsing abnormal cases --


class TestKVCAwareHydraAbnormal:
    """K09-K11: Hydra parsing abnormal cases"""

    def test_k09_top_level_unknown_keys_ignored(self):
        """
        Feature: from_config ignores top-level keys outside the config domain
        Description: from_config with a config carrying extra non-domain keys
        Expectation: result is KVCAwareConfig and the extra keys are dropped
        """
        kwargs = OmegaConf.create(
            {
                "_unknown_top_level": "ignored",
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result, KVCAwareConfig)
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(result)}
        assert "_unknown_top_level" not in field_names
        assert "router_strategy" not in field_names

    def test_k10_metrics_unknown_key(self):
        """
        Feature: collector with undefined key triggers parse error
        Description: from_config with collector containing unknown_key
        Expectation: raises exception matching "unknown_key"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
                "collector": {"unknown_key": 123, "http_polling": {"polling_interval": 5}},
            }
        )
        with pytest.raises(Exception, match="unknown_key"):
            KVCAwareConfig.from_config(kwargs)

    def test_k11_compact_repr(self):
        """
        Feature: KVCAwareConfig multi-line indented repr
        Description: from_config then call repr and inspect output
        Expectation: repr contains newline and starts with KVCAwareConfig(
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq", "mooncake_prometheus"],
                    },
                ],
                "collector": {
                    "http_polling": {"polling_interval": 5, "http_timeout": 10},
                    "long_connection": {
                        "base_retry_delay": 1.0,
                        "max_retry_delay": 30.0,
                        "max_retry_attempts": 5,
                        "retry_backoff_factor": 2.0,
                    },
                },
                "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        r = repr(result)
        assert "\n" in r
        assert r.startswith("KVCAwareConfig(")
        assert "weight=1.0" in r
        assert "collector_names" in r


# -- ⑤ Other cases --


class TestKVCAwareOther:
    """K12-K14: other cases"""

    def test_k12_only_strategies_defaults(self):
        """
        Feature: other fields take defaults when only strategies given
        Description: from_config with a config containing only strategies
        Expectation: collector/cache_store are their Config types
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.collector, CollectorConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)

    def test_k13_manual_instantiate_all_dataclass(self):
        """
        Feature: multi-strategy instances are all StrategyConfig
        Description: from_config with a config containing two strategies
        Expectation: each strategy is a StrategyConfig instance
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.7,
                        "collector_names": ["vllm_zmq"],
                    },
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.3,
                        "collector_names": ["vllm_prometheus"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        for s in result.strategies:
            assert isinstance(s, StrategyConfig)

    def test_k14_yaml_split_loading(self):
        """
        Feature: Hydra composition of real YAML parses end-to-end
        Description: compose ppo_trainer with kvcaware, then from_config on the strategy node
        Expectation: all fields match the YAML config
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_dir = str((Path(__file__).parent.parent.parent.parent.parent / "verl" / "trainer" / "config").resolve())
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="ppo_trainer",
                overrides=[
                    "rollout/router@actor_rollout_ref.rollout.router_config=kvcaware",
                    "actor_rollout_ref.rollout.router_strategy=kvcaware",
                ],
            )

        strat = cfg.actor_rollout_ref.rollout.router_config
        assert cfg.actor_rollout_ref.rollout.router_strategy == "kvcaware"

        result = KVCAwareConfig.from_config(strat)

        # ── strategies ──
        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 1
        strategy = result.strategies[0]
        assert isinstance(strategy, KVCAwareStrategyConfig)
        assert strategy.weight == 1.0
        assert strategy.alpha == 0.3
        assert strategy.load_threshold == 0.6
        assert strategy.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
        assert strategy.collector_names == [
            "vllm_zmq",
            "vllm_metrics",
            "sticky_stat",
            "inflight_stat",
        ]
        assert strategy.memory_overload_filter is True
        assert strategy.slow_cut == SlowCut.PREFIX_LOAD_AWARE
        assert result.collector.http_polling == {"polling_interval": 5.0, "http_timeout": 10.0}
        assert result.collector.long_connection == {
            "base_retry_delay": 1.0,
            "max_retry_delay": 30.0,
            "max_retry_attempts": 5,
            "retry_backoff_factor": 2.0,
        }

        # ── cache_store ──
        assert isinstance(result.cache_store, CacheStoreConfig)
        assert result.cache_store.kv_cache_store_type == "list"
        assert result.cache_store.ttl == 30.0


# ============================================================
# 5.5 Drop-in flow integration tests
# ============================================================


class TestDropInIntegration:
    """Drop-in flow integration tests (config side).

    VeRL drop-in: hydra.compose populates ``rollout.router_config`` from
    ``rollout/router/kvcaware.yaml``; ``get_router_handle`` dispatches by
    ``router_strategy`` → registry and passes that node to the Balancer, which
    runs ``KVCAwareConfig.from_config`` on it. Config side only; registry/
    dispatch is in test_router.py.
    """

    def _make_full_config(self) -> OmegaConf:
        """Helper: build a full config for drop-in tests.

        Mirrors the shape of ``verl/trainer/config/rollout/router/kvcaware.yaml``
        as it reaches ``KVCAwareConfig.from_config`` after Hydra composes it under
        the rollout ``router_config`` node (selected via ``router@router_config: kvcaware``).
        No ``router_class`` — the plugin_extension/FQN flow was retired.
        """
        return OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "alpha": 0.7,
                        "collector_names": ["vllm_zmq", "mooncake_prometheus"],
                    },
                ],
                "collector": {
                    "http_polling": {"polling_interval": 5, "http_timeout": 10},
                    "long_connection": {
                        "base_retry_delay": 1.0,
                        "max_retry_delay": 30.0,
                        "max_retry_attempts": 5,
                        "retry_backoff_factor": 2.0,
                    },
                },
                "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
            }
        )

    def test_d05_compose_expands_defaults(self):
        """
        Feature: hydra.compose expands the kvcaware router config
        Description: compose ppo_trainer with +rollout@...router_config=kvcaware
        Expectation: strategies/collector/cache_store all present under rollout.router_config
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_dir = str((Path(__file__).parent.parent.parent.parent.parent / "verl" / "trainer" / "config").resolve())
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="ppo_trainer",
                overrides=[
                    "rollout/router@actor_rollout_ref.rollout.router_config=kvcaware",
                    "actor_rollout_ref.rollout.router_strategy=kvcaware",
                ],
            )
        strat = cfg.actor_rollout_ref.rollout.router_config
        assert "strategies" in strat and "collector" in strat and "cache_store" in strat
        assert cfg.actor_rollout_ref.rollout.router_strategy == "kvcaware"
        assert "http_polling" in strat.collector
        assert "long_connection" in strat.collector

    def test_d08_balancer_from_config(self):
        """
        Feature: from_config fully parses drop-in config
        Description: from_config with config generated by _make_full_config
        Expectation: result and field types are correct
        """
        cfg = self._make_full_config()
        result = KVCAwareConfig.from_config(cfg)
        assert isinstance(result, KVCAwareConfig)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert isinstance(result.collector, CollectorConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)

    def test_d09_strategy_subtarget_instantiate(self):
        """
        Feature: strategy sub-layer _target_ instantiates correctly
        Description: take strategies[0] after from_config
        Expectation: type is KVCAwareStrategyConfig and fields are correct
        """
        cfg = self._make_full_config()
        result = KVCAwareConfig.from_config(cfg)
        s = result.strategies[0]
        assert type(s) is KVCAwareStrategyConfig
        assert s.collector_names == ["vllm_zmq", "mooncake_prometheus"]
        assert s.weight == 1.0

    def test_d11_from_config_drops_non_domain_top_keys(self):
        """
        Feature: from_config drops non-domain top-level keys
        Description: from_config with a config carrying router_strategy/_unknown
        Expectation: result field set excludes those keys
        """
        cfg = self._make_full_config()
        result = KVCAwareConfig.from_config(cfg)
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(result)}
        assert "router_strategy" not in field_names
        assert "_unknown_top_level" not in field_names
        assert not hasattr(result, "router_strategy")

    def test_d12_omegaconf_dictconfig_input(self):
        """
        Feature: OmegaConf DictConfig input parses normally
        Description: from_config with an OmegaConf-typed config
        Expectation: strategies[0] is a KVCAwareStrategyConfig instance
        """
        cfg = self._make_full_config()
        result = KVCAwareConfig.from_config(cfg)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)

    def test_d13_plain_dict_input(self):
        """
        Feature: plain dict input parses normally
        Description: to_container to plain dict then from_config
        Expectation: strategies[0] type/weight are correct
        """
        cfg = OmegaConf.to_container(self._make_full_config(), resolve=True)
        result = KVCAwareConfig.from_config(cfg)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert result.strategies[0].weight == 1.0

    def test_d14_yaml_file_e2e(self):
        """
        Feature: YAML file parses end-to-end
        Description: compose ppo_trainer with kvcaware, then from_config
        Expectation: all field types/values are correct and router_strategy is dropped
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_dir = str((Path(__file__).parent.parent.parent.parent.parent / "verl" / "trainer" / "config").resolve())

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="ppo_trainer",
                overrides=[
                    "rollout/router@actor_rollout_ref.rollout.router_config=kvcaware",
                    "actor_rollout_ref.rollout.router_strategy=kvcaware",
                ],
            )

        # Step 2: to_container → plain dict (VeRL input form). The kvcaware config
        # nests under rollout.router_config; from_config takes that sub-tree directly.
        strat = cfg.actor_rollout_ref.rollout.router_config
        cfg_dict = OmegaConf.to_container(strat, resolve=True)
        assert isinstance(cfg_dict, dict)
        assert cfg.actor_rollout_ref.rollout.router_strategy == "kvcaware"
        # router_strategy lives at the rollout level, not inside the strategy node.
        assert "router_strategy" not in cfg_dict

        # Step 3: from_config
        result = KVCAwareConfig.from_config(cfg_dict)

        # Step 4: validate
        assert isinstance(result, KVCAwareConfig)
        assert len(result.strategies) == 1
        s = result.strategies[0]
        assert isinstance(s, KVCAwareStrategyConfig)
        assert s.weight == 1.0
        assert s.alpha == 0.3
        assert s.collector_names == ["vllm_zmq", "vllm_metrics", "sticky_stat", "inflight_stat"]
        assert result.collector.http_polling["polling_interval"] == 5.0
        assert result.collector.long_connection["max_retry_attempts"] == 5
        assert result.cache_store.kv_cache_store_type == "list"
        assert result.cache_store.ttl == 30.0
        # router_strategy does not appear in KVCAwareConfig fields
        import dataclasses

        assert "router_strategy" not in {f.name for f in dataclasses.fields(result)}
