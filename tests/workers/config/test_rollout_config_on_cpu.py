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

import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import CheckpointEngineConfig, RolloutConfig


def test_max_num_batched_tokens_raised_when_chunked_prefill_disabled():
    # vLLMHttpServer._validate_configs raises max_num_batched_tokens to max_model_len
    # when chunked prefill is off; that write must not hit the frozen-field guard.
    config = RolloutConfig(name="vllm", enable_chunked_prefill=False, max_num_batched_tokens=8192)
    config.max_model_len = 32768
    config.max_num_batched_tokens = config.max_model_len
    assert config.max_num_batched_tokens == 32768


def test_partial_kv_offload_is_opt_in():
    config = RolloutConfig(name="vllm")
    assert config.partial_rollout_kv_offload is False
    assert config.partial_rollout_kv_offload_headroom_gb == 64.0
    config = RolloutConfig(name="vllm", partial_rollout_kv_offload=True)
    assert config.partial_rollout_kv_offload is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "sglang"},
        {"free_cache_engine": False},
        {"enable_sleep_mode": False},
        {"data_parallel_size": 2},
        {"pipeline_model_parallel_size": 2},
        {"checkpoint_engine": CheckpointEngineConfig(backend="nccl")},
        {"partial_rollout_kv_offload_headroom_gb": -1},
        {"partial_rollout_kv_offload_headroom_gb": float("nan")},
        {"partial_rollout_kv_offload_headroom_gb": float("inf")},
        {"partial_rollout_kv_offload_headroom_gb": True},
    ],
)
def test_partial_kv_offload_rejects_unsupported_configs(kwargs):
    with pytest.raises(ValueError):
        RolloutConfig(**{"name": "vllm", "partial_rollout_kv_offload": True, **kwargs})


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_partial_kv_flag_requires_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        RolloutConfig(name="vllm", partial_rollout_kv_offload=value)


@pytest.mark.parametrize("dict_config", [False, True])
def test_partial_kv_offload_normalizes_disaggregation_before_validation(dict_config):
    """Hydra's mapping-shaped disaggregation block must not crash an enabled rollout."""
    disaggregation = {"enabled": False}
    if dict_config:
        disaggregation = OmegaConf.create(disaggregation)
    config = RolloutConfig(name="vllm", partial_rollout_kv_offload=True, disaggregation=disaggregation)
    assert config.disaggregation.enabled is False
    with pytest.raises(ValueError, match="disaggregation"):
        RolloutConfig(name="vllm", partial_rollout_kv_offload=True, disaggregation={"enabled": True})


def test_partial_kv_offload_accepts_the_hydra_instantiation_path():
    """The server instantiates nested targetless blocks with Hydra's partial conversion."""
    config = omega_conf_to_dataclass(
        OmegaConf.create(
            {
                "_target_": "verl.workers.config.rollout.RolloutConfig",
                "name": "vllm",
                "partial_rollout_kv_offload": True,
                "disaggregation": {"enabled": False},
                "checkpoint_engine": {
                    "_target_": "verl.workers.config.rollout.CheckpointEngineConfig",
                    "backend": "naive",
                },
                "mtp": None,
            }
        )
    )
    assert isinstance(config, RolloutConfig)
    assert config.disaggregation.enabled is False
