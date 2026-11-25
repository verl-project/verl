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
"""CPU unit tests for MooncakeStoreConnector rollout config, YAML preset, and Ray runtime_env forwarding."""

import json
import os

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.workers.config import RolloutConfig


def test_rollout_config_mooncake_store_salted_passthrough():
    kv_transfer_cfg = {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "save_decode_cache": True,
            "load_async": True,
            "lookup_async": True,
            "lookup_prefetch_limit": 64,
            "lookup_prefetch_ttl_s": 30.0,
            "lookup_timeout_s": 5.0,
        },
    }
    cfg = RolloutConfig(
        name="vllm",
        mode="async",
        external_kv_cache_invalidation="salt",
        engine_kwargs={"vllm": {"kv_transfer_config": json.dumps(kv_transfer_cfg)}},
    )
    assert cfg.external_kv_cache_invalidation == "salt"


def test_rollout_config_rejects_unsalted_connector_under_salt_mode():
    with pytest.raises(ValueError, match="external_kv_cache_invalidation='salt'"):
        RolloutConfig(
            name="vllm",
            mode="async",
            external_kv_cache_invalidation="salt",
            engine_kwargs={"vllm": {"kv_transfer_config": '{"kv_connector": "LMCacheConnectorV1"}'}},
        )


def test_kv_offload_mooncake_hydra_preset():
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../verl/trainer/config"))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="ppo_trainer", overrides=["+kv_offload=mooncake"])
    rollout = cfg.actor_rollout_ref.rollout
    assert rollout.external_kv_cache_invalidation == "salt"
    kv_cfg = rollout.engine_kwargs.vllm.kv_transfer_config
    assert kv_cfg.kv_connector == "MooncakeStoreConnector"
    assert kv_cfg.kv_role == "kv_both"
    assert kv_cfg.kv_load_failure_policy == "recompute"
    assert kv_cfg.kv_connector_extra_config.save_decode_cache is True
    assert kv_cfg.kv_connector_extra_config.lookup_async is True
    assert kv_cfg.kv_connector_extra_config.lookup_prefetch_limit == 64
    env_vars = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    assert env_vars.PYTHONHASHSEED == "0"
    assert env_vars.MC_ENABLE_DEST_DEVICE_AFFINITY == "1"


def test_ppo_ray_runtime_env_forwards_mooncake_vars(monkeypatch):
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "/tmp/mooncake_config.json")
    monkeypatch.setenv("MC_MTU", "4096")
    monkeypatch.setenv("VLLM_MOONCAKE_LOAD_RECV_THREADS", "2")
    monkeypatch.setenv("MC_EMPTY_VAR", "")
    env_vars = get_ppo_ray_runtime_env()["env_vars"]
    assert env_vars["MOONCAKE_CONFIG_PATH"] == "/tmp/mooncake_config.json"
    assert env_vars["MC_MTU"] == "4096"
    assert env_vars["VLLM_MOONCAKE_LOAD_RECV_THREADS"] == "2"
    assert "MC_EMPTY_VAR" not in env_vars
