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

from verl.workers.config import RolloutConfig
from verl.workers.config.cache_pool import (
    KVCachePoolConfig,
    KVCachePoolMasterConfig,
    KVCachePoolStoreConfig,
)


def test_defaults_disabled_skip_validation():
    cfg = KVCachePoolConfig()
    assert cfg.enabled is False
    assert cfg.backend == "mooncake"
    assert cfg.python_hash_seed == 0
    assert cfg.master.auto_start is True
    assert cfg.master.log_master_output is False
    assert cfg.store.mode == "embedded"
    assert cfg.store.global_segment_size == "4GB"


def test_nested_dict_coercion():
    cfg = KVCachePoolConfig(
        master={"auto_start": False, "address": "10.0.0.1:50051"},
        store={"global_segment_size": "8GB"},
        connector={"save_decode_cache": True},
    )
    assert isinstance(cfg.master, KVCachePoolMasterConfig)
    assert cfg.master.address == "10.0.0.1:50051"
    assert isinstance(cfg.store, KVCachePoolStoreConfig)
    assert cfg.store.global_segment_size == "8GB"
    assert cfg.connector.save_decode_cache is True


def test_disabled_allows_illegal_mode():
    cfg = KVCachePoolConfig(enabled=False, store={"mode": "standalone-store"})
    assert cfg.enabled is False
    assert cfg.store.mode == "standalone-store"


def test_enabled_rejects_standalone_store():
    with pytest.raises(ValueError, match="embedded"):
        KVCachePoolConfig(enabled=True, store={"mode": "standalone-store"})


def test_enabled_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        KVCachePoolConfig(enabled=True, backend="bogus")


def test_enabled_memcache_not_implemented():
    with pytest.raises(NotImplementedError, match="mooncake"):
        KVCachePoolConfig(enabled=True, backend="memcache")


def test_auto_start_conflicts_with_config_path():
    with pytest.raises(ValueError, match="config_path"):
        KVCachePoolConfig(
            enabled=True,
            master={"auto_start": True},
            store={"config_path": "/tmp/mooncake.json"},
        )


def test_auto_start_false_requires_address_or_config_path():
    with pytest.raises(ValueError, match="address"):
        KVCachePoolConfig(enabled=True, master={"auto_start": False})


def test_auto_start_false_with_config_path_ok():
    KVCachePoolConfig(
        enabled=True,
        master={"auto_start": False},
        store={"config_path": "/tmp/mooncake.json"},
    )


def test_multi_tenants_requires_quota_fields():
    with pytest.raises(ValueError, match="tenant_quota"):
        KVCachePoolConfig(enabled=True, master={"enable_multi_tenants": True})


@pytest.mark.parametrize("backend", ["mooncake", "yuanrong"])
def test_use_layerwise_rejected_unless_memcache(backend):
    with pytest.raises(ValueError, match="use_layerwise"):
        KVCachePoolConfig(enabled=True, backend=backend, connector={"use_layerwise": True})


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_enabled_accepts_kv_load_failure_policy(policy):
    cfg = KVCachePoolConfig(enabled=True, kv_load_failure_policy=policy)
    assert cfg.kv_load_failure_policy == policy


def test_enabled_rejects_invalid_kv_load_failure_policy():
    with pytest.raises(ValueError, match="kv_load_failure_policy"):
        KVCachePoolConfig(enabled=True, kv_load_failure_policy="typo")


def test_disabled_allows_invalid_kv_load_failure_policy():
    cfg = KVCachePoolConfig(enabled=False, kv_load_failure_policy="typo")
    assert cfg.kv_load_failure_policy == "typo"


def test_enabled_rejects_negative_master_port():
    with pytest.raises(ValueError, match="port"):
        KVCachePoolConfig(enabled=True, master={"port": -1})


def test_enabled_allows_master_port_zero():
    cfg = KVCachePoolConfig(enabled=True, master={"port": 0})
    assert cfg.master.port == 0


def test_disabled_allows_negative_master_port():
    cfg = KVCachePoolConfig(enabled=False, master={"port": -1})
    assert cfg.master.port == -1


def test_rollout_cache_pool_default_disabled():
    cfg = RolloutConfig(name="vllm")
    assert cfg.cache_pool.enabled is False


def test_rollout_cache_pool_non_pd_ok():
    cfg = RolloutConfig(name="vllm", cache_pool={"enabled": True})
    assert cfg.cache_pool.enabled is True
    assert cfg.disaggregation.enabled is False


@pytest.mark.parametrize(
    ("connector", "match"),
    [
        ({"enable_store_tp_lcm": True}, "enable_store_tp_lcm"),
        ({"prefill_tp_sizes": [4, 2]}, "prefill_tp_sizes"),
        ({"save_decode_cache": True}, "save_decode_cache"),
        ({"consumer_is_to_put": True}, "consumer_is_to_put"),
        ({"consumer_is_to_load": True}, "consumer_is_to_load"),
        ({"prefill_pp_size": 2}, "prefill_pp_size"),
        ({"prefill_pp_layer_partition": "16,16"}, "prefill_pp_layer_partition"),
    ],
)
def test_rollout_cache_pool_non_pd_rejects_pd_only_fields(connector, match):
    with pytest.raises(ValueError, match=match):
        RolloutConfig(name="vllm", cache_pool={"enabled": True, "connector": connector})


def test_rollout_cache_pool_pd_allows_save_decode_cache():
    cfg = RolloutConfig(
        name="vllm",
        disaggregation={"enabled": True, "transfer_backend": "mooncake"},
        cache_pool={"enabled": True, "connector": {"save_decode_cache": True}},
    )
    assert cfg.cache_pool.connector.save_decode_cache is True


def test_rollout_cache_pool_allows_nixl_transfer():
    cfg = RolloutConfig(
        name="vllm",
        disaggregation={"enabled": True, "transfer_backend": "nixl"},
        cache_pool={"enabled": True},
    )
    assert cfg.cache_pool.enabled is True
    assert cfg.disaggregation.transfer_backend == "nixl"


def test_rollout_cache_pool_rejects_unsupported_transfer():
    with pytest.raises(ValueError, match="mooncake"):
        RolloutConfig(
            name="vllm",
            disaggregation={"enabled": True, "transfer_backend": "mori"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_rejects_sglang():
    with pytest.raises(ValueError, match="vllm"):
        RolloutConfig(
            name="sglang",
            disaggregation={"enabled": True, "transfer_backend": "mooncake"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_requires_prefix_caching():
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        RolloutConfig(
            name="vllm",
            enable_prefix_caching=False,
            disaggregation={"enabled": True, "transfer_backend": "mooncake"},
            cache_pool={"enabled": True},
        )


def test_rollout_cache_pool_vllm_pd_mooncake_ok():
    cfg = RolloutConfig(
        name="vllm",
        disaggregation={"enabled": True, "transfer_backend": "mooncake"},
        cache_pool={"enabled": True},
    )
    assert cfg.cache_pool.enabled is True
