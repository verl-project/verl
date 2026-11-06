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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = [
    "KVCachePoolConfig",
    "KVCachePoolMasterConfig",
    "KVCachePoolStoreConfig",
    "KVCachePoolConnectorConfig",
]

_ALLOWED_BACKENDS = ("mooncake", "memcache", "yuanrong")
_ALLOWED_STORE_MODES = ("embedded",)
_ALLOWED_KV_LOAD_FAILURE_POLICIES = ("recompute", "fail")


def _coerce(value, cls):
    if isinstance(value, cls):
        return value
    return cls(**dict(value))


@dataclass
class KVCachePoolMasterConfig(BaseConfig):
    """Mooncake master lifecycle settings for the job-level KV cache pool."""

    auto_start: bool = True
    address: Optional[str] = None
    port: Optional[int] = None
    eviction_high_watermark_ratio: Optional[float] = None
    eviction_ratio: Optional[float] = None
    default_kv_lease_ttl: Optional[int] = None
    client_ttl: Optional[int] = None
    log_master_output: bool = False
    enable_multi_tenants: bool = False
    tenant_quota_connector_type: Optional[str] = None
    tenant_quota_connector_uri: Optional[str] = None


@dataclass
class KVCachePoolStoreConfig(BaseConfig):
    """Mooncake store JSON settings for the job-level KV cache pool."""

    config_path: Optional[str] = None
    mode: str = "embedded"
    protocol: Optional[str] = None
    metadata_server: str = "P2PHANDSHAKE"
    global_segment_size: str = "4GB"
    local_buffer_size: str = "4GB"
    device_name: str = ""
    tenant_id: str = "default"
    enable_offload: bool = False
    ssd_offload_path: Optional[str] = None
    preferred_segment: bool = False
    prefer_alloc_in_same_node: bool = True


@dataclass
class KVCachePoolConnectorConfig(BaseConfig):
    """Store sub-connector extra_config fields for MultiConnector."""

    load_async: Optional[bool] = None
    lookup_async: bool = False
    cache_prefix: str = ""
    save_decode_cache: bool = False
    store_tp_size: Optional[int] = None
    enable_store_tp_lcm: Optional[bool] = None
    prefill_tp_sizes: Optional[list[int]] = None
    consumer_is_to_put: bool = False
    consumer_is_to_load: bool = False
    use_layerwise: bool = False
    prefill_pp_size: Optional[int] = None
    prefill_pp_layer_partition: Optional[str] = None


@dataclass
class KVCachePoolConfig(BaseConfig):
    """KV cache pool knobs for vLLM PD MultiConnector and non-PD Store."""

    enabled: bool = False
    backend: str = "mooncake"
    python_hash_seed: int = 0
    kv_load_failure_policy: Optional[str] = None
    extra_config: dict[str, Any] = field(default_factory=dict)
    master: KVCachePoolMasterConfig = field(default_factory=KVCachePoolMasterConfig)
    store: KVCachePoolStoreConfig = field(default_factory=KVCachePoolStoreConfig)
    connector: KVCachePoolConnectorConfig = field(default_factory=KVCachePoolConnectorConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "master", _coerce(self.master, KVCachePoolMasterConfig))
        object.__setattr__(self, "store", _coerce(self.store, KVCachePoolStoreConfig))
        object.__setattr__(self, "connector", _coerce(self.connector, KVCachePoolConnectorConfig))
        if not self.enabled:
            return
        if self.backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"cache_pool.backend={self.backend!r} not in {_ALLOWED_BACKENDS}")
        if self.connector.use_layerwise and self.backend != "memcache":
            raise ValueError(
                "cache_pool.connector.use_layerwise is only supported by the NPU memcache backend on a prefill node"
            )
        if self.backend != "mooncake":
            raise NotImplementedError(
                f"cache_pool.backend={self.backend!r} is not implemented; only 'mooncake' is supported"
            )
        if self.store.mode not in _ALLOWED_STORE_MODES:
            raise ValueError(f"cache_pool.store.mode={self.store.mode!r} not in {_ALLOWED_STORE_MODES}")
        if self.master.auto_start and self.store.config_path:
            raise ValueError("cache_pool.master.auto_start=True is incompatible with store.config_path")
        if not self.master.auto_start and not self.master.address and not self.store.config_path:
            raise ValueError("cache_pool.master.auto_start=False requires master.address or store.config_path")
        if self.master.enable_multi_tenants and (
            not self.master.tenant_quota_connector_type or not self.master.tenant_quota_connector_uri
        ):
            raise ValueError(
                "cache_pool.master.enable_multi_tenants=True requires tenant_quota_connector_type "
                "and tenant_quota_connector_uri"
            )
        if (
            self.kv_load_failure_policy is not None
            and self.kv_load_failure_policy not in _ALLOWED_KV_LOAD_FAILURE_POLICIES
        ):
            raise ValueError(
                f"cache_pool.kv_load_failure_policy={self.kv_load_failure_policy!r} "
                f"not in {_ALLOWED_KV_LOAD_FAILURE_POLICIES}"
            )
        if self.master.port is not None and (
            not isinstance(self.master.port, int)
            or isinstance(self.master.port, bool)
            or self.master.port < 0
        ):
            raise ValueError(
                f"cache_pool.master.port={self.master.port!r} must be None, 0 (auto), or a positive int"
            )
