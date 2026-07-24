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

"""CacheStore config."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ConfigError, _multiline_repr

_VALID_KV_CACHE_STORE_TYPES = {"list", "radix_tree"}


@dataclass(repr=False)
class CacheStoreConfig:
    """Config for CacheStore — passive storage layer for decoded metrics data."""

    kv_cache_store_type: str = "list"
    ttl: float = 30.0

    def __post_init__(self) -> None:
        if self.kv_cache_store_type not in _VALID_KV_CACHE_STORE_TYPES:
            raise ConfigError(
                f"kv_cache_store_type must be one of {_VALID_KV_CACHE_STORE_TYPES}, got '{self.kv_cache_store_type}'"
            )
        if self.ttl <= 0:
            raise ConfigError(f"ttl must be > 0, got {self.ttl}")

    __repr__ = _multiline_repr
