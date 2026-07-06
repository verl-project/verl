"""CacheStore config."""

from __future__ import annotations

from dataclasses import dataclass

from verl.workers.rollout.llm_router.config.base import ConfigError, _multiline_repr

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
