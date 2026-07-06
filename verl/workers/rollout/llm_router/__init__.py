"""KV-cache-aware LLM Router configuration and routing primitives."""

from verl.workers.rollout.llm_router.balancer import KVCAwareBalancer
from verl.workers.rollout.llm_router.collectors import MetricKey, RouteDataProvider
from verl.workers.rollout.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from verl.workers.rollout.llm_router.strategies import (
    KVCacheAwareStrategy,
    ReplicaInfo,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)

__all__ = [
    "KVCAwareBalancer",
    "CacheStoreConfig",
    "CollectorConfig",
    "ConfigError",
    "KVCAwareConfig",
    "KVCAwareStrategyConfig",
    "StrategyConfig",
    "KVCacheAwareStrategy",
    "RoutingStrategy",
    "StrategyError",
    "StrategyRegistry",
    "route",
    "MetricKey",
    "ReplicaInfo",
    "RouteDataProvider",
]
