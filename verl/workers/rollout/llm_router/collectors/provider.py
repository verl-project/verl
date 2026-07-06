"""RouteDataProvider — unified query entry point for routing decisions.

Strategy layers call ``RouteDataProvider`` methods to get metrics data.
It delegates to store instances (``MetricsStore`` for polling metrics,
``KVCacheStore`` for GPU prefix cache data).

Collectors are created from ``BUILTIN_REGISTRY`` as ``Collector``
instances combining Transport + Decoder.  Stores are singletons —
deduplication happens automatically at the store level.

All query computations are delegated to the respective store classes.
"""

from __future__ import annotations

from typing import Any

from verl.workers.rollout.llm_router.collectors.registry import BUILTIN_REGISTRY
from verl.workers.rollout.llm_router.logging import get_router_logger
from verl.workers.rollout.llm_router.store.kv_cache_store import KVCacheStore
from verl.workers.rollout.llm_router.store.metrics_store import MetricsStore

logger = get_router_logger("provider")


class RouteDataProvider:
    """Unified query entry point — strategies use this to access all metrics.

    ``RouteDataProvider`` creates collectors via the registry, which
    combines Transport + Decoder into ``Collector`` instances.  Store
    deduplication is handled by the store classes themselves (singleton).

    All query computations are delegated to the respective store classes.

    Args:
        collectors_config: ``CollectorConfig`` — provides common settings
            and endpoint addresses.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
        server_addresses: ``{replica_id: ip:port}`` for HTTP transport.
        kv_event_endpoints: ``{replica_id: [sub_addr, replay_addr]}`` for ZMQ transport.
    """

    def __init__(
        self,
        collectors_config,
        collection_names,
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ) -> None:
        self._collectors: list[Any] = []
        # Snapshot the collection names for lifecycle logging.
        self._collection_names = list(collection_names)

        http_polling = collectors_config.http_polling
        long_conn = collectors_config.long_connection

        for name in collection_names:
            if name == "vllm_metrics":
                collector = BUILTIN_REGISTRY.get_collector(
                    name,
                    endpoints=server_addresses or {},
                    interval=http_polling["polling_interval"],
                    http_timeout=http_polling["http_timeout"],
                )
            elif name == "vllm_zmq":
                collector = BUILTIN_REGISTRY.get_collector(
                    name,
                    endpoints=kv_event_endpoints or {},
                    base_retry_delay=long_conn["base_retry_delay"],
                    max_retry_delay=long_conn["max_retry_delay"],
                    max_retry_attempts=long_conn["max_retry_attempts"],
                    retry_backoff_factor=long_conn["retry_backoff_factor"],
                )
            else:
                collector = BUILTIN_REGISTRY.get_collector(name)
            self._collectors.append(collector)

        logger.info(
            "RouteDataProvider created: collection_names=%s, collectors=[%s]",
            collection_names,
            ", ".join(type(c).__name__ for c in self._collectors) or "<none>",
        )

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        logger.info("RouteDataProvider starting %d collector(s)", len(self._collectors))
        for name, collector in zip(self._collection_names, self._collectors, strict=False):
            collector.start()
            logger.info(
                "collector started: name=%s type=%s",
                name,
                type(collector).__name__,
            )

    def stop(self) -> None:
        """Stop all collectors and await their cleanup."""
        logger.info("RouteDataProvider stopping %d collector(s)", len(self._collectors))
        for collector in self._collectors:
            collector.stop()

    # ── Query proxies (delegate to the singleton stores) ────────────────
    #
    # The strategy layer calls these on the provider; they forward to the
    # appropriate singleton store so callers don't need to know which store
    # holds which data.  Computation lives in the store classes — the
    # provider is only a routing shim.

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Per-replica GPU prefix-cache hit percent (0–100).

        Delegates to ``KVCacheStore.default().get_gpu_prefix_hit_rate``.
        """
        return KVCacheStore.default().get_gpu_prefix_hit_rate(prompt_ids)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full polling metrics snapshot.

        Delegates to ``MetricsStore.default().get(node_id)``.
        """
        return MetricsStore.default().get(node_id)

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a single polling metric by canonical key.

        Delegates to ``MetricsStore.default().get(node_id, key)``.
        """
        return MetricsStore.default().get(node_id, key)

    def get_tier_prefix_hit_rate(
        self,
        node_id: str,
        prompt_ids: list[int],
        tier: str,
    ) -> float | None:
        """Tier-level (cpu/ssd) prefix-cache hit rate, or ``None`` if unavailable.

        Mooncake tier collection is not yet implemented; the store returns
        ``0.0`` in that case, which we surface as ``None`` so the slow-path
        caller can distinguish "no data" from a genuine 0% hit and warn.
        """
        return KVCacheStore.default().get_tier_prefix_hit_rate(node_id, prompt_ids, tier) or None
