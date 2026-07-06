"""Unified metrics store for reading/writing polling metric data."""

from __future__ import annotations

import threading
from typing import Any

from verl.workers.rollout.llm_router.metric_spec import METRIC_SPECS


class MetricsStore:
    """Unified metrics store: ``{node_id: {canonical_key: value}}``.

    Thread-safe — a ``threading.Lock`` protects all reads and writes,
    because the store can be written by collector asyncio tasks (on the
    event-loop thread) and read by the balancer (on a Ray actor thread)
    concurrently.

    Singleton — use ``MetricsStore.default()`` to get the shared instance.
    ``store_cls()`` (called by collectors) also returns the singleton via
    the class-level ``__call__`` override.

    - ``get(node_id, key)``  → single value; falls back to ``METRIC_SPECS[key]["default"]``;
                               raises ``KeyError`` if key is not a valid canonical key
    - ``get(node_id)``       → entire node dict (empty dict if unknown)
    - ``refresh(new_data)``  → batch update; only updates nodes present in ``new_data``,
                               existing nodes NOT in ``new_data`` are left untouched
    """

    _default: MetricsStore | None = None

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def default(cls) -> MetricsStore:
        """Return the shared singleton instance."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def _get(self, node_id: str, key: str | None = None) -> Any | dict[str, Any]:
        """Read metrics.

        ``get(node_id, key)``  → single value, falls back to spec default.
            Raises ``KeyError`` if ``key`` is not a valid canonical key
            (i.e. not present in ``METRIC_SPECS``).
        ``get(node_id)``       → entire node dict
        """
        if key is not None:
            if key not in METRIC_SPECS:
                raise KeyError(f"Unknown metric key '{key}'. Valid keys: {sorted(METRIC_SPECS.keys())}")
            with self._lock:
                node = self._data.get(node_id, {})
                if key in node:
                    return node[key]
                return METRIC_SPECS[key]["default"]
        with self._lock:
            return dict(self._data.get(node_id, {}))

    # Public aliases — ``get`` / ``get_metric`` / ``get_metrics`` all delegate
    # to ``_get`` so callers can use whichever naming fits their context.
    def get(self, node_id: str, key: str | None = None) -> Any | dict[str, Any]:
        """Read metrics (public alias for ``_get``)."""
        return self._get(node_id, key)

    def refresh(self, new_data: dict[str, dict[str, Any]]) -> None:
        """Batch refresh from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.
        """
        with self._lock:
            for node_id, metrics in new_data.items():
                existing = self._data.get(node_id, {})
                merged = dict(existing)
                merged.update(metrics)
                self._data[node_id] = merged

    def all_ids(self) -> list[str]:
        """Return all node IDs currently in the store."""
        with self._lock:
            return list(self._data.keys())

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a polling metric by canonical key.

        Delegates to ``MetricsStore.get(node_id, key)``.

        Args:
            node_id: Target node.
            key: ``MetricKey`` constant, e.g. ``MetricKey.KV_CACHE_USAGE_PERC``.

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if
            node or metric is absent.
        """
        return self._get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full polling metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node
            is absent in the store.
        """
        return self._get(node_id)
