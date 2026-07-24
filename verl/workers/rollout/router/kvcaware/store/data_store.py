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

"""StoreProvider — unified data access layer for all stores."""

from __future__ import annotations

from typing import Any

from ..types import Layer, MetricKey
from .kv_cache_store import KVCacheStore
from .per_replica_store import PerReplicaStore
from .per_request_store import PerRequestStore

_STICKY_KEY = "sticky_replica"


class DataStore:
    """Unified data access layer — single entry point for all store operations.

    Wraps the singleton stores and exposes a unified interface for all
    reads and writes.  Stateless — instantiate once and reuse.

    Usage:
        provider = StoreProvider()
        # Write metrics
        provider.refresh_metrics({'node1': {'kv_cache_usage_perc': 45.0}})
        # Read metrics
        value = provider.get_metric('node1', 'kv_cache_usage_perc')
        # Write KV cache blocks
        provider.add_kv_blocks('node1', ['hash1', 'hash2'])
    """

    def __init__(self) -> None:
        self._metrics = PerReplicaStore.singleton()
        self._kv = KVCacheStore.singleton()
        self._per_request = PerRequestStore.singleton()

    # ── PerReplicaStore operations ─────────────────────────────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a single metric by canonical key.

        Args:
            node_id: Target node.
            key: Canonical metric key (e.g., ``MetricKey.KV_CACHE_USAGE_PERC``).

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if absent.

        Raises:
            KeyError: If key is not a valid canonical key.
        """
        return self._metrics.get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self._metrics.get(node_id)

    def get_metric_node_ids(self) -> list[str]:
        """Return all node IDs that have metrics in the store."""
        return self._metrics.all_ids()

    def refresh_metrics(self, new_data: dict[str, dict[str, Any]]) -> None:
        """Batch refresh metrics from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.

        Args:
            new_data: Dict of {node_id: {canonical_key: value}}.
        """
        self._metrics.refresh(new_data)

    # ── KVCacheStore operations ─────────────────────────────────────────

    def get_block_size(self) -> int | None:
        """Get learned block size.

        Returns:
            Block size in tokens, or None if not yet learned.
        """
        return self._kv.block_size

    def set_block_size(self, size: int) -> None:
        """Set block size (learned from first BlockStored event).

        Args:
            size: Block size in tokens.
        """
        if self._kv.block_size is None:
            self._kv.block_size = size

    def add_kv_blocks(self, node_id: str, block_hashes: list[str], layer: Layer = Layer.GPU) -> None:
        """Add KV cache blocks to a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to add.
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).
        """
        self._kv.add_blocks(node_id, block_hashes, layer=layer)

    def remove_kv_blocks(self, node_id: str, block_hashes: list[str], layer: Layer = Layer.GPU) -> None:
        """Remove KV cache blocks from a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to remove.
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).
        """
        self._kv.remove_blocks(node_id, block_hashes, layer=layer)

    def clear_kv_node(self, node_id: str) -> None:
        """Clear all KV cache blocks for a node.

        Args:
            node_id: Target node.
        """
        self._kv.clear_replica(node_id)

    def get_kv_block_count(self) -> int:
        """Return the number of unique block hashes currently cached."""
        return len(self._kv.replicas_by_block)

    def kv_node_has_blocks(self, node_id: str) -> bool:
        """Return True if node_id appears in at least one cached block."""
        return any(node_id in replicas for replicas in self._kv.replicas_by_block.values())

    def has_kv_block(self, block_hash: str) -> bool:
        """Return True if block_hash is present in the cache index."""
        return block_hash in self._kv.replicas_by_block

    # ── KV cache prefix hit rate queries ────────────────────────────────

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        prompt_ids: list[int],
        layer: Layer = Layer.GPU,
    ) -> float:
        """Query prefix-cache hit rate for a node at a given layer.

        Args:
            node_id: Target node.
            prompt_ids: Current request's prompt token IDs.
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).

        Returns:
            Hit rate 0.0–1.0.
        """
        return self._kv.get_layer_prefix_hit_rate(node_id, prompt_ids, layer)

    # ── KV-cache load (load signal) ─────────────────────────────────────

    def kv_cache_load(self, node_id: str) -> float:
        """KV-cache load = ``retained_blocks / num_gpu_blocks`` (∈ [0,1]).

        Retained blocks (free-pool blocks hash-marked for reuse) over the GPU
        block pool — rises as distinct prefixes accumulate and signals impending
        LRU eviction. Returns 0.0 when retained blocks or ``num_gpu_blocks`` is
        unavailable (e.g. mc-off groups emit no kv-events).
        """
        retained = self._kv.per_replica_block_counts().get(node_id, 0)
        if retained <= 0:
            return 0.0
        total = self._metrics.get(node_id, MetricKey.NUM_GPU_BLOCKS)
        if not total or total <= 0:
            return 0.0
        return min(1.0, retained / float(total))

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct prefix blocks it retains}``.

        Thin pass-through to ``KVCacheStore``; used by the periodic kv-events
        tally log to surface retained-block counts per replica.
        """
        return self._kv.per_replica_block_counts()

    # ── PerReplicaStore incremental write ──────────────────────────────────

    def incr_metric(self, node_id: str, key: str, delta: int | float = 1) -> None:
        """Apply a signed delta to one metric for one node (inflight ±1).

        Routes to ``PerReplicaStore.incr`` (not ``refresh``) so a stateless delta
        emitter (``InflightDecoder``) can move a running counter without
        tracking the absolute value itself.
        """
        self._metrics.incr(node_id, key, delta)

    def incr_metrics(self, node_id: str, deltas: dict[str, int | float]) -> None:
        """Apply multiple signed deltas to one node under a single lock.

        Batched variant of :meth:`incr_metric` for the ``on_acquire`` decoder,
        which emits several deltas per dispatch (INFLIGHT / DISPATCHED /
        PROMPT_LEN_SUM) — batching avoids one lock cycle per key on the hot path.
        """
        self._metrics.incr_many(node_id, deltas)

    # ── Sticky bindings (a per-request value stored under _STICKY_KEY) ───

    def get_sticky_binding(self, request_id: str) -> str | None:
        """Return the bound replica_id for ``request_id`` (None if cold/evicted)."""
        return self._per_request.get(request_id, _STICKY_KEY)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        """Bind / refresh ``request_id → replica_id`` (driven by ``on_acquire``)."""
        self._per_request.set(request_id, _STICKY_KEY, replica_id)

    def invalidate_sticky_binding(self, request_id: str) -> None:
        """Drop one request_id's sticky binding."""
        self._per_request.delete(request_id, _STICKY_KEY)

    def invalidate_sticky_replica(self, replica_id: str) -> None:
        """Drop every sticky binding pointing at a removed replica."""
        self._per_request.delete_where(_STICKY_KEY, replica_id)

    def sticky_status(self) -> dict:
        """Return a debugging snapshot of the sticky bindings."""
        return {"max_size": self._per_request.max_size, "size": self._per_request.count(_STICKY_KEY)}

    # ── PerRequestStore operations ─────────────────────────────────────

    def incr_per_request(self, request_id: str, key: str, delta: int | float = 1):
        """Apply a signed delta to one per-request value; return the new value.

        Routes to ``PerRequestStore.incr`` — the per-request store is a generic
        ``request_id → {key: value}`` scratch space (not tied to any one metric),
        so the caller owns ``key`` and its semantics. Mirrors ``incr_metric``.
        """
        return self._per_request.incr(request_id, key, delta)

    def get_per_request(self, request_id: str, key: str, default: Any = None):
        """Return one per-request value (``default`` if unset/evicted)."""
        return self._per_request.get(request_id, key, default)
