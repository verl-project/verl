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

"""Decoder abstract base and its update return types.

A ``Decoder`` turns a raw transport payload (bytes/str) into a structured
update that the ``Collector`` applies to the store.  The update types
(``KVCacheUpdate``, ``MetricsUpdate``) live here, next to the base, because
they are the decoder's output contract — the layer that both produces and
consumes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...types import Layer


@dataclass
class KVCacheUpdate:
    """Mutable accumulator for KVCacheStore updates, built by the decoder.

    The decoder dispatches each event to a handler that calls ``add``/``remove``
    to fold blocks in; the collector then reads ``add_blocks``/``remove_blocks``
    to write the store.

    Attributes:
        node_id: Target endpoint identifier.
        add_blocks: layer → block hashes accumulated to add.
        remove_blocks: layer → block hashes accumulated to remove.
        clear_all: If True, clear all blocks for this node.
        block_size: Block size learned from the first BlockStored (None until then).
    """

    node_id: str
    add_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    remove_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    clear_all: bool = False
    block_size: int | None = None

    def add(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold stored blocks into ``add_blocks`` under ``layer``."""
        self.add_blocks.setdefault(layer, []).extend(block_hashes)

    def remove(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold removed blocks into ``remove_blocks`` under ``layer``."""
        self.remove_blocks.setdefault(layer, []).extend(block_hashes)

    def clear(self) -> None:
        """Mark the replica for a full block clear."""
        self.clear_all = True

    def set_block_size(self, size: int) -> None:
        """Set the learned block size (first BlockStored wins; set by the decoder)."""
        self.block_size = size


@dataclass
class MetricsUpdate:
    """Structured update command for PerReplicaStore.

    Attributes:
        node_id: Target endpoint identifier.
        metrics: Dict of canonical_key → value.
        is_delta: When ``False`` (default, ``VLLMMetricsDecoder``) the values
            are absolute gauges applied via ``refresh`` (merge overwrite). When
            ``True`` (``InflightDecoder``) the values are signed deltas applied
            via ``incr`` — keeps the decoder stateless (it emits only ±1; the
            store owns the running counter).
        request_id: Optional routing request id carried with the update
            (``None`` when the update isn't request-scoped).
    """

    node_id: str
    metrics: dict[str, Any]
    is_delta: bool = False
    request_id: str | None = None


@dataclass
class StickyUpdate:
    """Structured update command for the per-request store (sticky binding).

    Emitted by ``StickyDecoder`` from the Balancer's ``on_acquire`` /
    ``on_servers_removed`` callbacks (packed into a ``StatisticEvent`` by
    ``CallbackTransport``). Distinct from ``KVCacheUpdate`` (per-replica
    block) and ``MetricsUpdate`` (per-replica gauge) because sticky is a
    per-request dimension with LRU recency semantics — refresh/merge can't
    express ``put`` / ``invalidate_replica``.

    Attributes:
        action: ``"put"`` / ``"invalidate"`` / ``"invalidate_replica"``.
        request_id: Bound request id (for ``put`` / ``invalidate``).
        replica_id: Bound replica id (for ``put``); ignored on invalidate_replica.
        replica_ids: Replicas to clear (for ``invalidate_replica``).
    """

    action: str
    request_id: str | None = None
    replica_id: str | None = None
    replica_ids: tuple[str, ...] = ()


class Decoder(ABC):
    """Abstract base for data decoders.

    Subclasses implement ``decode()`` with their backend-specific parsing logic,
    returning a ``KVCacheUpdate`` / ``MetricsUpdate`` / ``StickyUpdate``
    (or ``None`` on failure).
    """

    @abstractmethod
    def decode(self, raw_data: bytes | str | Any, node_id: str) -> KVCacheUpdate | MetricsUpdate | StickyUpdate | None:
        """Decode raw data and return a structured update.

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) / ``str`` (from HTTP
                response text) for network transports, or a ``StatisticEvent``
                for the callback transport (sticky / inflight decoders).
            node_id: Source endpoint identifier (empty string for the callback
                transport — the event carries its own request/replica ids).

        Returns:
            A ``KVCacheUpdate`` / ``MetricsUpdate`` / ``StickyUpdate``, or
            ``None`` if decode fails or the payload type is not handled.
        """
