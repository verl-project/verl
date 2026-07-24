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

"""KVCacheStore — backend-agnostic data carrier for KV cache mapping tables."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from ..types import Layer
from ..utils.hash import get_prefix_hashes


class KVCacheStore:
    """Mutable data carrier for KV cache mapping tables.

    Attributes:
        block_size: Learned block size (None until first BlockStored event).
        replicas_by_block: local prefix hash → set of replica_ids that cache
            it on GPU.  CPU/SSD blocks are counted but not indexed here.
    """

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self.replicas_by_block: dict[str, set[str]] = {}
        # Per-layer per-replica block counts, maintained alongside replicas_by_block.
        self._replica_layer_counts: dict[Layer, dict[str, int]] = {
            Layer.GPU: {},
            Layer.CPU: {},
            Layer.SSD: {},
        }
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear all blocks for a replica from the reverse index.

        O(n) in the number of unique blocks, but replica count is typically small (< 100).
        """
        with self._lock:
            stale_hashes: list[str] = []
            for bh, replicas in self.replicas_by_block.items():
                if replica_id not in replicas:
                    continue
                replicas.discard(replica_id)
                if not replicas:
                    stale_hashes.append(bh)
            for bh in stale_hashes:
                del self.replicas_by_block[bh]
            for layer_counts in self._replica_layer_counts.values():
                layer_counts.pop(replica_id, None)

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Add blocks to a replica at a layer, updating the reverse index.

        Only GPU blocks are indexed in ``replicas_by_block`` (they drive
        prefix-hit routing); CPU/SSD blocks are counted only.
        """
        with self._lock:
            layer_counts = self._replica_layer_counts.setdefault(layer, {})
            for bh in block_hashes:
                layer_counts[replica_id] = layer_counts.get(replica_id, 0) + 1
                if layer != Layer.GPU:
                    continue
                if bh not in self.replicas_by_block:
                    self.replicas_by_block[bh] = set()
                self.replicas_by_block[bh].add(replica_id)

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Remove blocks from a replica at a layer, updating the reverse index."""
        with self._lock:
            layer_counts = self._replica_layer_counts.setdefault(layer, {})
            for bh in block_hashes:
                layer_counts[replica_id] = layer_counts.get(replica_id, 0) - 1
                if layer != Layer.GPU:
                    continue
                replicas = self.replicas_by_block.get(bh)
                if replicas is not None and replica_id in replicas:
                    replicas.discard(replica_id)
                    if not replicas:
                        del self.replicas_by_block[bh]

    # ── Retained-cache size ─────────────────────────────────────────────

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct GPU prefix blocks it retains}``.

        GPU-only count — feeds the retained-load formula. Maintained incrementally,
        O(replicas). Divide by the per-replica block pool size for occupancy.
        """
        with self._lock:
            return dict(self._replica_layer_counts.get(Layer.GPU, {}))

    # ── Prefix hit rate queries ─────────────────────────────────────────

    def get_layer_prefix_hit_rate(self, node_id: str, prompt_ids: list[int], layer: Layer = Layer.GPU) -> float:
        """Prefix-cache hit rate for a node at a layer, ∈ [0.0, 1.0].

        GPU: walk the local reverse index (``replicas_by_block``) along the
        prompt's prefix-hash chain until a hash isn't cached on this node.
        CPU/SSD: placeholder 0.0 (mooncake /batch_query not wired yet).
        """
        with self._lock:
            if layer != Layer.GPU or self.block_size is None:
                return 0.0
            prefix_hashes = get_prefix_hashes(prompt_ids, self.block_size)
            if not prefix_hashes:
                return 0.0
            hash_strs = [str(h) for h in prefix_hashes]
            matched = 0
            for i, hs in enumerate(hash_strs):
                cached = self.replicas_by_block.get(hs)
                if cached is None or node_id not in cached:
                    break  # chain break — this node doesn't cache this hash
                matched = i + 1
            return matched / len(hash_strs)
