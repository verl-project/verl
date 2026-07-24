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

"""VLLMKVDecoder — vLLM KV-cache event decoder.

Decodes msgpack payloads from ZMQ and returns structured update commands.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

import msgpack

from ....collectors.decoder import Decoder, KVCacheUpdate
from ....collectors.decoder.vllm.kv_event import KVCacheEvent
from ....logging import get_router_logger
from ....types import Layer
from ....utils.hash import compute_hash

logger = get_router_logger("vllm-kv")


class VLLMKVDecoder(Decoder):
    """vLLM KV-cache decoder — msgpack payload → KVCacheUpdate.

    Each event type has a dedicated handler that folds its blocks into a single
    ``KVCacheUpdate`` accumulator; ``decode`` is pure dispatch.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  Used for chained
            hash computation.
        _block_size: Learned block size from first event.
    """

    # vLLM BlockStored/BlockRemoved ``medium`` → canonical layer.
    # Unknown / None (older vLLM) → GPU (see ``_medium_to_layer``).
    _MEDIUM_TO_LAYER: dict[str, Layer] = {"GPU": Layer.GPU, "cpu": Layer.CPU}

    def __init__(self) -> None:
        self.remote_to_local_block_hash: dict[str, str] = {}
        self._block_size: int | None = None

    def decode(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | None:
        """Decode msgpack payload and return structured update command.

        Handles both single event (real-time) and multiple events (replay):
          - Single: [timestamp, [[tag, fields...], ...]]
          - Multiple: [[timestamp, [...]], [timestamp, [...]]]

        Args:
            raw_data: ZMQ payload bytes (msgpack-encoded).
            node_id: The endpoint that sent this payload.

        Returns:
            KVCacheUpdate with operations to apply, or None if decode failed.
        """
        if isinstance(raw_data, str):
            logger.debug("VLLMKVDecoder received string data, expected bytes — skipping")
            return None

        try:
            raw = msgpack.unpackb(raw_data, raw=False)

            if not isinstance(raw, list) or len(raw) == 0:
                logger.warning(f"Unexpected msgpack format from node {node_id} (type={type(raw).__name__})")
                return None
            event_payloads = raw if isinstance(raw[0], list) else [raw]

            update = KVCacheUpdate(node_id=node_id)
            for payload in event_payloads:
                events = KVCacheEvent.from_raw(payload, default_node_id=node_id)
                for event in events:
                    if event.event_type == "stored":
                        self._on_block_stored(event, update)
                    elif event.event_type == "removed":
                        self._on_block_removed(event, update)
                    elif event.event_type == "clear":
                        self._on_all_blocks_cleared(event, update)
                    else:
                        raise ValueError(f"Unknow event.event_type {event.event_type}.")
            return update

        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            preview = bytes(raw_data[:32]).hex() if isinstance(raw_data, bytes | bytearray) else str(raw_data)[:64]
            logger.warning(
                f"Failed to decode msgpack payload from node {node_id}: {exc!r} (len={len(raw_data)}, head={preview})"
            )
            return None

    @classmethod
    def _medium_to_layer(cls, medium: str | None) -> Layer:
        """Map a vLLM ``medium`` to a canonical layer; None/unknown → GPU."""
        return cls._MEDIUM_TO_LAYER.get(medium, Layer.GPU)

    # ── Event handlers ──────────────────────────────────────────────────

    def _on_block_stored(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle BlockStored: learn block_size, compute local hashes, fold into update."""
        if event.token_ids is None:
            logger.debug("Stored event has no token_ids — skipping")
            return

        if self._block_size is None and event.block_size is not None:
            self._block_size = event.block_size
            update.set_block_size(event.block_size)

        seed = 0
        local_parent_hash = seed
        if event.parent_block_hash is not None:
            local_parent_str = self.remote_to_local_block_hash.get(event.parent_block_hash)
            if local_parent_str is not None:
                local_parent_hash = int(local_parent_str)

        local_hashes: list[str] = []
        for i, block_bytes in enumerate(event.token_ids):
            if i >= len(event.block_hashes):
                break
            local_hash_int = compute_hash(
                local_parent_hash,
                block_bytes,
                seed=seed,
            )
            local_hash_str = str(local_hash_int)
            bh = event.block_hashes[i]
            self.remote_to_local_block_hash[bh] = local_hash_str
            local_hashes.append(local_hash_str)
            local_parent_hash = local_hash_int  # chain

        update.add(self._medium_to_layer(event.medium), local_hashes)

    def _on_block_removed(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle BlockRemoved: convert remote hashes to local, fold into update."""
        local_hashes = [
            self.remote_to_local_block_hash[bh] for bh in event.block_hashes if bh in self.remote_to_local_block_hash
        ]
        for bh in event.block_hashes:
            self.remote_to_local_block_hash.pop(bh, None)

        update.remove(self._medium_to_layer(event.medium), local_hashes)

    def _on_all_blocks_cleared(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle AllBlocksCleared: mark the replica for a full clear."""
        update.clear()
