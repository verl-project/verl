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

"""PerRequestStore — bounded per-request state (``request_id → {key: value}``).

Generic per-request key/value storage, LRU-bounded and thread-safe. The store
owns only storage + locking; callers own the keys and their semantics. A full
row is LRU-evicted as a unit when ``request_id`` stops recurring.
``singleton()`` returns the shared instance; tests reset ``_instance``.
"""

from __future__ import annotations

import threading
from typing import Any

from cachetools import LRUCache

from ..logging import get_router_logger

logger = get_router_logger("per-request")

# Max request_ids retained; least-recently-used evicted past this.
DEFAULT_PER_REQUEST_MAX_SIZE = 10000


class PerRequestStore:
    """Singleton per-request state store — ``request_id → {key: value}``, LRU-bounded."""

    _instance: PerRequestStore | None = None

    def __init__(self, max_size: int = DEFAULT_PER_REQUEST_MAX_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = int(max_size)
        self._lock = threading.Lock()
        self._data: LRUCache[str, dict[str, Any]] = LRUCache(maxsize=self._max_size)
        logger.info(f"PerRequestStore created: max_size={self._max_size}")

    @classmethod
    def singleton(cls) -> PerRequestStore:
        """Return the shared singleton (tests construct a fresh instance / reset _instance)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def max_size(self) -> int:
        """Configured per-request table capacity."""
        return self._max_size

    def get(self, request_id: str, key: str, default: Any = None) -> Any:
        """Return the per-request value for ``key`` (``default`` if unset/evicted)."""
        with self._lock:
            row = self._data.get(request_id)
            return default if row is None else row.get(key, default)

    def set(self, request_id: str, key: str, value: Any) -> None:
        """Set ``request_id``'s ``key`` to ``value`` (creates the row if new)."""
        with self._lock:
            row = self._data.get(request_id, {})
            row[key] = value
            self._data[request_id] = row  # insert (new) or touch LRU recency (existing)

    def incr(self, request_id: str, key: str, delta: int | float = 1) -> int | float:
        """Add ``delta`` to ``request_id``'s numeric ``key``; return the new value."""
        with self._lock:
            row = self._data.get(request_id, {})
            value = row.get(key, 0) + delta
            row[key] = value
            self._data[request_id] = row  # insert (new) or touch LRU recency (existing)
            return value

    def delete(self, request_id: str, key: str) -> None:
        """Drop ``key`` from ``request_id``'s row (no-op if absent)."""
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return
            row.pop(key, None)
            if not row:
                del self._data[request_id]

    def delete_where(self, key: str, value: Any) -> None:
        """Drop ``key`` from every request whose value for it equals ``value``."""
        with self._lock:
            for request_id, row in [(rid, r) for rid, r in self._data.items() if r.get(key) == value]:
                row.pop(key, None)
                if not row:
                    del self._data[request_id]

    def count(self, key: str) -> int:
        """Number of requests that currently have ``key`` set."""
        with self._lock:
            return sum(1 for row in self._data.values() if key in row)

    def reset(self) -> None:
        """Clear all per-request state (test helper; not used on the hot path)."""
        with self._lock:
            self._data.clear()
