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

"""Per-replica metric store for reading/writing polled metric data."""

from __future__ import annotations

import threading
from typing import Any

from ..types import METRIC_SPECS


class PerReplicaStore:
    """Per-replica metric store: ``{node_id: {canonical_key: value}}``.

    - ``get(node_id, key)``  → single value; falls back to ``METRIC_SPECS[key]["default"]``;
                               raises ``KeyError`` if key is not a valid canonical key
    - ``get(node_id)``       → entire node dict (empty dict if unknown)
    - ``refresh(new_data)``  → batch update; only updates nodes present in ``new_data``,
                               existing nodes NOT in ``new_data`` are left untouched
    """

    _instance: PerReplicaStore | None = None

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> PerReplicaStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, node_id: str, key: str | None = None) -> Any | dict[str, Any]:
        """Read metrics.

        ``get(node_id, key)``  → single value, falls back to spec default.
            Raises ``KeyError`` if ``key`` is not a valid canonical key
            (i.e. not present in ``METRIC_SPECS``).
        ``get(node_id)``       → entire node dict
        """
        if key is None:
            return dict(self._data.get(node_id, {}))

        if key not in METRIC_SPECS:
            raise KeyError(f"Unknown metric key '{key}'. Valid keys: {sorted(METRIC_SPECS.keys())}")
        node = self._data.get(node_id, {})
        if key in node:
            return node[key]
        return METRIC_SPECS[key]["default"]

    def incr(self, node_id: str, key: str, delta: int | float = 1) -> None:
        """Apply a numeric delta to one key for one node (inflight ±1).

        Unlike ``refresh`` (batch merge overwrite), this is an incremental
        write: it reads the current value (falling back to the spec default)
        and stores ``current + delta``. Keeps the writer (decoder) stateless
        — it only emits the +/-1 delta; the store owns the running counter.

        Args:
            node_id: Target node.
            key: Canonical metric key (must be in ``METRIC_SPECS``).
            delta: Signed delta to add (default +1).

        Raises:
            KeyError: If ``key`` is not a valid canonical key.
        """
        if key not in METRIC_SPECS:
            raise KeyError(f"Unknown metric key '{key}'. Valid keys: {sorted(METRIC_SPECS.keys())}")
        self._apply(node_id, {key: delta})

    def incr_many(self, node_id: str, deltas: dict[str, int | float]) -> None:
        """Apply multiple signed deltas to one node under a single lock.

        Same incremental semantics as :meth:`incr` (reads current, adds delta,
        falling back to spec defaults), but takes the lock once for the whole
        batch — the ``on_acquire`` decoder emits several deltas per dispatch
        (INFLIGHT / DISPATCHED / PROMPT_LEN_SUM), so batching avoids N lock
        cycles per request on the hot path.

        Args:
            node_id: Target node.
            deltas: ``{canonical_key: signed_delta}`` (all keys in ``METRIC_SPECS``).

        Raises:
            KeyError: If any key is not a valid canonical key.
        """
        if not deltas:
            return
        bad = [k for k in deltas if k not in METRIC_SPECS]
        if bad:
            raise KeyError(f"Unknown metric keys: {sorted(bad)}. Valid keys: {sorted(METRIC_SPECS.keys())}")
        self._apply(node_id, deltas)

    def _apply(self, node_id: str, deltas: dict[str, int | float]) -> None:
        """Apply signed deltas to one node under one lock (caller validates keys).

        Each key falls back to its ``METRIC_SPECS`` default before adding the
        delta, so the writer stays stateless — it only emits the +/-delta.
        Shared by :meth:`incr` (single key) and :meth:`incr_many` (batch).
        """
        with self._lock:
            node = self._data.setdefault(node_id, {})
            for key, delta in deltas.items():
                node[key] = node.get(key, METRIC_SPECS[key]["default"]) + delta

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
