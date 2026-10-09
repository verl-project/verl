# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Ray concrete ObjectStore using ObjectRef values."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

import ray

from verl.runtime.object_store import ObjectStore


class RayObjectStore(ObjectStore[ray.ObjectRef]):
    """Store values in Ray's distributed object store.

    ``delete()`` is intentionally a no-op: object lifetime follows Ray's
    distributed reference tracking. Aliases remain gettable after delete.
    """

    def __init__(self) -> None:
        self._put_condition = threading.Condition()
        self._idle_put_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active_put_batches = 0

    @classmethod
    def start(cls) -> None:
        """Start Ray storage for the current process."""
        from verl.runtime.object_store import _start_object_store

        _start_object_store(cls())

    def put(self, key: str, value: Any, /) -> ray.ObjectRef:
        """Store one value in Ray.

        Args:
            key: Stable logical key. Ray does not require it for placement.
            value: Value submitted to Ray's object store.

        Returns:
            The Ray object reference.
        """
        if not isinstance(key, str) or not key:
            raise ValueError(f"key must be a non-empty str, got {key!r}")
        return ray.put(value)

    def put_many(self, entries: list[tuple[str, Any]], /) -> list[ray.ObjectRef]:
        """Store values in parallel while preserving input order."""
        if not entries:
            return []
        self._validate_put_many_entries(entries)

        # Each overlapping (including reentrant) batch owns its own executor.
        # Only one idle executor is retained between calls.
        with self._put_condition:
            executor = self._idle_put_executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
            self._idle_put_executor = None
            self._active_put_batches += 1
        futures = []
        submission_complete = False
        try:
            for key, value in entries:
                futures.append(executor.submit(self.put, key, value))
            submission_complete = True
            return [future.result() for future in futures]
        finally:
            # Match the old executor context manager: even partial submission
            # or an early rank failure waits for every submitted put to finish.
            concurrent.futures.wait(futures)
            with self._put_condition:
                # submit can enqueue work before thread startup raises, without
                # returning that work's Future. Only shutdown drains that queue.
                retain = submission_complete and self._idle_put_executor is None
                if retain:
                    self._idle_put_executor = executor
            try:
                if not retain:
                    executor.shutdown(wait=True)
            finally:
                with self._put_condition:
                    self._active_put_batches -= 1
                    self._put_condition.notify_all()

    def close(self) -> None:
        """Join batches and release cached threads after the owner stops admission.

        This releases only local executor resources, not Ray-owned objects.
        """
        with self._put_condition:
            self._put_condition.wait_for(lambda: self._active_put_batches == 0)
            executor, self._idle_put_executor = self._idle_put_executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def get(self, reference: ray.ObjectRef, /) -> Any:
        """Resolve one Ray object reference.

        Args:
            reference: Ray object reference to resolve.

        Returns:
            The stored value.
        """
        return ray.get(reference)

    def get_many(self, references: list[ray.ObjectRef], /) -> list[Any]:
        """Resolve multiple Ray references in one native batch."""
        if not references:
            return []
        return ray.get(references)

    def delete(self, reference: ray.ObjectRef, /) -> None:
        """Accept that the caller will stop using one Ray reference.

        Args:
            reference: Ray object reference the caller will stop using.
        """
        return None

    def delete_many(self, references: list[ray.ObjectRef], /) -> None:
        """Accept batched release while Ray reference tracking owns lifetime."""
        return None


__all__ = ["RayObjectStore"]
