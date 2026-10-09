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

"""TorchStore ObjectStore with self-addressing (store, key) references.

HostStrategy selects the producer's local volume; references route consumers
back to the producer's store. DTensor shards retain their native slice metadata.
TorchStoreBackend owns placement and lifecycle independently of references.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import os
import socket
import threading
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from verl.runtime.object_store import ObjectStore
from verl.single_controller.base.errors import ExceptionGroup

if TYPE_CHECKING:
    from verl.single_controller.monarch.object_store.local_cache import LocalCache
    from verl.single_controller.monarch.object_store.ref_count import RefCountActor


class _TorchStoreLoop:
    """Run TorchStore's async client behind the synchronous ObjectStore seam."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="torchstore-client", daemon=True)
        self._thread.start()

    def submit(self, awaitable, *, timeout_s: float | None):
        async def wait():
            if timeout_s is None:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_s)

        return asyncio.run_coroutine_threadsafe(wait(), self._loop)

    def run(self, awaitable, *, timeout_s: float):
        future = self.submit(awaitable, timeout_s=timeout_s)
        try:
            return future.result(timeout=timeout_s)
        except BaseException:
            future.cancel()
            raise

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


_CLIENT_LOOP = _TorchStoreLoop()
atexit.register(_CLIENT_LOOP.close)


@dataclass(frozen=True, slots=True)
class TorchStoreReference:
    """Address of a value in the named TorchStore selected by its producer."""

    store_name: str
    key: str
    is_tensor: bool = field(default=False, compare=False)
    shared_reader: str | None = None
    shared_owner: RefCountActor | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TensorRowRange:
    """A contiguous first-axis selection in a stored tensor's global shape."""

    shape: tuple[int, ...]
    start: int
    stop: int
    segments: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in self.shape):
            raise ValueError(f"invalid tensor shape: {self.shape!r}")
        if any(not isinstance(n, int) or isinstance(n, bool) for n in (self.start, self.stop)):
            raise ValueError(f"row bounds must be integers: {self.start!r}, {self.stop!r}")
        if not 0 <= self.start <= self.stop <= self.shape[0]:
            raise ValueError(f"invalid row range [{self.start}, {self.stop}) for {self.shape!r}")
        previous = self.start
        for start, stop in self.segments:
            if not self.start <= start < stop <= self.stop or start < previous:
                raise ValueError(f"invalid row segments: {self.segments!r}")
            previous = stop


class TorchStoreObjectStore(ObjectStore[TorchStoreReference]):
    """Synchronous client for native storage operations and process-local cleanup."""

    def __init__(
        self,
        store_name: str,
        *,
        timeout_s: float = 300.0,
        local_cache_bytes: int = 0,
        ref_count_actors: RefCountActor | None = None,
    ) -> None:
        if not isinstance(store_name, str) or not store_name:
            raise ValueError(f"store_name must be a non-empty str, got {store_name!r}")
        if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        from verl.single_controller.monarch.object_store.local_cache import LocalCache

        self._local_cache = LocalCache(local_cache_bytes)
        self._store_name = store_name
        self._timeout_s = float(timeout_s)
        self._ref_count_actors = ref_count_actors
        # Submitters may be worker executor threads; completion callbacks run
        # on the client loop (or inline if already done). Keep bookkeeping under
        # one condition, and never wait for RPC completion while holding it.
        self._release_condition = threading.Condition()
        self._pending_releases: dict[concurrent.futures.Future[None], set[tuple[str, str]]] = {}
        self._release_errors: list[Exception] = []
        _ensure_host_identity()

    @classmethod
    def start(
        cls,
        store_name: str,
        *,
        local_cache_bytes: int = 0,
        ref_count_actors: RefCountActor | None = None,
    ) -> None:
        """Start TorchStore storage for the current process."""
        from verl.runtime.object_store import _start_object_store
        from verl.single_controller.monarch.patches.torchstore_compat import install_torchstore_patches

        install_torchstore_patches()
        _start_object_store(cls(store_name, local_cache_bytes=local_cache_bytes, ref_count_actors=ref_count_actors))

    def put(self, key: str, value: Any, /) -> TorchStoreReference:
        """Store one value under the caller's non-empty logical key."""
        if not isinstance(key, str) or not key:
            raise ValueError(f"key must be a non-empty str, got {key!r}")
        import torch

        self._wait_for_deletes({(self._store_name, key)})
        self._local_cache.invalidate({(self._store_name, key)})
        import torchstore

        _run(torchstore.put(key, value, store_name=self._store_name), timeout_s=self._timeout_s)
        return TorchStoreReference(store_name=self._store_name, key=key, is_tensor=isinstance(value, torch.Tensor))

    def put_many(self, entries: list[tuple[str, Any]], /) -> list[TorchStoreReference]:
        """Store uniquely keyed values in one native TorchStore batch."""
        if not entries:
            return []
        self._validate_put_many_entries(entries)
        values_by_key = dict(entries)
        import torch

        self._wait_for_deletes({(self._store_name, key) for key in values_by_key})
        self._local_cache.invalidate({(self._store_name, key) for key in values_by_key})
        import torchstore

        _run(torchstore.put_batch(values_by_key, store_name=self._store_name), timeout_s=self._timeout_s)
        return [
            TorchStoreReference(store_name=self._store_name, key=key, is_tensor=isinstance(value, torch.Tensor))
            for key, value in entries
        ]

    def put_shared(self, key: str, value: Any, readers: int, /) -> list[TorchStoreReference]:
        """Publish once under a fresh physical UUID, with a lease per reader.

        The caller's key is a prefix; an independent physical key prevents
        cross-volume publishers from overwriting/deleting each other's values.
        """
        if type(readers) is not int or readers < 1:
            raise ValueError("readers must be a positive integer")
        self._validate_put_many_entries([(key, value)])
        from verl.single_controller.monarch.object_store.ref_count import (
            abort_ref_count,
            register_ref_count,
            select_ref_count_actor,
        )

        if self._ref_count_actors is None:
            raise RuntimeError("shared snapshots require the global TorchStore backend")
        owner = _run(select_ref_count_actor(self._store_name, self._ref_count_actors), timeout_s=self._timeout_s)
        snapshot_key = f"{key}-{uuid.uuid4().hex}"
        tokens = tuple(uuid.uuid4().hex for _ in range(readers))
        try:
            _run(register_ref_count(owner, snapshot_key, tokens), timeout_s=self._timeout_s)
            from verl.single_controller.monarch.patches.torchstore_compat import _prepare_shared_snapshot

            reference = self.put(snapshot_key, _prepare_shared_snapshot(value))
        except BaseException as error:
            # Reserve before writing: a duplicate shared key cannot overwrite
            # another publisher. Abort checks this attempt's unique tokens, so
            # an ambiguous or rejected registration cannot delete its payload.
            try:
                _run(abort_ref_count(owner, snapshot_key, tokens), timeout_s=self._timeout_s)
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise
        return [replace(reference, shared_reader=token, shared_owner=owner) for token in tokens]

    @property
    def local_cache(self) -> LocalCache:
        """Process-local read cache; snapshot eligibility belongs to the reader.

        Writes and deletes invalidate entries here, regardless of which reader
        populated them. Generic ObjectStore reads never use this cache.
        """
        return self._local_cache

    def get(self, reference: TorchStoreReference, /) -> Any:
        """Resolve a self-addressing reference through its producer's store."""
        if not isinstance(reference, TorchStoreReference):
            raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")

        import torchstore

        if os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1":
            self._local_cache.clear()
        self._wait_for_deletes({(reference.store_name, reference.key)})
        return _run(torchstore.get(reference.key, store_name=reference.store_name), timeout_s=self._timeout_s)

    def get_many(
        self,
        references: list[TorchStoreReference],
        /,
        *,
        row_ranges: dict[TorchStoreReference, TensorRowRange] | None = None,
    ) -> list[Any]:
        """Resolve multiple references concurrently on the shared client loop.

        ``row_ranges`` optionally narrows tensor transfers; non-tensor objects
        retain their complete values. Duplicate references share one fetched
        value and result ordering follows ``references``.

        This is a concrete TorchStore optimization; the backend-neutral
        :class:`ObjectStore` contract remains the minimal single-value API.
        """
        if not references:
            return []
        for reference in references:
            if not isinstance(reference, TorchStoreReference):
                raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")
        if os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1":
            self._local_cache.clear()
        self._wait_for_deletes({(ref.store_name, ref.key) for ref in references})
        return _run(_get_many(references, row_ranges=row_ranges), timeout_s=self._timeout_s)

    def delete(self, reference: TorchStoreReference, /) -> None:
        """Queue cleanup after the caller stops using one reference.

        Completion and failures are observed by close. Reusing this client's
        same key waits for pending deletion; cross-client reuse needs close.

        Args:
            Self-addressing TorchStore reference the caller will stop using.
        """
        if not isinstance(reference, TorchStoreReference):
            raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")

        import torchstore

        self._local_cache.invalidate({(reference.store_name, reference.key)})
        if reference.shared_reader is not None:
            self._defer_shared_release(reference)
        else:
            self._defer_cleanup(
                lambda: torchstore.delete(reference.key, store_name=reference.store_name),
                delete_keys={(reference.store_name, reference.key)},
            )

    def delete_many(self, references: list[TorchStoreReference], /) -> None:
        """Delete references in native batches grouped by their source store."""
        if not references:
            return
        for reference in references:
            if not isinstance(reference, TorchStoreReference):
                raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")
        self._local_cache.invalidate({(ref.store_name, ref.key) for ref in references})
        ordinary = [ref for ref in references if ref.shared_reader is None]
        for reference in references:
            if reference.shared_reader is not None:
                self._defer_shared_release(reference)
        if ordinary:
            self._defer_cleanup(
                lambda: _delete_many(ordinary),
                delete_keys={(ref.store_name, ref.key) for ref in ordinary},
            )

    def _defer_shared_release(self, reference: TorchStoreReference) -> None:
        from verl.single_controller.monarch.object_store.ref_count import release_ref

        if reference.shared_owner is None or reference.shared_reader is None:
            raise ValueError("shared reference has no release owner/token")
        self._defer_cleanup(lambda: release_ref(reference.shared_owner, reference.key, reference.shared_reader))

    def _defer_cleanup(
        self,
        operation: Callable[[], Coroutine[Any, Any, None]],
        *,
        delete_keys: set[tuple[str, str]] | None = None,
    ) -> None:
        with self._release_condition:
            errors, self._release_errors = self._release_errors, []
            self._raise_release_errors(errors)
            # Bound retained RPCs; backpressure is only needed when cleanup
            # falls behind. Condition.wait releases the bookkeeping lock.
            if not self._release_condition.wait_for(lambda: len(self._pending_releases) < 256, timeout=self._timeout_s):
                raise TimeoutError("pending object-store cleanup queue is full")
            errors, self._release_errors = self._release_errors, []
            self._raise_release_errors(errors)
            future = _CLIENT_LOOP.submit(operation(), timeout_s=None)
            self._pending_releases[future] = delete_keys or set()
        # Completed futures can invoke callbacks inline, outside the lock.
        future.add_done_callback(self._released)

    def _wait_for_deletes(self, keys: set[tuple[str, str]]) -> None:
        """Order this client's later same-key reads/writes after its deletes.

        Other keys can proceed while cleanup is pending. Independent clients
        must close their client before handing a reusable key to another writer.
        """
        with self._release_condition:
            if not self._release_condition.wait_for(
                lambda: not any(keys & pending for pending in self._pending_releases.values()),
                timeout=self._timeout_s,
            ):
                raise TimeoutError("timed out waiting for same-key object-store deletion")
            errors, self._release_errors = self._release_errors, []
        self._raise_release_errors(errors)

    def _released(self, future: concurrent.futures.Future[None]) -> None:
        try:
            future.result()
        except Exception as error:
            failure = error
        else:
            failure = None
        with self._release_condition:
            if failure is not None:
                self._release_errors.append(failure)
            del self._pending_releases[future]
            self._release_condition.notify_all()

    @staticmethod
    def _raise_release_errors(errors: list[Exception]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("object-store cleanup failed", errors)

    def close(self) -> None:
        """Drain deletes and shared releases while the global backend is alive.

        The lifecycle owner stops new operations before this barrier. A timeout
        retains pending tasks; it does not pretend their remote work stopped.
        """
        with self._release_condition:
            if not self._release_condition.wait_for(lambda: not self._pending_releases, timeout=self._timeout_s):
                raise TimeoutError("timed out draining object-store cleanup")
            errors, self._release_errors = self._release_errors, []
        self._raise_release_errors(errors)
        self._local_cache.clear()


def _ensure_host_identity() -> None:
    # HostStrategy requires HOSTNAME for clients; volume IDs fall back to the
    # socket hostname. Align both sides on the same spelling.
    os.environ.setdefault("HOSTNAME", socket.gethostname())


def _keys_by_store(references: list[TorchStoreReference]) -> dict[str, list[str]]:
    """Group unique keys in first-reference order for native batch operations."""
    grouped: dict[str, dict[str, None]] = {}
    for reference in references:
        grouped.setdefault(reference.store_name, {})[reference.key] = None
    return {store: list(keys) for store, keys in grouped.items()}


async def _get_many(
    references: list[TorchStoreReference],
    *,
    row_ranges: dict[TorchStoreReference, TensorRowRange] | None = None,
) -> list[Any]:
    import torchstore

    keys_by_store = _keys_by_store(references)

    store_names = list(keys_by_store)

    async def fetch_store(store_name: str) -> dict[str, Any]:
        ranges = {
            ref.key: (
                TensorRowRange(selection.shape, selection.start, selection.stop)
                if os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1"
                else selection
            )
            for ref, selection in (row_ranges or {}).items()
            if ref.store_name == store_name and (not selection.segments or ref.is_tensor)
        }
        if not ranges:
            return await torchstore.get_batch(keys_by_store[store_name], store_name=store_name)

        from torchstore.transport import Request, TensorSlice

        requests = []
        for key in keys_by_store[store_name]:
            selection = ranges.get(key)
            segments = (
                ((None, None),) if selection is None else (selection.segments or ((selection.start, selection.stop),))
            )
            for start, stop in segments:
                tensor_slice = None
                if selection is not None:
                    tensor_slice = TensorSlice(
                        offsets=(start,) + (0,) * (len(selection.shape) - 1),
                        coordinates=(),
                        global_shape=selection.shape,
                        local_shape=(stop - start,) + selection.shape[1:],
                        mesh_shape=(),
                    )
                requests.append(Request.from_any(key, None, tensor_slice=tensor_slice))
        from verl.single_controller.monarch.patches.torchstore_batch import fetch_torchstore_tensor_rows

        client = await torchstore.client(store_name)
        return await fetch_torchstore_tensor_rows(client, requests, ranges)

    tasks = [asyncio.create_task(fetch_store(store_name)) for store_name in store_names]
    batches = await _gather_tasks(tasks)
    values_by_store = dict(zip(store_names, batches, strict=True))
    return [values_by_store[reference.store_name][reference.key] for reference in references]


async def _delete_many(references: list[TorchStoreReference]) -> None:
    import torchstore

    keys_by_store = _keys_by_store(references)

    tasks = [
        asyncio.create_task(torchstore.delete_batch(keys, store_name=store_name))
        for store_name, keys in keys_by_store.items()
    ]
    await _gather_tasks(tasks)


async def _gather_tasks(tasks: list[asyncio.Task[Any]]) -> list[Any]:
    try:
        return await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _run(awaitable, *, timeout_s: float):
    return _CLIENT_LOOP.run(awaitable, timeout_s=timeout_s)


__all__ = ["TorchStoreObjectStore", "TorchStoreReference"]
