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

"""Single-writer leases for shared snapshots in a named TorchStore.

Only metadata crosses these endpoints. Payload reads remain native TorchStore
reads. The Runtime's TorchStore backend owns the registry and closes all leases
(including consumers that never arrived) before shutting down its storage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from monarch.actor import Actor, endpoint

from verl.single_controller.base.errors import ExceptionGroup


async def _delete_stored_key(store_name: str, key: str) -> None:
    import torchstore

    await torchstore.delete_batch([key], store_name=store_name)


@dataclass
class _RefCountEntry:
    readers: set[str]
    deleting: asyncio.Task[None] | None = field(default=None)


class RefCountRegistry:
    """Own reader tokens on one actor event loop; no thread may mutate it."""

    def __init__(self, store_name: str) -> None:
        self._store_name = store_name
        self._entries: dict[str, _RefCountEntry] = {}
        self._closed = False

    def register(self, key: str, readers: tuple[str, ...]) -> None:
        """Reserve a fresh key before its publisher may write the payload."""
        if self._closed:
            raise RuntimeError("reference count registry is closed")
        if not readers or len(set(readers)) != len(readers):
            raise ValueError("reader tokens must be nonempty and unique")
        if key in self._entries:
            raise ValueError(f"reference key was reused: {key}")
        self._entries[key] = _RefCountEntry(set(readers))

    async def release(self, key: str, reader: str) -> None:
        """Consume one token; only the last reader starts native deletion."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.readers.discard(reader)
        if not entry.readers:
            await self._delete(key, entry)

    async def _delete(self, key: str, entry: _RefCountEntry) -> None:
        # The transition before the await is atomic on the owner actor loop.
        # All simultaneous final releases join the same native deletion.
        task = entry.deleting
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.create_task(_delete_stored_key(self._store_name, key))
            entry.deleting = task
        await asyncio.shield(task)
        if self._entries.get(key) is entry:
            del self._entries[key]

    async def abort(self, key: str, readers: tuple[str, ...]) -> None:
        """Roll back only the failed publisher's own unissued reservation."""
        entry = self._entries.get(key)
        if entry is None or entry.readers != set(readers):
            return
        await self._delete(key, entry)

    async def close(self) -> None:
        """Reclaim unconsumed snapshots; retain failures for a close retry."""
        self._closed = True
        results = await asyncio.gather(
            *(self._delete(key, entry) for key, entry in list(self._entries.items())),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
        errors = [result for result in results if isinstance(result, Exception)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("reference cleanup failed", errors)


class RefCountActor(Actor):
    def __init__(self, store_name: str) -> None:
        from verl.single_controller.monarch.patches.torchstore_compat import install_torchstore_patches

        install_torchstore_patches()
        self._registry = RefCountRegistry(store_name)

    @endpoint
    async def register(self, key: str, readers: tuple[str, ...]) -> None:
        self._registry.register(key, readers)

    @endpoint
    async def release(self, key: str, reader: str) -> None:
        await self._registry.release(key, reader)

    @endpoint
    async def abort(self, key: str, readers: tuple[str, ...]) -> None:
        await self._registry.abort(key, readers)

    @endpoint
    async def close(self) -> None:
        await self._registry.close()


async def select_ref_count_actor(store_name: str, owners: RefCountActor) -> RefCountActor:
    """Select the counter colocated with the producer's native payload volume."""
    import torchstore

    client = await torchstore.client(store_name=store_name)
    volume = client.strategy.select_storage_volume()
    coordinate = client.strategy.volume_id_to_coord[volume.volume_id]
    return owners.slice(**coordinate)


async def register_ref_count(owner: RefCountActor, key: str, readers: tuple[str, ...]) -> None:
    await owner.register.call_one(key, readers)


async def release_ref(owner: RefCountActor, key: str, reader: str) -> None:
    await owner.release.call_one(key, reader)


async def abort_ref_count(owner: RefCountActor, key: str, readers: tuple[str, ...]) -> None:
    await owner.abort.call_one(key, readers)


async def close_ref_counts(owners: RefCountActor) -> None:
    """Drain every volume's counter before the owning volume mesh is stopped."""
    await owners.close.call()
