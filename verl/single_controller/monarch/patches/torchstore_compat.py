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

"""Temporary compatibility/optimization patches for the pinned TorchStore SDK.

These native transport and storage-volume overrides bridge SDK behavior required
by this adapter. Retire each override when the supported upstream SDK passes its
wire-state, lifetime, and transport regressions without it."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.distributed.tensor import DTensor
from torchstore.client import LocalClient
from torchstore.logging import LatencyTracker
from torchstore.storage_volume import InMemoryStore, StorageVolume
from torchstore.transport import create_transport_buffer
from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer
from torchstore.transport.monarch_rdma import MonarchRDMATransportBuffer
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.shared_memory import (
    SharedMemoryCache,
    SharedMemoryTransportBuffer,
    ShmContext,
    allocate_shared_tensor,
)
from torchstore.transport.torchcomms.buffer import TorchCommsRdmaTransportBuffer
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from verl.single_controller.monarch.object_store.ref_count import RefCountActor
    from verl.single_controller.monarch.object_store.store import TorchStoreReference

_PATCH_LOCK = threading.Lock()
_PATCH_SENTINEL = "_verl_torchstore_compat_patch"


class _TransportResolutionOnce(logging.Filter):
    """Keep each fixed transport/availability combination once per process."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO or not record.getMessage().startswith("[ts-transport] resolved="):
            return True
        message = record.getMessage()
        # Buffers can be created on multiple worker/client threads. Claim the
        # first log atomically; the SDK emits a finite set of type/flag tuples.
        with self._lock:
            if message in self._seen:
                return False
            self._seen.add(message)
            return True


def install_torchstore_patches() -> bool:
    """Install all pinned TorchStore fixes once per process.

    The pinned TorchStore snapshot serializes ``storage_volume_ref`` with RDMA
    and Monarch RPC transport buffers. That reference owns a process-local
    transport context, including the shared-memory cache. Monarch reconstructs
    the cached storages in the receiving process without pinning them there;
    destroying the copied cache then attempts to unregister unowned pointers.

    Returns ``True`` only when this call installs the process-wide patch.
    """
    with _PATCH_LOCK:
        if TransportBuffer.__dict__.get(_PATCH_SENTINEL, False):
            return False

        logging.getLogger("torchstore.transport").addFilter(_TransportResolutionOnce())

        GlooTransportBuffer.perform_handshake = _perform_handshake
        GlooTransportBuffer.requires_handshake = _requires_handshake
        GlooTransportBuffer._post_handshake = _post_handshake
        TorchCommsRdmaTransportBuffer.perform_handshake = _perform_handshake
        LocalClient.put_batch = _put_batch_metadata_notification
        SharedMemoryTransportBuffer.put_to_storage_volume = _put_shm_objects
        _install_reference_reducer()
        _install_shared_snapshot_transport()
        _install_concurrent_volume_endpoints()

        original_buffer_getstate = _get_getstate(TransportBuffer)
        original_rpc_getstate = _get_getstate(MonarchRPCTransportBuffer)

        def transport_buffer_getstate(self: Any) -> dict[str, Any]:
            state = _object_state(self, original_buffer_getstate)
            state["storage_volume_ref"] = None
            return state

        def monarch_rpc_getstate(self: Any) -> dict[str, Any]:
            state = _object_state(self, original_rpc_getstate)
            state["storage_volume_ref"] = None
            state["inplace_tensor"] = None
            return state

        TransportBuffer.__getstate__ = transport_buffer_getstate
        MonarchRPCTransportBuffer.__getstate__ = monarch_rpc_getstate
        setattr(TransportBuffer, _PATCH_SENTINEL, True)
        return True


def _install_reference_reducer() -> None:
    """Use the constructor directly for native TorchStore handle decoding.

    Register the exact type with Monarch's picklers so other dataclasses and
    subclasses retain their normal reduction. Actor handles remain normal
    fields and continue through Monarch's native reference serialization.
    """
    import monarch._src.actor.pickle as native_pickle

    from verl.single_controller.monarch.object_store.store import TorchStoreReference

    native_pickle._ensure_torch_pickle()
    native_pickle._TorchPickler.dispatch_table[TorchStoreReference] = _reduce_reference
    native_pickle._Pickler._dispatch_table[TorchStoreReference] = _reduce_reference


def _reduce_reference(
    reference: TorchStoreReference,
) -> tuple[type[TorchStoreReference], tuple[str, str, bool, str | None, RefCountActor | None]]:
    return type(reference), (
        reference.store_name,
        reference.key,
        reference.is_tensor,
        reference.shared_reader,
        reference.shared_owner,
    )


_SNAPSHOT_CODEC = "monarch-storage-bucket-v1"
_SnapshotSlot = tuple[Literal["storage", "native", "alias"], int, torch.dtype | None]


@dataclass(frozen=True)
class _SnapshotTag:
    codec: str
    metadata: bytes
    segments: tuple[tuple[int, int], ...]
    slots: tuple[_SnapshotSlot, ...]
    native_values: tuple[Any, ...]
    data_bytes: int


@dataclass(frozen=True)
class _EncodedSnapshot:
    tag: _SnapshotTag
    data: torch.Tensor


class _UnsupportedSnapshot(Exception):
    """Keep an unknown Monarch value on its existing native object path."""


class _SharedSnapshot:
    """One immutable publication and its volume-owned, lazily encoded data.

    Keeping ``value`` alive preserves its ownership until native key deletion.
    The cache never travels with PUT; readers receive the tag and tensor data.
    """

    def __init__(self, value: Any) -> None:
        self.value = value
        self._encoded: _EncodedSnapshot | None = None
        self._supported = True

    def __reduce__(self) -> tuple[type[_SharedSnapshot], tuple[Any]]:
        return _SharedSnapshot, (self.value,)

    def encode(self) -> _EncodedSnapshot | None:
        if self._encoded is None and self._supported:
            try:
                self._encoded = _encode_snapshot(self.value)
            except _UnsupportedSnapshot:
                self._supported = False
        return self._encoded


def _prepare_shared_snapshot(value: Any) -> Any:
    """Leave top-level tensors on their native path; mark object snapshots."""
    return value if isinstance(value, torch.Tensor) else _SharedSnapshot(value)


def _encode_snapshot(value: Any) -> _EncodedSnapshot:
    import cloudpickle
    from monarch._src.actor import pickle as native_pickle
    from monarch._src.actor.actor_mesh import ActorMesh
    from monarch.actor import HostMesh, Port, ProcMesh

    handles = (ActorMesh, HostMesh, ProcMesh, Port)
    monarch_types: dict[type, bool] = {}

    def select(item: Any) -> bool:
        kind = type(item)
        if kind is torch.storage.TypedStorage:
            return item._untyped_storage.device.type == "cpu"
        if kind is torch.storage.UntypedStorage:
            return item.device.type == "cpu"
        if kind in handles:
            # A by-value actor class may itself contain application payload.
            if kind is ActorMesh and not cloudpickle.cloudpickle._should_pickle_by_reference(item._class):
                raise _UnsupportedSnapshot()
            return True
        native = monarch_types.get(kind)
        if native is None:
            native = any(
                isinstance(base.__module__, str) and base.__module__.startswith("monarch.") for base in kind.__mro__
            )
            monarch_types[kind] = native
        if native:
            # Containers such as ValueMesh may share payload with the outer
            # object. Do not split that graph across independent pickle memos.
            raise _UnsupportedSnapshot()
        return False

    saved, buffer = native_pickle.flatten(value, select)
    metadata = buffer.freeze().read()
    storages: list[torch.UntypedStorage] = []
    storage_indices: dict[int, int] = {}
    saved_indices: dict[int, int] = {}
    segments: list[tuple[int, int]] = []
    slots: list[_SnapshotSlot] = []
    native_values: list[Any] = []
    size = 0
    for item in saved:
        identity = id(item)
        if identity in saved_indices:
            slots.append(("alias", saved_indices[identity], None))
            continue
        saved_indices[identity] = len(slots)
        if type(item) in (torch.storage.TypedStorage, torch.storage.UntypedStorage):
            typed = type(item) is torch.storage.TypedStorage
            storage = item._untyped_storage if typed else item
            storage_identity = id(storage)
            if storage_identity not in storage_indices:
                storage_indices[storage_identity] = len(storages)
                storages.append(storage)
                segments.append((size, storage.nbytes()))
                size += storage.nbytes()
            slots.append(("storage", storage_indices[storage_identity], item.dtype if typed else None))
        else:
            slots.append(("native", len(native_values), None))
            native_values.append(item)
    # A nonempty byte carrier also supports object-only and empty-storage
    # snapshots on transports that cannot map a zero-byte SHM segment.
    data = allocate_shared_tensor(torch.Size([max(1, size)]), torch.uint8)
    for storage, (offset, length) in zip(storages, segments, strict=True):
        data[offset : offset + length].copy_(torch.empty(0, dtype=torch.uint8).set_(storage))
    tag = _SnapshotTag(_SNAPSHOT_CODEC, metadata, tuple(segments), tuple(slots), tuple(native_values), size)
    return _EncodedSnapshot(tag, data)


def _decode_snapshot(tag: _SnapshotTag, data: torch.Tensor) -> Any:
    from monarch._src.actor import pickle as native_pickle

    if not isinstance(tag, _SnapshotTag) or tag.codec != _SNAPSHOT_CODEC:
        raise ValueError("unsupported TorchStore snapshot codec")
    if tag.data_bytes < 0:
        raise ValueError("invalid TorchStore snapshot data size")
    if (
        not isinstance(data, torch.Tensor)
        or data.device.type != "cpu"
        or data.dtype != torch.uint8
        or data.ndim != 1
        or not data.is_contiguous()
        or data.numel() != max(1, tag.data_bytes)
    ):
        raise ValueError("invalid TorchStore snapshot data tensor")
    base = data.storage_offset()
    packed_storage = data.untyped_storage()
    restored: list[torch.UntypedStorage] = []
    for offset, length in tag.segments:
        if offset < 0 or length < 0 or offset + length > tag.data_bytes:
            raise ValueError("invalid TorchStore snapshot storage segment")
        # Own each original storage independently of the receive buffer; views
        # of the same source storage still share this one restored storage.
        restored.append(packed_storage[base + offset : base + offset + length].clone())
    values: list[Any] = []
    for kind, index, dtype in tag.slots:
        if kind == "alias":
            if not 0 <= index < len(values):
                raise ValueError("invalid TorchStore snapshot alias")
            values.append(values[index])
        elif kind == "native":
            if not 0 <= index < len(tag.native_values):
                raise ValueError("invalid TorchStore snapshot native value")
            values.append(tag.native_values[index])
        elif kind == "storage":
            if not 0 <= index < len(restored):
                raise ValueError("invalid TorchStore snapshot storage value")
            storage = restored[index]
            values.append(
                storage
                if dtype is None
                else torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)
            )
        else:
            raise ValueError("invalid TorchStore snapshot slot")
    return native_pickle.unflatten(tag.metadata, values)


def _install_shared_snapshot_transport() -> None:
    original_meta = InMemoryStore._get_meta
    original_get = InMemoryStore.get

    def get_meta(store: InMemoryStore, request: Request) -> Any:
        entry = store.kv.get(request.key)
        if isinstance(entry, dict) and isinstance(entry.get("obj"), _SharedSnapshot):
            encoded = entry["obj"].encode()
            if encoded is not None:
                return encoded.data.shape, encoded.data.dtype
        return original_meta(store, request)

    async def get(store: InMemoryStore, buffer: TransportBuffer, requests: list[Request]) -> TransportBuffer:
        if not any(
            isinstance(store.kv.get(request.key), dict)
            and isinstance(store.kv[request.key].get("obj"), _SharedSnapshot)
            for request in requests
        ):
            return await original_get(store, buffer, requests)
        entries = []
        tags: list[_SnapshotTag | None] = []
        for request in requests:
            value = store._get_data(request)
            tag = None
            if isinstance(value, _SharedSnapshot):
                encoded = value.encode()
                if encoded is None:
                    value = value.value
                else:
                    tag, value = encoded.tag, encoded.data
                    request = replace(request, is_object=False)
            entries.append((request, value))
            tags.append(tag)
        if any(tag is not None for tag in tags):
            buffer._verl_snapshot_tags = tuple(tags)
        await buffer.handle_get_request(store.transport_context, entries)
        return buffer

    InMemoryStore._get_meta = get_meta
    InMemoryStore.get = get
    TransportBuffer._get_requests = _get_requests_with_snapshots


async def _get_requests_with_snapshots(client: TransportBuffer, requests: list[Request]) -> list[Any]:
    """Keep native get/handshake/drop sequencing around every transport type."""
    tracker = LatencyTracker("get")
    metadata = [request.meta_only() for request in requests]
    try:
        if client.requires_handshake(requests):
            await client.perform_handshake(requests, metadata, tracker)
        await client._pre_get_hook(requests)
        tracker.track_step("_pre_get_hook")
        response = await client.storage_volume_ref.volume.get.call_one(client, metadata)
        tags = getattr(response, "_verl_snapshot_tags", None)
        packed_keys = (
            {request.key for request, tag in zip(requests, tags, strict=True) if tag is not None}
            if tags is not None
            else set()
        )
        try:
            values = await client._handle_storage_volume_response(requests, response)
            if tags is not None:
                values = [
                    _decode_snapshot(tag, value) if tag is not None else value
                    for tag, value in zip(tags, values, strict=True)
                ]
        finally:
            if packed_keys and isinstance(client, SharedMemoryTransportBuffer):
                client.storage_volume_ref.transport_context.get(SharedMemoryCache).delete(packed_keys)
        tracker.track_step("volume.get.call")
        await client._post_request_success()
        tracker.track_step("_post_request_success")
    finally:
        await client.drop()
        tracker.track_step("drop")
        tracker.track_e2e()
    return values


class _VolumeGate:
    """One actor-loop owner: concurrent reads, exclusive writes/handshakes."""

    def __init__(self) -> None:
        self._turnstile = asyncio.Lock()
        self._drained = asyncio.Event()
        self._drained.set()
        self._readers = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._turnstile:
            self._readers += 1
            self._drained.clear()
        try:
            yield
        finally:
            self._readers -= 1
            if not self._readers:
                self._drained.set()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._turnstile:
            await self._drained.wait()
            yield


def _install_concurrent_volume_endpoints() -> None:
    from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask
    from monarch._src.actor.concurrent import _explicit_response_signature
    from monarch.actor import ActorError, Future, concurrent_endpoint, context, endpoint

    def wrap(name: str, original: Any) -> Any:
        method = original._method

        if name == "reset":
            # Native Controller handles can predate client/volume patching.
            # Preserve their ordinary response protocol during teardown.
            @wraps(method)
            async def reset(volume: StorageVolume, *args: Any, **kwargs: Any) -> Any:
                gate = volume.__dict__.get("_verl_volume_gate")
                if gate is None:
                    gate = _VolumeGate()
                    volume.__dict__["_verl_volume_gate"] = gate
                async with gate.write():
                    return await method(volume, *args, **kwargs)

            return endpoint(reset, propagate=original._propagator, instrument=original._instrument)

        @wraps(method)
        async def run(volume: StorageVolume, port: Any, *args: Any, **kwargs: Any) -> None:
            gate = volume.__dict__.get("_verl_volume_gate")
            if gate is None:
                gate = _VolumeGate()
                volume.__dict__["_verl_volume_gate"] = gate
            buffer = args[0] if args else kwargs.get("transport_buffer")
            readable = name == "get_meta" or (
                name == "get"
                and isinstance(
                    buffer, SharedMemoryTransportBuffer | MonarchRDMATransportBuffer | MonarchRPCTransportBuffer
                )
            )
            try:
                async with gate.read() if readable else gate.write():
                    result = await method(volume, *args, **kwargs)
                    response = port.resolve_and_send(result)
                    if isinstance(response, PythonTask):
                        await Future._from_coro(response)
                    else:
                        await response
            except Exception as error:
                actor = context().actor_instance
                port.exception(ActorError(error, f"Actor call {actor.name}.{name} failed."))

        run.__signature__ = _explicit_response_signature(method, already_explicit=False)
        result = concurrent_endpoint(
            run,
            propagate=original._propagator,
            explicit_response_port=True,
            instrument=original._instrument,
        )
        result.__set_name__(StorageVolume, name)
        return result

    for name in ("get", "get_meta", "put", "handshake", "delete", "delete_batch", "reset"):
        setattr(StorageVolume, name, wrap(name, StorageVolume.__dict__[name]))


async def _put_shm_objects(self: SharedMemoryTransportBuffer, requests: list[Request]) -> None:
    """Object-only writes do not allocate or reuse tensor shared memory.

    Keep tensor and mixed batches on the native handshake path. For objects,
    the payload lives in ShmContext; the volume request needs only metadata.
    Native request handling still owns write completion and finally-drop.
    """
    if requests and all(request.is_object for request in requests):
        self._needs_handshake = False
        self._contexts = [ShmContext(objects=request.objects, use_rpc=True) for request in requests]
        metadata = [Request(key=request.key, tensor_slice=request.tensor_slice, is_object=True) for request in requests]
        await TransportBuffer.put_to_storage_volume(self, metadata)
    else:
        self._needs_handshake = True
        await TransportBuffer.put_to_storage_volume(self, requests)


@torch.no_grad()
async def _put_batch_metadata_notification(self: LocalClient, entries: dict[str, Any]) -> None:
    """Do not send object payloads again when updating the controller index.

    Request.meta_only() intentionally retains objects for storage transports.
    Controller._notify_put only consumes key/is_object/tensor_slice, so its
    notification must be constructed separately from the storage request.
    """
    assert isinstance(entries, dict) and entries, "put_batch requires a non-empty dict"
    latency = LatencyTracker("put_batch")
    requests = [
        Request.from_any(key, value) if isinstance(value, torch.Tensor | DTensor) else Request.from_objects(key, value)
        for key, value in entries.items()
    ]
    volume = self.strategy.select_storage_volume()
    transport = create_transport_buffer(volume)
    latency.track_step("create transport buffer")
    await transport.put_to_storage_volume(requests)
    latency.track_step("put_to_storage_volume")
    metadata = [
        Request(key=request.key, tensor_slice=request.tensor_slice, is_object=request.is_object) for request in requests
    ]
    await self._controller.notify_put_batch.call(metadata, volume.volume_id)
    latency.track_step("notify_put_batch")
    latency.track_e2e()


def _get_getstate(cls: type[Any]) -> Callable[[Any], Any] | None:
    getstate = cls.__dict__.get("__getstate__")
    if getstate is not None and not callable(getstate):
        raise RuntimeError(f"{cls.__name__}.__getstate__ is not callable")
    return getstate


def _object_state(self: Any, getstate: Callable[[Any], Any] | None) -> dict[str, Any]:
    state = self.__dict__ if getstate is None else getstate(self)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported TorchStore serialization state: {type(state).__name__}")
    return dict(state)


class _GlooAddresses(TransportCache):
    """Connection addresses owned by the same context as its process groups."""

    def __init__(self) -> None:
        self.addresses: dict[str, tuple[str, int, str]] = {}

    def clear(self) -> None:
        self.addresses.clear()


async def _perform_handshake(
    self: TransportBuffer,
    requests: list[Request],
    meta_requests: list[Request],
    latency_tracker: LatencyTracker | None = None,
) -> None:
    # Connection-only transports rendezvous here; it never reads existing values. Passing GET
    # slices to InMemoryStore.handshake invokes its PUT-only type validation.
    # The native server accepts an empty list and creates the same connection.
    await TransportBuffer.perform_handshake(self, requests, [], latency_tracker)


def _requires_handshake(self: GlooTransportBuffer, requests: list[Request]) -> bool:
    addresses = self.storage_volume_ref.transport_context.get(_GlooAddresses).addresses
    address = addresses.get(self.storage_volume_ref.volume_id)
    if address is None:
        return True
    self.master_addr, self.master_port, self.store_key = address
    return False


async def _post_handshake(
    self: GlooTransportBuffer,
    handshake_results: list[Any],
    requests: list[Request],
) -> None:
    assert self._pg_task is not None
    assert self.master_addr is not None and self.master_port is not None and self.store_key is not None
    pg = await self._pg_task
    context = self.storage_volume_ref.transport_context
    context.get(GlooProcessGroupCache).put(self.store_key, pg)
    context.get(_GlooAddresses).addresses[self.storage_volume_ref.volume_id] = (
        self.master_addr,
        self.master_port,
        self.store_key,
    )
    self._tcp_store = None
    self._pg_task = None


__all__ = ["install_torchstore_patches"]
