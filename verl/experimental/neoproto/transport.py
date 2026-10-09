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
"""Ref-table serialization and ObjectStore reference ownership for NeoProto."""

from __future__ import annotations

import uuid
from typing import Any

from verl.experimental.neoproto.storage.engine import RefTable


class _RefTableTransport:
    """One ref table whose every serialization owns a fresh store reference.

    A local table may be shared by all prepared chunks in the sending process,
    but each serialized occurrence gets an independent store key. The receiving
    occurrence caches the resolved table and consumes its reference exactly
    once, so re-pickling a restored NeoProto creates a new transport lifetime.
    """

    __slots__ = ("_issued_references", "_key_prefix", "_prepared_reference", "_reference", "_table")

    def __init__(self, table: RefTable, *, key_prefix: str) -> None:
        self._issued_references = []
        self._key_prefix = key_prefix
        self._prepared_reference = None
        self._table = table
        self._reference = None

    @classmethod
    def prepare_many(cls, transports: list[_RefTableTransport]) -> None:
        from verl.runtime import put_many

        entries = []
        for transport in transports:
            key = f"{transport._key_prefix}-{uuid.uuid4().hex}"
            entries.append((key, transport._table))
        references = put_many(entries)
        for transport, reference in zip(transports, references, strict=True):
            transport._prepared_reference = reference

    @classmethod
    def prepare_shared(cls, table: RefTable, *, key_prefix: str, readers: int) -> list[_RefTableTransport]:
        """Publish one snapshot with an independent reference per receiver."""
        from verl.runtime.object_store import _object_store
        from verl.single_controller.ray.object_store import RayObjectStore

        if type(readers) is not int or readers < 1:
            raise ValueError("readers must be a positive integer")
        key = f"{key_prefix}-{uuid.uuid4().hex}"
        store = _object_store()
        # TODO: Move these backend-specific ref-table publication paths into
        # NeoProto backend adapters; keep shared publication out of ObjectStore.
        if isinstance(store, RayObjectStore):
            references = [store.put(key, table)] * readers
        else:
            # Importing the TorchStore client starts its event-loop thread; keep
            # that out of Ray processes.
            from verl.single_controller.monarch.object_store import TorchStoreObjectStore

            if not isinstance(store, TorchStoreObjectStore):
                raise TypeError(f"unsupported NeoProto shared-publication store: {type(store)!r}")
            references = store.put_shared(key, table, readers)
        transports = [cls(table, key_prefix=f"{key_prefix}-{rank}") for rank in range(readers)]
        for transport, reference in zip(transports, references, strict=True):
            transport._prepared_reference = reference
        return transports

    def __getstate__(self):
        reference = self._prepared_reference
        if reference is None:
            table = self.get()
            from verl.runtime import put

            key = f"{self._key_prefix}-{uuid.uuid4().hex}"
            reference = put(key, table)
        self._prepared_reference = None
        # Standard pickle bytes do not participate in Ray's distributed
        # reference counting. Retain issued handles on the sending transport
        # for at least as long as its prepared chunk can be retried.
        self._issued_references.append(reference)
        return self._key_prefix, reference

    def release_prepared(self) -> None:
        """Roll back an unpublished prepared handle after dispatch setup fails."""
        reference = self._prepared_reference
        if reference is None:
            return
        from verl.runtime import delete

        self._prepared_reference = None
        delete(reference)

    def __setstate__(self, state) -> None:
        self._issued_references = []
        self._prepared_reference = None
        self._key_prefix, self._reference = state
        self._table = None

    def get(self) -> RefTable:
        if self._table is not None:
            return self._table

        reference = self._reference
        if reference is None:
            raise RuntimeError("transported ref table has no unresolved reference")

        from verl.runtime import get

        resolved = get(reference)
        if not isinstance(resolved, RefTable):
            try:
                raise TypeError(f"transported ref table must resolve to RefTable, got {type(resolved)!r}")
            finally:
                self.release()

        self._table = resolved
        return resolved

    @classmethod
    def get_many(cls, transports: tuple[_RefTableTransport, ...]) -> list[RefTable]:
        """Fetch unresolved tables in one batch without re-reading cached ones."""
        pending = [transport for transport in transports if transport._table is None]
        if pending:
            from verl.runtime import get_many

            references = [transport._reference for transport in pending]
            if any(reference is None for reference in references):
                raise RuntimeError("transported ref table has no unresolved reference")
            values = get_many(references)
            try:
                if len(values) != len(pending) or any(not isinstance(value, RefTable) for value in values):
                    raise TypeError("transported ref table batch must resolve to RefTable values")
            except Exception as error:
                try:
                    _release_ref_table_transports(tuple(pending))
                except Exception as cleanup_error:
                    raise error from cleanup_error
                raise
            for transport, value in zip(pending, values, strict=True):
                transport._table = value
        return [transport.get() for transport in transports]

    def release(self) -> None:
        reference = self._reference
        if reference is None:
            return

        from verl.runtime import delete

        # Consume local ownership before remote cleanup. If delete reports an
        # ambiguous failure after deleting remotely, retrying this transport
        # cannot issue a second delete for the same reference.
        self._reference = None
        delete(reference)

    def resolve(self) -> RefTable:
        resolved = self.get()
        self.release()
        return resolved


def _resolve_transported_payload(payload: Any) -> RefTable:
    """Resolve one pickled ref-table handle through Runtime contracts.

    ``RefTable`` is already local. ``RemoteCall`` is observed through the RPC
    contract. Any other handle is an ObjectStore reference produced by the
    dispatch adapter.
    """
    if isinstance(payload, RefTable):
        return payload
    if isinstance(payload, _RefTableTransport):
        return payload.resolve()
    from verl.runtime import RemoteCall

    if isinstance(payload, RemoteCall):
        resolved = payload.result()
    else:
        from verl.runtime import delete, get

        resolved = get(payload)
        try:
            if not isinstance(resolved, RefTable):
                raise TypeError(f"transported ref table must resolve to RefTable, got {type(resolved)!r}")
        finally:
            delete(payload)
    if not isinstance(resolved, RefTable):
        raise TypeError(f"transported ref table must resolve to RefTable, got {type(resolved)!r}")
    return resolved


def _resolve_transported_ref_table(payload: Any) -> RefTable:
    """Restore a ref table shipped as local metadata or deferred store handles."""
    if not isinstance(payload, tuple):
        return _resolve_transported_payload(payload)
    if len(payload) != 2:
        raise TypeError(f"transported ref table tuple must have length 2, got {len(payload)}")
    from verl.runtime import RemoteCall

    if all(isinstance(item, RemoteCall) for item in payload):
        obj_table, local_table = RemoteCall.gather(list(payload)).result()
        if not isinstance(obj_table, RefTable) or not isinstance(local_table, RefTable):
            raise TypeError("transported ref table pair must resolve to RefTable values")
    elif all(isinstance(item, _RefTableTransport) for item in payload):
        obj_transport, local_transport = payload
        # Resolve the pair before consuming either reference. A transient
        # failure on the second get can then retry the same serialized
        # payload without finding that the first table was already deleted.
        obj_table, local_table = _RefTableTransport.get_many((obj_transport, local_transport))
        try:
            merged = RefTable(dict(obj_table.items()), batch_size=obj_table.batch_size)
            merged.update(local_table)
        except Exception as error:
            try:
                _release_ref_table_transports((obj_transport, local_transport))
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise
        _release_ref_table_transports((obj_transport, local_transport))
        return merged
    else:
        obj_table = _resolve_transported_payload(payload[0])
        local_table = _resolve_transported_payload(payload[1])
    merged = RefTable(dict(obj_table.items()), batch_size=obj_table.batch_size)
    merged.update(local_table)
    return merged


def _release_ref_table_transports(transports: tuple[_RefTableTransport, ...]) -> None:
    first_error = None
    for transport in transports:
        try:
            transport.release()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
