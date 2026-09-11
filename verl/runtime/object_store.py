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

"""Backend-neutral synchronous object-store capability."""

from __future__ import annotations

import atexit
from typing import Any, Protocol, TypeVar, cast

ReferenceT = TypeVar("ReferenceT")


class ObjectStore(Protocol[ReferenceT]):
    """Store explicitly keyed values behind backend-specific references."""

    def put(self, key: str, value: Any, /) -> ReferenceT:
        """Store one value and return its opaque reference.

        Args:
            key: Stable logical key chosen by the caller.
            value: Value to store.

        Returns:
            A store-specific reference for the stored value.
        """
        ...

    def get(self, reference: ReferenceT, /) -> Any:
        """Resolve one opaque reference to its value.

        Args:
            reference: Store-specific reference returned by put().

        Returns:
            The value identified by the reference.
        """
        ...

    def delete(self, reference: ReferenceT, /) -> None:
        """Notify the store that the caller will stop using one reference.

        Args:
            reference: Store-specific reference the caller will stop using.
        """
        ...

    def put_many(self, entries: list[tuple[str, Any]], /) -> list[ReferenceT]:
        """Store values in input order using the single-value contract."""
        self._validate_put_many_entries(entries)
        return [self.put(key, value) for key, value in entries]

    def get_many(self, references: list[ReferenceT], /) -> list[Any]:
        """Resolve references in input order using the single-value contract."""
        return [self.get(reference) for reference in references]

    def delete_many(self, references: list[ReferenceT], /) -> None:
        """Release references in input order using the single-value contract."""
        for reference in references:
            self.delete(reference)

    def close(self) -> None:
        """Drain pending cleanup and release process-local resources."""
        return None

    @staticmethod
    def _validate_put_many_entries(entries: list[tuple[str, Any]]) -> None:
        keys = []
        for key, _value in entries:
            if not isinstance(key, str) or not key:
                raise ValueError(f"key must be a non-empty str, got {key!r}")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("put_many keys must be unique")


_OBJECT_STORE: ObjectStore[Any] | None = None
_ATEXIT_REGISTERED = False


def _start_object_store(store: ObjectStore[Any]) -> None:
    """Start the process-global object store.

    Runtime construction is serialized per process. Replacing a live store
    would lose its pending cleanup, so callers must close the owning Runtime
    before installing another one.
    """
    global _ATEXIT_REGISTERED, _OBJECT_STORE
    if _OBJECT_STORE is not None:
        raise RuntimeError("ObjectStore is already started in this process")
    _OBJECT_STORE = store
    if not _ATEXIT_REGISTERED:
        # Register lazily, after backend modules have initialized their own
        # process-global resources. LIFO atexit ordering then closes the store
        # before those lower-level resources.
        atexit.register(_close_object_store)
        _ATEXIT_REGISTERED = True


def _close_object_store() -> None:
    """Close and remove the process-global client.

    A failed close keeps the store installed so the lifecycle owner can retry.
    """
    global _OBJECT_STORE
    if _OBJECT_STORE is not None:
        _OBJECT_STORE.close()
        _OBJECT_STORE = None


def _object_store() -> ObjectStore[Any]:
    """Return the process-global backend implementation."""
    if _OBJECT_STORE is None:
        raise RuntimeError("No ObjectStore is started in this process")
    return cast(ObjectStore[Any], _OBJECT_STORE)


def put(key: str, value: Any, /) -> Any:
    """Store one value and return its backend-specific opaque reference."""
    return _object_store().put(key, value)


def put_many(entries: list[tuple[str, Any]], /) -> list[Any]:
    """Store values in input order."""
    return _object_store().put_many(entries)


def get(reference: Any, /) -> Any:
    """Resolve one opaque ObjectStore reference."""
    return _object_store().get(reference)


def get_many(references: list[Any], /) -> list[Any]:
    """Resolve references in input order."""
    return _object_store().get_many(references)


def delete(reference: Any, /) -> None:
    """Release one ObjectStore reference."""
    _object_store().delete(reference)


def delete_many(references: list[Any], /) -> None:
    """Release ObjectStore references."""
    _object_store().delete_many(references)


__all__ = ["ObjectStore", "delete", "delete_many", "get", "get_many", "put", "put_many"]
