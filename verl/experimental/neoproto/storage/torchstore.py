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

"""Monarch-only TorchStore engine for the experimental NeoProto data plane.

TorchStore lifecycle and client installation belong to the Monarch Runtime.
This adapter only translates NeoProto refs to the process-local Runtime
``ObjectStore`` capability, keeping backend selection and storage ownership out
of the data container.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    Ref,
    SliceSpec,
    _BaseStorageEngine,
    _infer_dtype,
    _infer_shape,
    new_uid,
)
from verl.runtime.object_store import _object_store
from verl.single_controller.monarch.object_store import TensorRowRange, TorchStoreObjectStore, TorchStoreReference

__all__ = ["TorchStorageEngine"]


def _torchstore_object_store() -> TorchStoreObjectStore:
    store = _object_store()
    if not isinstance(store, TorchStoreObjectStore):
        raise TypeError(
            f"TorchStorageEngine requires the Runtime-started TorchStoreObjectStore, got {type(store).__name__}"
        )
    return store


class TorchStorageEngine(_BaseStorageEngine):
    """NeoProto storage adapter over Monarch's Runtime-started TorchStore client."""

    backend = "torchstore"

    def to_wire(self, value: Any) -> Any:
        return value

    def from_wire(self, value: Any) -> Any:
        return value

    def put(
        self,
        value: Any,
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> Ref:
        store = _torchstore_object_store()
        key = new_uid(prefix=f"{_key_prefix(key_hint)}-")
        reference = store.put(key, self.to_wire(value))
        return Ref(
            backend=self.backend,
            uid=key,
            dataptr=reference,
            dtype=_infer_dtype(value, spec),
            shape=_infer_shape(value, spec),
        )

    def put_many(
        self,
        values: Iterable[Any],
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> list[Ref]:
        values = list(values)
        if not values:
            return []
        store = _torchstore_object_store()

        keys = [new_uid(prefix=f"{_key_prefix(key_hint)}-") for _value in values]
        references = store.put_many([(key, self.to_wire(value)) for key, value in zip(keys, values, strict=True)])
        return [
            Ref(
                backend=self.backend,
                uid=key,
                dataptr=reference,
                dtype=_infer_dtype(value, spec),
                shape=_infer_shape(value, spec),
            )
            for key, reference, value in zip(keys, references, values, strict=True)
        ]

    def get(self, ref: Ref) -> Any:
        if ref.backend != self.backend:
            raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        return self.get_many([ref])[0]

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]:
        values: list[Any] = [None] * len(refs)
        remote_positions: list[int] = []
        remote_refs: list[Any] = []
        for index, ref in enumerate(refs):
            if ref is None:
                continue
            if ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
            remote_positions.append(index)
            remote_refs.append(ref.dataptr)

        if remote_refs:
            for reference in remote_refs:
                if not isinstance(reference, TorchStoreReference):
                    raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")
            row_ranges = _plan_row_ranges([refs[index] for index in remote_positions]) if apply_ops else {}
            store = _torchstore_object_store()
            remote_values = _read_snapshot_rows(store, remote_refs, row_ranges)
            for index, value in zip(remote_positions, remote_values, strict=True):
                ref = refs[index]
                value = self.from_wire(value)
                if apply_ops:
                    selection = row_ranges.get(ref.dataptr)
                    # Native OBJECT reads ignore tensor selections and return
                    # the original NumPy/list value, so their indices stay global.
                    spec = (
                        _shift_row_spec(ref, selection)
                        if selection is not None and isinstance(value, torch.Tensor)
                        else ref.slice_spec
                    )
                    value = self.apply_slice(value, spec)
                    value = ref.apply_ops(value)
                values[index] = value
        return values

    def get_rows_many(self, refs: list[Ref], rows: Any) -> list[Any]:
        """Resolve shared tensor columns using their actual logical row selection."""
        flattened = []
        selections = []
        for ref in refs:
            selected = _shared_row_refs(ref, rows)
            selections.append(selected)
            flattened.extend(selected if selected is not None else [ref])
        fetched = self.get_many(flattened)
        result = []
        offset = 0
        for selected in selections:
            if selected is None:
                from verl.experimental.neoproto.neo import NeoProto

                result.append(NeoProto._index_along(fetched[offset], 0, rows))
                offset += 1
            else:
                result.append(torch.cat(fetched[offset : offset + len(selected)], dim=0))
                offset += len(selected)
        return result

    def release(self, refs: Ref | list[Ref]) -> None:
        batch = [refs] if isinstance(refs, Ref) else list(refs)
        remote: dict[Any, Ref] = {}
        for ref in batch:
            if ref is None:
                continue
            if ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
            remote[ref.dataptr] = ref
        if remote:
            _torchstore_object_store().delete_many([ref.dataptr for ref in remote.values()])


def _read_snapshot_rows(
    store: TorchStoreObjectStore,
    references: list[TorchStoreReference],
    row_ranges: dict[TorchStoreReference, TensorRowRange],
) -> list[Any]:
    """Cache exact reads of Engine snapshots, whose UID is never overwritten.

    The Engine creates a fresh UID on every put. This contract does not extend
    to arbitrary raw-key overwrites from another process. Local raw writes and
    deletes still invalidate the shared process cache through the ObjectStore.
    """
    cache = store.local_cache
    if os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1":
        cache.clear()
        return store.get_many(references, row_ranges=row_ranges or None)
    if not cache.capacity_bytes:
        return store.get_many(references, row_ranges=row_ranges or None)
    unique = list(dict.fromkeys(references))
    eligible = {ref for ref in unique if ref.is_tensor and ref in row_ranges}
    # A legacy alias lacks wire-kind evidence even if an equal new ref has it.
    eligible.difference_update(ref for ref in references if not ref.is_tensor)
    generation = cache.generation()
    values = {}
    missing = []
    for ref in unique:
        value = cache.get(ref, row_ranges[ref]) if ref in eligible else None
        if value is None:
            missing.append(ref)
        else:
            values[ref] = value
    if missing:
        missing_ranges = {ref: row_ranges[ref] for ref in missing if ref in row_ranges}
        fetched = store.get_many(missing, row_ranges=missing_ranges or None)
        for ref, value in zip(missing, fetched, strict=True):
            if ref in eligible:
                cache.put(ref, row_ranges[ref], value, generation)
            values[ref] = value
    return [values[ref] for ref in references]


def _shared_row_refs(ref: Ref, rows: Any) -> list[Ref] | None:
    """Push row selection only through operations that preserve row meaning."""
    if (
        os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1"
        or not isinstance(ref.dataptr, TorchStoreReference)
        or not ref.dataptr.is_tensor
        or ref.apply_funcs
        or not ref.shape
        or ref.slice_spec is None
    ):
        return None
    first = ref.slice_spec[0]
    if not isinstance(first, slice) or first.step not in (None, 1):
        return None
    if len(rows) == 0 or any(
        not isinstance(dim, slice | int | np.integer) or isinstance(dim, bool | np.bool_) for dim in ref.slice_spec[1:]
    ):
        return None
    start, stop, _ = first.indices(ref.shape[0])
    count = max(0, stop - start)
    selected = []
    for row in rows:
        if not isinstance(row, int | np.integer) or isinstance(row, bool | np.bool_):
            return None
        row = int(row)
        if not 0 <= row < count:
            return None  # preserve the original indexing error at the value seam
        selected.append(ref.with_slice((slice(row, row + 1),)))
    return selected


def _row_bounds(ref: Ref) -> tuple[int, int] | None:
    shape = ref.shape
    if not shape or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
        return None
    first = ref.slice_spec[0] if ref.slice_spec else None
    if first is None:
        return 0, shape[0]
    if isinstance(first, bool | np.bool_):
        return None
    if isinstance(first, int | np.integer):
        index = int(first)
        if index < 0:
            index += shape[0]
        return (index, index + 1) if 0 <= index < shape[0] else None
    if isinstance(first, slice) and first.step in (None, 1):
        start, stop, _ = first.indices(shape[0])
        return start, max(start, stop)
    return None


def _plan_row_ranges(refs: list[Ref]) -> dict[TorchStoreReference, TensorRowRange]:
    grouped: dict[TorchStoreReference, list[Ref]] = {}
    for ref in refs:
        grouped.setdefault(ref.dataptr, []).append(ref)
    result = {}
    for reference, group in grouped.items():
        # Shape alone does not identify the native wire kind: a shaped object
        # can even deserialize to Tensor. Old references conservatively read full.
        if not all(ref.dataptr.is_tensor for ref in group):
            continue
        shape = group[0].shape
        bounds = [_row_bounds(ref) for ref in group]
        if shape is None or any(bound is None for bound in bounds) or any(ref.shape != shape for ref in group):
            continue
        valid_bounds = [bound for bound in bounds if bound is not None]
        start = min(bound[0] for bound in valid_bounds)
        stop = max(bound[1] for bound in valid_bounds)
        merged: list[tuple[int, int]] = []
        for left, right in sorted(valid_bounds):
            if left == right:
                continue
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        if os.environ.get("TORCHSTORE_MUTABLE_SHM", "0") == "1":
            # Native mutable SHM reads expose the producer storage. Segment
            # assembly would silently replace that writable view with a copy.
            merged = []
        if start != 0 or stop != shape[0] or len(merged) > 1:
            result[reference] = TensorRowRange(tuple(shape), start, stop, tuple(merged) if len(merged) > 1 else ())
    return result


def _shift_row_spec(ref: Ref, selection: TensorRowRange) -> SliceSpec:
    assert ref.slice_spec is not None
    first = ref.slice_spec[0]
    if isinstance(first, int | np.integer):
        index = int(first)
        if index < 0:
            index += selection.shape[0]
        first = index - selection.start
    elif isinstance(first, slice):
        start, stop, step = first.indices(selection.shape[0])
        first = slice(start - selection.start, max(start, stop) - selection.start, step)
    else:
        raise AssertionError("a partial row read requires a basic first-axis selector")
    return (first, *ref.slice_spec[1:])


def _key_prefix(key_hint: str | None) -> str:
    if not key_hint:
        return "neo"
    normalized = "".join(character if character.isalnum() or character in "_.-" else "_" for character in key_hint)
    return normalized[:64] or "neo"
