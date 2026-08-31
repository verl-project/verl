# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Storage engine abstractions for :class:`NeoProto`.

This module defines the low-level primitives:

- :class:`Ref` -- opaque handle that survives Ray transport / pickling.
- :class:`RefTable` -- per-key mapping of refs (shared, per-sample, or per-token).
- :class:`StorageEngine` -- protocol with ``put / get / apply / release``.

Concrete implementations live in :mod:`verl.experimental.neoproto.storage.default`.
"""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

try:  # pragma: no cover - optional dep
    import torch
    import torch.nn.functional as F

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Slice specification helpers
# ---------------------------------------------------------------------------

# A ``SliceSpec`` is a tuple of per-dim selectors that maps directly to
# ``numpy`` / ``torch`` advanced indexing. Each entry may be:
#
#   * None                 -- ``slice(None)`` (take everything on this axis)
#   * int                  -- pick a single index
#   * slice                -- a standard Python slice
#   * List[int] / ndarray  -- fancy indexing along this axis
#
# The first axis is, by convention, the batch axis; the second axis is the
# token axis for token-level indexing.
SliceDim = int | slice | list[int] | tuple[int, ...] | np.ndarray | None
SliceSpec = tuple[SliceDim, ...]


def compose_slice(outer: SliceSpec, inner: SliceSpec) -> SliceSpec:
    """Compose two ``SliceSpec`` by logical application: ``outer`` first, then ``inner``.

    Only handles the common cases we need in the driver (int, slice, list, None);
    for richer composition we just materialize at the worker.
    """
    if not outer:
        return inner
    if not inner:
        return outer

    out: list[SliceDim] = []
    oi = 0
    for dim in outer:
        if oi >= len(inner):
            out.append(dim)
            continue
        # an integer selector collapses the axis, so no inner selector applies
        if isinstance(dim, int | np.integer):
            out.append(dim)
            continue
        inner_dim = inner[oi]
        oi += 1
        out.append(_compose_axis(dim, inner_dim))
    # any remaining inner dims apply to axes beyond outer
    out.extend(inner[oi:])
    return tuple(out)


def _compose_axis(outer: SliceDim, inner: SliceDim) -> SliceDim:
    if outer is None:
        return inner
    if inner is None:
        return outer
    if isinstance(outer, slice) and isinstance(inner, slice):
        o_start, o_stop, o_step = outer.start or 0, outer.stop, outer.step or 1
        i_start, i_stop, i_step = inner.start or 0, inner.stop, inner.step or 1
        new_start = o_start + i_start * o_step
        new_step = o_step * i_step
        if o_stop is None and i_stop is None:
            new_stop = None
        elif o_stop is None:
            new_stop = o_start + i_stop * o_step
        elif i_stop is None:
            new_stop = o_stop
        else:
            new_stop = min(o_stop, o_start + i_stop * o_step)
        return slice(new_start, new_stop, new_step)
    elif isinstance(outer, slice) and isinstance(inner, int | np.integer):
        return inner
    elif isinstance(outer, int | np.integer) and isinstance(inner, slice):
        return outer
    elif isinstance(outer, int | np.integer) and isinstance(inner, int | np.integer):
        assert outer == inner
        return outer
    # fall back: defer to the Materializer which applies them sequentially
    return (outer, inner)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _infer_dtype(value: Any, spec: Optional[FieldSpec]) -> Optional[str]:
    if spec is not None and spec.dtype not in (None, "object"):
        return spec.dtype
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    if isinstance(value, np.ndarray):
        return str(value.dtype)
    if isinstance(value, bytes | bytearray):
        return "bytes"
    if isinstance(value, int | float | bool):
        return type(value).__name__
    return "object"


def _infer_shape(value: Any, spec: Optional[FieldSpec]):
    if spec is not None and spec.shape:
        return tuple(spec.shape)
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    return None


# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Schema for a single key in a :class:`NeoProto`.

    Attributes
    ----------
    dtype:
        Canonical dtype string ("int64", "float32", "object", "bytes", ...).
    shape:
        Full shape of a single sample's value, e.g. ``(seq_len,)`` for
        ``input_ids`` or ``()`` for scalar per-sample values. ``None`` dims
        mean "variable length" (e.g. response length).
    granularity:
        ``"sample"`` -- one ref per sample.
        ``"shared"`` -- a single ref covers the entire batch.
        ``"shards"`` -- one ref per sample shards.
        ``"slice"``  -- one ref per token slice (rarely used; typically we keep
                       per-sample refs and select with a token ``SliceSpec``).
    token_axis:
        Index of the axis that represents the token dimension, or ``None`` if
        the field has no token dimension.
    """

    dtype: str = "object"
    shape: tuple[Optional[int], ...] = ()
    granularity: str = "sample"
    token_axis: Optional[int] = None

    def with_shape(self, shape: tuple[Optional[int], ...]) -> FieldSpec:
        return replace(self, shape=shape)


# ---------------------------------------------------------------------------
# Ref
# ---------------------------------------------------------------------------

FUNC_IMPLS = {
    "PAD": lambda x, pad, value: F.pad(x, pad, value=value)
    if torch.is_tensor(x)
    else F.pad(torch.from_numpy(x), pad, value=value),
    "UNSQUEEZE": lambda x, dim: x.unsqueeze(dim),
    "SQUEEZE": lambda x, dim: x.squeeze(dim),
    "TO": lambda x, kwargs: x.to(**kwargs),
}


@dataclass
class Ref:
    """Opaque handle to a value stored in some :class:`StorageEngine`.

    A ``Ref`` is *only* metadata; its payload lives in the storage engine.
    Refs are picklable (survive Ray transport) and hashable by ``uid``.

    Attributes
    ----------
    backend:
        Identifier of the backend (for example, "object_store" or "local").
    uid:
        Stable ID assigned by the backend (e.g. Ray object hex, local uuid).
    dtype / shape:
        Type info; used for validation and for the driver to reason about sizes
        without materializing the tensor.
    slice_spec:
        Optional :class:`SliceSpec` describing a view into the stored tensor.
        Compose driver-side via :func:`compose_slice`.
    """

    backend: str
    uid: str
    # Raw backend handle (Ray ObjectRef, in-memory value, etc.). Only the
    # storage engine interprets this; the driver treats it as opaque.
    dataptr: Any
    dtype: Optional[str] = None
    shape: Optional[tuple[Optional[int], ...]] = None
    slice_spec: Optional[SliceSpec] = None
    apply_funcs: list[Any] = field(default_factory=list)

    def __new__(cls, *args, **kwargs):
        """Factory dispatch: constructing the base :class:`Ref` returns the
        backend-specific subclass (:class:`LocalRef` for ``backend="local"``,
        :class:`StorageRef` otherwise). Constructing a subclass keeps its type.
        """
        if cls is Ref:
            backend = kwargs.get("backend")
            if backend is None and args:
                backend = args[0]
            cls = LocalRef if backend == "local" else StorageRef
        return object.__new__(cls)

    def __init__(
        self,
        backend: str,
        uid: str,
        dataptr: Any,
        dtype: Optional[str] = None,
        shape: Optional[tuple[Optional[int], ...]] = None,
        slice_spec: Optional[SliceSpec] = None,
        apply_funcs: list[Any] = None,
    ):
        self.backend = backend
        self.uid = uid
        self.dataptr = dataptr
        self.dtype = dtype
        self.shape = shape
        if slice_spec is None and shape is not None:
            self.slice_spec = tuple([slice(0, length) for length in shape])
        else:
            self.slice_spec = slice_spec
        if apply_funcs is None:
            apply_funcs = []
        self.apply_funcs = apply_funcs

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------
    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash((self.backend, self.uid, repr(self.slice_spec)))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.shape[0] if self.shape else 1

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, Ref):
            return False
        return self.backend == other.backend and self.uid == other.uid

    def __deepcopy__(self, memo: dict[Any, Any]) -> Ref:
        return Ref.from_ref(self)

    def __getitem__(self, item: Any) -> Ref:
        if self.slice_spec is None:
            return self
        if isinstance(item, slice):
            return self.slice(start=item.start, stop=item.stop, step=item.step)
        else:
            return self.with_sample(item)

    # ------------------------------------------------------------------
    # Picklability (explicit so we're robust across Ray transport)
    # ------------------------------------------------------------------
    def __reduce__(self):  # pragma: no cover - trivial
        return (
            _rebuild_ref,
            (self.backend, self.uid, self.dataptr, self.dtype, self.shape, self.slice_spec, self.apply_funcs),
        )

    # ------------------------------------------------------------------
    # Slice composition
    # ------------------------------------------------------------------
    def with_slice(self, slice_spec: SliceSpec) -> Ref:
        """Return a new ``Ref`` whose ``slice_spec`` is composed with ``slice_spec``."""
        if not slice_spec:
            return self
        new_spec = compose_slice(self.slice_spec or (), slice_spec)
        return replace(self, slice_spec=new_spec)

    def with_sample(self, idx: int) -> Ref:
        """Return a new ``Ref`` that picks a single sample (first axis)."""
        return self.with_slice((idx,))

    def with_token_slice(self, start: int, stop: Optional[int], step: int = 1, token_axis: int = 1) -> Ref:
        """Return a new ``Ref`` that restricts the token axis to ``[start, stop)``."""
        dims: list[SliceDim] = [None] * (token_axis + 1)
        dims[token_axis] = slice(start, stop, step)
        return self.with_slice(tuple(dims))

    def slice(self, start: int, stop: Optional[int], step: int = 1) -> Ref:
        """Return a new ``Ref`` that restricts the view to ``slice_spec``."""
        token_axis = 0
        if len(self.shape) > 1:
            token_axis = 1
        return self.with_token_slice(start, stop, step, token_axis=token_axis)

    def copy(self) -> Ref:
        return Ref.from_ref(self)

    @staticmethod
    def from_ref(ref: Ref) -> Ref:
        new_ref = Ref(
            backend=ref.backend,
            uid=ref.uid,
            dataptr=ref.dataptr,
            dtype=ref.dtype,
            shape=ref.shape,
            slice_spec=copy.deepcopy(ref.slice_spec),
            apply_funcs=ref.apply_funcs,
        )
        return new_ref

    def to(self, dtype=None, device=None) -> Ref:
        """Return a new ``Ref`` that points to the same tensor on ``device``."""
        if dtype is not None and str(dtype) != self.dtype:
            self.dtype = str(dtype)
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        self.apply_funcs.append(("TO", kwargs))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "uid": self.uid,
            "dataptr": self.dataptr,
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
            "slice_spec": _slice_spec_to_primitive(self.slice_spec),
        }

    def is_local(self) -> bool:
        return self.backend == "local"

    def release(self) -> None:
        from verl.experimental.neoproto.storage.default import get_engine_for_backend

        get_engine_for_backend(self.backend).release(self)

    def apply_ops(self, value: Any) -> Any:
        for func in self.apply_funcs:
            func_name = func[0]
            func_args = func[1:]
            func_impl = FUNC_IMPLS[func_name]
            value = func_impl(value, *func_args)
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ref:
        shape = data.get("shape")
        if shape is not None:
            shape = tuple(shape)
        return cls(
            backend=data["backend"],
            uid=data["uid"],
            dataptr=data["dataptr"],
            dtype=data.get("dtype"),
            shape=shape,
            slice_spec=_slice_spec_from_primitive(data.get("slice_spec")),
        )

    @classmethod
    def from_local(cls, value: Any) -> LocalRef:
        return LocalRef.of(value)

    # ------------------------------------------------------------------
    # Apply torch functions
    # ------------------------------------------------------------------
    def squeeze(self, dim: int = 0) -> Ref:
        self.apply_funcs.append(("SQUEEZE", dim))
        return self

    def unsqueeze(self, dim: int = 0) -> Ref:
        self.apply_funcs.append(("UNSQUEEZE", dim))
        return self

    def pad(self, pad: Sequence[int], value: Optional[float] = None) -> Ref:
        self.apply_funcs.append(("PAD", pad, value))
        return self

    @property
    def estimated_shape(self) -> Optional[tuple[int, ...]]:
        if self.slice_spec is None:
            return self.shape
        else:
            shape = []
            for i in range(len(self.shape)):
                if isinstance(self.slice_spec[i], slice):
                    step = self.slice_spec[i].step if self.slice_spec[i].step is not None else 1
                    slice_n = (self.slice_spec[i].stop - self.slice_spec[i].start) // step
                    shape.append(min(slice_n, self.shape[i]))
                else:
                    shape.append(1)
            return tuple(shape)


class StorageRef(Ref):
    """A :class:`Ref` whose payload lives in an external storage backend.

    ``backend`` is the engine-specific marker and ``dataptr`` is an opaque
    backend handle (for example, a Ray ``ObjectRef``); the value is fetched
    through a :class:`StorageEngine`.
    """

    pass


class LocalRef(Ref):
    """A :class:`Ref` whose value is held inline in ``dataptr`` (``backend="local"``).

    No storage-engine round-trip is needed to read it back; an engine that sees
    a local ref returns ``dataptr`` directly (after any ``slice_spec`` / ops).
    """

    # def __getitem__(self, item: Any) -> LocalRef:
    #     ref = self.copy()
    #     ref.dataptr = ref.dataptr[item]
    #     return ref

    def __add__(self, other: Any) -> Any:
        assert isinstance(other, LocalRef), f"Can only add LocalRefs, got {type(other)}"
        ref = self.copy()
        ref.dataptr = self.dataptr + other.dataptr
        return ref

    @classmethod
    def of(cls, value: Any, *, dtype: str = "object", shape: tuple[int, ...] = ()) -> LocalRef:
        """Wrap an in-memory ``value`` as a ``backend="local"`` ref."""
        return cls(backend="local", uid=new_uid(), dataptr=value, dtype=dtype, shape=shape)


def _rebuild_ref(backend, uid, dataptr, dtype, shape, slice_spec, apply_funcs):  # pragma: no cover
    return Ref(
        backend=backend,
        uid=uid,
        dataptr=dataptr,
        dtype=dtype,
        shape=shape,
        slice_spec=slice_spec,
        apply_funcs=apply_funcs,
    )


def _slice_spec_to_primitive(spec: Optional[SliceSpec]) -> Any:
    if spec is None:
        return None
    out = []
    for d in spec:
        if d is None:
            out.append({"type": "all"})
        elif isinstance(d, slice):
            out.append({"type": "slice", "start": d.start, "stop": d.stop, "step": d.step})
        elif isinstance(d, list | tuple):
            out.append({"type": "list", "values": list(d)})
        elif isinstance(d, np.ndarray):
            out.append({"type": "list", "values": d.tolist()})
        else:
            out.append({"type": "int", "value": int(d)})
    return out


def _slice_spec_from_primitive(raw: Any) -> Optional[SliceSpec]:
    if raw is None:
        return None
    out: list[SliceDim] = [d for d in raw]
    return tuple(out)


# ---------------------------------------------------------------------------
# RefTable
# ---------------------------------------------------------------------------


RefContainer = Ref | np.ndarray | list["RefContainer"]


def _release_ref_container(container: RefContainer) -> None:
    """Release every :class:`Ref` held by a (possibly nested) ref container."""
    from verl.experimental.neoproto.storage.default import get_engine_for_backend

    if isinstance(container, Ref):
        container.release()
    elif isinstance(container, list | np.ndarray):
        ref_list = [r for r in container if isinstance(r, Ref)]
        if len(ref_list) > 0:
            get_engine_for_backend(ref_list[0].backend).release(ref_list)


# ---------------------------------------------------------------------------
# Columnar serialization helpers for RefTable
# ---------------------------------------------------------------------------
# A per-sample column is an ``ndarray[object]`` (or list) holding one ``Ref`` per
# sample. Pickling it element-by-element pays the full per-object cost N times
# (dataclass reduce + a ``slice_spec`` tuple + a uid string, each). When every
# ref in the column shares its metadata we store that metadata once and bulk the
#  only per-sample fields (uid, dataptr) into lists so that pickle walks 1 header
#  + 2 lists instead of N objects.


def _pack_ref_column(v: RefContainer):
    if not isinstance(v, np.ndarray | list):
        return ("raw", v)  # single shared Ref (or anything exotic): keep as-is
    refs = list(v)
    non_null = [r for r in refs if r is not None]
    if not non_null or not all(isinstance(r, Ref) for r in non_null):
        return ("raw", v)
    r0 = non_null[0]
    homogeneous = all(
        r.backend == r0.backend
        and r.dtype == r0.dtype
        and r.shape == r0.shape
        and r.slice_spec == r0.slice_spec
        and r.apply_funcs == r0.apply_funcs
        for r in non_null
    )
    if not homogeneous:
        return ("raw", v)
    uids = [None if r is None else r.uid for r in refs]  # plain list of str (None marks holes)
    dataptrs = [None if r is None else r.dataptr for r in refs]  # opaque handles stay a list
    return (
        "col",
        isinstance(v, np.ndarray),
        r0.backend,
        r0.dtype,
        r0.shape,
        r0.slice_spec,
        r0.apply_funcs,
        uids,
        dataptrs,
    )


def _unpack_ref_column(p):
    if p[0] == "raw":
        return p[1]
    _, was_ndarray, backend, dtype, shape, slice_spec, apply_funcs, uids, dataptrs = p
    out: list[Any] = []
    for i in range(len(uids)):
        if uids[i] is None:
            out.append(None)
            continue
        out.append(
            Ref(
                backend=backend,
                uid=uids[i],
                dataptr=dataptrs[i],
                dtype=dtype,
                shape=shape,
                slice_spec=slice_spec,
                apply_funcs=list(apply_funcs),  # per-ref copy: apply_funcs is mutated in place
            )
        )
    if was_ndarray:
        arr = np.empty(len(out), dtype=object)
        arr[:] = out
        return arr
    return out


class RefTable:
    """A per-key mapping from a field name to either a shared ``Ref`` or an
    ``ndarray[object]`` of ``Ref`` (one per sample).

    Only metadata is stored here; payload lives in the storage engine. The
    table is pickle-stable across Ray transport.
    """

    __slots__ = ("_refs", "_n", "_gc_queue")

    def __init__(self, refs: Optional[dict[str, RefContainer]] = None, *, batch_size: int = 0):
        self._refs: dict[str, RefContainer] = dict(refs or {})
        self._n: int = batch_size
        self._gc_queue: list[RefContainer] = []

    # ------------------------------------------------------------------
    # Basic mapping protocol
    # ------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._refs

    def __getitem__(self, key: str) -> RefContainer:
        return self._refs[key]

    def __setitem__(self, key: str, value: RefContainer) -> None:
        if key in self._refs:
            old_ref = self._refs[key]
            if isinstance(old_ref, Ref):
                if old_ref.uid != getattr(value, "uid", None):
                    if old_ref.backend != "local":
                        self._gc_queue.append(old_ref)
            elif isinstance(old_ref, list | np.ndarray):
                # per-sample column of Refs: retire the non-local ones for deferred GC
                non_local = [r for r in old_ref if isinstance(r, Ref) and r.backend != "local"]
                if non_local:
                    self._gc_queue.append(non_local)
        self._refs[key] = value

    def set_refs(self, key: str, value: RefContainer) -> None:
        self._refs[key] = value

    def __iter__(self):
        return iter(self._refs)

    def __copy__(self):
        return self.copy()

    def __del__(self):
        # Do not force-free overwritten Ray ObjectRefs here. RefTables are
        # copied across chunk views and processes, so another live table may
        # still reference the same object. Dropping this queue lets Ray's
        # distributed reference counting reclaim objects safely. Explicit
        # ``NeoProto.release()`` remains available for owned data.
        self._gc_queue.clear()

    # ------------------------------------------------------------------
    # Columnar pickling. Avoids per-Ref serialization for homogeneous per-sample
    # columns; ``_gc_queue`` is transport-irrelevant and dropped. Falls back to
    # per-object pickling for non-homogeneous columns.
    # ------------------------------------------------------------------
    def __getstate__(self):
        return {"cols": {k: _pack_ref_column(v) for k, v in self._refs.items()}, "n": self._n}

    def __setstate__(self, state):
        # slots class: __init__ is NOT called on unpickle, so set every slot here.
        self._refs = {k: _unpack_ref_column(p) for k, p in state["cols"].items()}
        self._n = state["n"]
        self._gc_queue = []

    def keys(self):
        return self._refs.keys()

    def values(self):
        return self._refs.values()

    def items(self):
        return self._refs.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._refs.get(key, default)

    def pop(self, key: str, default: Any = ...) -> RefContainer:
        if default is ...:
            return self._refs.pop(key)
        return self._refs.pop(key, default)

    def select(self, idx: int) -> RefTable:
        new_refs = {}
        for k, v in self._refs.items():
            if isinstance(v, Ref):
                assert v.backend == "local" or idx == 0, f"Ref {v} backend {v.backend} not supported"
                new_refs[k] = v.copy()
            elif isinstance(v, np.ndarray):
                new_refs[k] = v[idx]
            elif isinstance(v, list):
                new_refs[k] = v[idx]
            else:  # pragma: no cover
                new_refs[k] = v
        return RefTable(new_refs, batch_size=1)

    def copy(self) -> RefTable:
        new_refs = {}
        for k, v in self._refs.items():
            if isinstance(v, Ref):
                new_refs[k] = v.copy()
            elif isinstance(v, np.ndarray):
                new_refs[k] = v.view()
            elif isinstance(v, list):
                new_refs[k] = [r for r in v]
            else:  # pragma: no cover
                new_refs[k] = v
        return RefTable(new_refs, batch_size=self._n)

    def update(self, other: RefTable, value_keys: set[str] = None, check_consistency: bool = True) -> None:
        if self.batch_size > 0:
            if check_consistency:
                assert other.batch_size == self.batch_size, (
                    f"Batch size mismatch: {other.batch_size} != {self.batch_size}"
                )
        else:
            self._n = other.batch_size
        for k, v in other._refs.items():
            if value_keys is not None and k not in value_keys:
                continue
            self._refs[k] = v

    @property
    def batch_size(self) -> int:
        return self._n

    @batch_size.setter
    def batch_size(self, n: int) -> None:
        self._n = n

    # ------------------------------------------------------------------
    # Per-sample view helpers
    # ------------------------------------------------------------------
    def sample(self, idx: int) -> dict[str, Any]:
        """Return a view for a single sample ``idx`` (ref per key)."""
        out: dict[str, Any] = {}
        for k, v in self._refs.items():
            if isinstance(v, Ref):
                out[k] = v.with_sample(idx)
            elif isinstance(v, np.ndarray):
                out[k] = v[idx]
            elif isinstance(v, list):
                out[k] = v[idx]
            else:  # pragma: no cover
                out[k] = v
        return out

    def take(self, indices: Iterable[int]) -> RefTable:
        """Return a new ``RefTable`` restricted to ``indices``."""
        idx = np.asarray(list(indices), dtype=np.int64)
        new: dict[str, RefContainer] = {}
        for k, v in self._refs.items():
            if isinstance(v, np.ndarray):
                new[k] = v[idx]
            elif isinstance(v, list):
                new[k] = [v[i] for i in idx]
            elif isinstance(v, Ref):
                # Shared ref -- keep as-is but remember we sliced the sample axis
                new[k] = v.with_slice((idx.tolist(),))
            else:  # pragma: no cover
                new[k] = v
        return RefTable(new, batch_size=len(idx))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._refs.items():
            if isinstance(v, Ref):
                out[k] = {"kind": "shared", "ref": v.to_dict()}
            elif isinstance(v, np.ndarray):
                out[k] = {
                    "kind": "per_sample",
                    "refs": [r.to_dict() if isinstance(r, Ref) else None for r in v.tolist()],
                }
            elif isinstance(v, list):
                out[k] = {
                    "kind": "per_sample",
                    "refs": [r.to_dict() if isinstance(r, Ref) else None for r in v],
                }
            else:  # pragma: no cover
                raise TypeError(f"Unsupported ref container for key {k}: {type(v)}")
        return {"batch_size": self._n, "entries": out}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefTable:
        refs: dict[str, RefContainer] = {}
        for k, entry in data.get("entries", {}).items():
            if entry["kind"] == "shared":
                refs[k] = Ref.from_dict(entry["ref"])
            else:
                arr = np.empty(len(entry["refs"]), dtype=object)
                for i, r in enumerate(entry["refs"]):
                    arr[i] = Ref.from_dict(r) if r is not None else None
                refs[k] = arr
        return cls(refs, batch_size=data.get("batch_size", 0))


# ---------------------------------------------------------------------------
# StorageEngine protocol
# ---------------------------------------------------------------------------


class StorageEngine(Protocol):
    """Pluggable storage engine used by :class:`NeoProto`.

    Implementations MUST provide at least ``put / get / release``; ``apply`` has
    a sensible default (get -> fn -> put) supplied by :class:`_BaseStorageEngine`.
    """

    def from_wire(self, value: Any) -> Any: ...

    def to_wire(self, value: Any) -> Any: ...

    def put(self, value: Any, *, key_hint: Optional[str] = None, spec: Optional[FieldSpec] = None) -> Ref: ...

    def put_many(
        self, values: Iterable[Any], *, key_hint: Optional[str] = None, spec: Optional[FieldSpec] = None
    ) -> list[Ref]: ...

    def get(self, ref: Ref) -> Any: ...

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]: ...

    def consolidate(
        self,
        old_engine,
        refs: list[Ref],
    ) -> Ref: ...

    def release(self, refs: Ref | list[Ref]) -> None: ...

    def apply(
        self, ref: Ref, fn: Callable[[Any], Any], *, key_hint: Optional[str] = None, spec: Optional[FieldSpec] = None
    ) -> Ref: ...


# ---------------------------------------------------------------------------
# Base with defaults for ``apply`` and slicing
# ---------------------------------------------------------------------------


class _BaseStorageEngine:
    """Mixin providing default implementations for the optional methods of
    :class:`StorageEngine` and convenience helpers for sliced reads."""

    # each concrete engine must set this to the backend string stored in Ref
    backend: str = "base"

    def apply(
        self,
        ref: Ref,
        fn: Callable[[Any], Any],
        *,
        key_hint: Optional[str] = None,
        spec: Optional[FieldSpec] = None,
    ) -> Ref:
        value = self.get(ref)
        new_value = fn(value)
        return self.put(new_value, key_hint=key_hint, spec=spec)

    def put_many(
        self,
        values: Iterable[Any],
        *,
        key_hint: Optional[str] = None,
        spec: Optional[FieldSpec] = None,
    ) -> list[Ref]:
        return [self.put(v, key_hint=key_hint, spec=spec) for v in values]

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]:
        return [self.get(r) for r in refs]

    def consolidate(
        self,
        old_engine,
        refs: list[Ref],
    ) -> Ref:
        """Materialize a ref column once and store it as one batch ref."""
        raise NotImplementedError("consolidate not implemented on this engine")

    def release(self, refs: Ref | list[Ref]) -> None:
        # default no-op; Ray-backed engines override this
        return None

    # ------------------------------------------------------------------
    # Slice application (applied at materialize time)
    # ------------------------------------------------------------------
    @staticmethod
    def apply_slice(value: Any, spec: Optional[SliceSpec]) -> Any:
        if not spec:
            return value
        # Handle composed specs (tuples of specs) recursively
        pieces: list[SliceSpec] = []
        for dim in spec:
            if isinstance(dim, tuple) and not all(_is_scalar_dim(d) for d in dim):
                # nested composition: apply each then combine
                raise NotImplementedError("Nested SliceSpec composition not yet materialized on this path")
            pieces.append(dim)
        idx = tuple(_to_index(d) for d in pieces)
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            return value[idx]
        if isinstance(value, np.ndarray):
            return value[idx]
        # Fall back: treat as a list-of-first-axis
        first = idx[0] if idx else slice(None)
        if isinstance(first, int | np.integer):
            return value[int(first)]
        return [value[i] for i in _iterate_axis(first, len(value))]


def _is_scalar_dim(d: Any) -> bool:
    return d is None or isinstance(d, int | slice | list | tuple | np.ndarray | np.integer)


def _to_index(d: SliceDim) -> Any:
    if d is None:
        return slice(None)
    if isinstance(d, list | tuple):
        return list(d)
    return d


def _iterate_axis(sel: Any, length: int):
    if isinstance(sel, slice):
        return range(*sel.indices(length))
    if isinstance(sel, list | tuple | np.ndarray):
        return list(sel)
    raise TypeError(f"Unsupported axis selector: {sel!r}")


# ---------------------------------------------------------------------------
# UID generator used by local backends
# ---------------------------------------------------------------------------


_uid_lock = threading.Lock()


def new_uid(prefix: str = "") -> str:
    with _uid_lock:
        u = uuid.uuid4().hex
    return f"{prefix}{u}" if prefix else u
