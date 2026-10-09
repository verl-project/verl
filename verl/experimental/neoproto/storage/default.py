# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Default storage backends for :class:`NeoProto`.

This module provides two concrete implementations of :class:`StorageEngine`:

- :class:`InMemoryStorageEngine` -- process-local test adapter; no Runtime
  dependency and not used as a production fallback.
- :class:`RayStorageEngine` -- stores payloads through :mod:`verl.runtime`
  (tensors are converted to numpy on the wire via
  :meth:`RayStorageEngine.to_wire`).

The active Runtime backend selects the process default explicitly: Ray uses
:class:`RayStorageEngine`; Monarch uses the experimental TorchStore adapter.
Without a Runtime, the default Runtime backend selects it instead.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - optional dep
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    LocalRef,
    Ref,
    StorageEngine,
    _BaseStorageEngine,
    _infer_dtype,
    _infer_shape,
    new_uid,
)
from verl.runtime.config import DEFAULT_BACKEND
from verl.runtime.object_store import ObjectStore, _object_store

# ---------------------------------------------------------------------------
# Data-plane I/O throughput accounting (gated by NEO_IO_STATS=1).
# ---------------------------------------------------------------------------
_IO_ON = os.environ.get("NEO_IO_STATS", "0") == "1"
_IO_STATS = {
    "put_bytes": 0,
    "put_s": 0.0,
    "put_n": 0,
    "get_bytes": 0,
    "get_s": 0.0,
    "get_n": 0,
}


def io_stats_snapshot() -> dict[str, float]:
    return dict(_IO_STATS)


def io_stats_reset() -> None:
    _IO_STATS.update({"put_bytes": 0, "put_s": 0.0, "put_n": 0, "get_bytes": 0, "get_s": 0.0, "get_n": 0})


# ---------------------------------------------------------------------------
# In-memory (process-local) backend -- used by tests and offline debug
# ---------------------------------------------------------------------------


class InMemoryStorageEngine(_BaseStorageEngine):
    """Trivial storage backed by a process-local dict.

    Not thread-safe across processes; used only when tests explicitly inject
    it as the default engine.
    """

    backend = "memory"

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._refcount: dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def put(
        self,
        value: Any,
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> Ref:
        del key_hint
        uid = new_uid()
        with self._lock:
            self._store[uid] = value
            self._refcount[uid] = 1
        return Ref(
            backend=self.backend,
            uid=uid,
            dataptr=value,
            dtype=_infer_dtype(value, spec),
            shape=_infer_shape(value, spec),
        )

    def get(self, ref: Ref) -> Any:
        if ref.backend != self.backend:
            raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        if ref.dataptr is not None:
            value = ref.dataptr
        else:
            value = self._store[ref.uid]
        value = self.apply_slice(value, ref.slice_spec)
        return ref.apply_ops(value)

    def release(self, refs: Ref | list[Ref]) -> None:
        if isinstance(refs, Ref):
            refs = [refs]
        with self._lock:
            for r in refs:
                if r is None:
                    continue
                if r.backend != self.backend:
                    raise ValueError(f"Ref belongs to backend {r.backend!r}, not {self.backend!r}")
                c = self._refcount.get(r.uid, 0) - 1
                if c <= 0:
                    self._store.pop(r.uid, None)
                    self._refcount.pop(r.uid, None)
                else:
                    self._refcount[r.uid] = c

    def add_ref_counts(self, refs: list[Ref]) -> None:
        with self._lock:
            for r in refs:
                if r is None:
                    continue
                self._refcount[r.uid] = self._refcount.get(r.uid, 0) + 1


# ---------------------------------------------------------------------------
# Ray-backed default
# ---------------------------------------------------------------------------


def _is_tensor_like(value: Any) -> bool:
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        return True
    if isinstance(value, np.ndarray) and value.dtype != object:
        return True
    return False


# ---------------------------------------------------------------------------
# torch.Tensor <-> numpy wire format
#
# ``torch.Tensor`` payloads are stored in Ray as plain ``numpy`` arrays: Ray
# serializes numpy via Arrow with out-of-band (zero-copy) buffers, which is
# markedly cheaper to ``ray.get`` than pickling a ``torch.Tensor``. The original
# tensor is reconstructed on the read path so every caller above the storage
# engine still sees a ``torch.Tensor``.
# ---------------------------------------------------------------------------

# torch dtypes that numpy cannot represent (bf16, fp8, ...) are bit-reinterpreted
# through a same-width signed integer type before storage and restored on read.
_REINTERPRET_INT = {1: "int8", 2: "int16", 4: "int32", 8: "int64"}


def _key_prefix(key_hint: str | None) -> str:
    if not key_hint:
        return "neo"
    normalized = "".join(character if character.isalnum() or character in "_.-" else "_" for character in key_hint)
    return normalized[:64] or "neo"


class _TensorAsNumpy:
    """Self-describing wire wrapper marking a numpy payload as a torch tensor.

    Kept tiny (``__slots__``) so Ray still discovers the contained ``numpy``
    array and stores it zero-copy.
    """

    __slots__ = ("data", "torch_dtype", "reinterpret")

    def __init__(self, data: np.ndarray, torch_dtype: str, reinterpret: bool) -> None:
        self.data = data
        self.torch_dtype = torch_dtype
        self.reinterpret = reinterpret

    def __getstate__(self):
        return (self.data, self.torch_dtype, self.reinterpret)

    def __setstate__(self, state):
        self.data, self.torch_dtype, self.reinterpret = state


def _optional_ray_object_store() -> ObjectStore[Any] | None:
    try:
        return _object_store()
    except RuntimeError:
        import ray

        if not ray.is_initialized():
            return None
    from verl.single_controller.ray.object_store import RayObjectStore

    # Refs are plain ObjectRefs either way, so they stay readable once a Runtime starts.
    return RayObjectStore()


def _ray_object_store() -> ObjectStore[Any]:
    store = _optional_ray_object_store()
    if store is None:
        raise RuntimeError("Ray storage needs a Runtime ObjectStore or an initialized Ray process")
    return store


class RayStorageEngine(_BaseStorageEngine):
    """Ray-based default storage engine.

    Payloads go through the ObjectStore installed by the Runtime; Ray processes
    without a Runtime (raw ``ray.remote`` actors, drivers before Runtime start)
    use Ray's object store directly. Processes without Ray keep payloads inline
    in ``LocalRef`` instead of starting a Ray cluster. Tensor values are
    converted with :meth:`to_wire` / :meth:`from_wire` so stored buffers are
    not corrupted by later in-place mutation.
    """

    backend = "ray_object_store"

    def to_wire(self, value: Any) -> Any:
        """Convert a ``torch.Tensor`` into a storable numpy payload (identity otherwise).

        Always copies into a fresh numpy buffer so later in-place mutation of the
        source tensor (or Ray zero-copy) cannot corrupt the stored object.
        """
        if not (_HAVE_TORCH and isinstance(value, torch.Tensor)):
            return value
        t = value.detach().cpu().contiguous().clone()
        try:
            return _TensorAsNumpy(t.numpy(), str(t.dtype), False)
        except (TypeError, RuntimeError):
            # numpy has no matching dtype (e.g. bfloat16/float8): reinterpret the
            # raw bits through a same-width integer type, restored on read.
            int_dtype = getattr(torch, _REINTERPRET_INT[t.element_size()])
            return _TensorAsNumpy(t.view(int_dtype).numpy().copy(), str(t.dtype), True)

    def from_wire(self, value: Any) -> Any:
        """Inverse of :func:`to_wire`: rebuild a ``torch.Tensor`` from the wrapper.

        Clone so worker-side in-place ops (padding / loss prep) cannot mutate the
        Ray object-store payload shared across ranks.
        """
        if not isinstance(value, _TensorAsNumpy):
            return value
        # Ray object-store numpy arrays are read-only.  Copy before handing the
        # buffer to torch so it never exposes a tensor backed by immutable
        # storage (and does not emit torch's non-writable-array warning).
        tensor = torch.from_numpy(value.data.copy())
        if value.reinterpret:
            tensor = tensor.view(getattr(torch, value.torch_dtype.split(".")[-1]))
        return tensor

    # ------------------------------------------------------------------
    def put(
        self,
        value: Any,
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> Ref:
        store = _optional_ray_object_store()
        if store is None:
            ref = LocalRef.of(value)
            ref.dtype = _infer_dtype(value, spec)
            ref.shape = _infer_shape(value, spec)
            return ref
        uid = new_uid(prefix=f"{_key_prefix(key_hint)}-")
        reference = store.put(uid, self.to_wire(value))
        return Ref(
            backend=self.backend,
            uid=uid,
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
        return [self.put(value, key_hint=key_hint, spec=spec) for value in values]

    # ------------------------------------------------------------------
    def get(self, ref: Ref) -> Any:
        if ref.backend != self.backend:
            raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        value = self.from_wire(_ray_object_store().get(ref.dataptr))
        slice_item = self.apply_slice(value, ref.slice_spec)
        slice_item = ref.apply_ops(slice_item)
        return slice_item

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]:
        if not refs:
            return []
        for ref in refs:
            if ref is not None and ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        indexed_refs = [(index, ref) for index, ref in enumerate(refs) if ref is not None]
        remote_values = _ray_object_store().get_many([ref.dataptr for _, ref in indexed_refs])
        values: list[Any] = [None] * len(refs)
        for (index, ref), value in zip(indexed_refs, remote_values, strict=True):
            value = self.from_wire(value)
            if apply_ops:
                value = ref.apply_ops(self.apply_slice(value, ref.slice_spec))
            values[index] = value
        return values

    def consolidate(
        self,
        old_engine,
        refs: list[Ref],
    ) -> Ref:
        raise NotImplementedError("consolidate not implemented for RayStorageEngine")

    def release(self, refs: Ref | list[Ref]) -> None:
        if isinstance(refs, Ref):
            refs = [refs]
        for ref in refs:
            if ref is not None and ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        remote = {ref.dataptr for ref in refs if ref is not None}
        if remote:
            store = _ray_object_store()
            for reference in remote:
                store.delete(reference)


# Ray-oriented alias used by existing verl tests / smoke scripts.
DefaultStorageEngine = RayStorageEngine


# ---------------------------------------------------------------------------
# Module-level default engine
# ---------------------------------------------------------------------------


_DEFAULT: StorageEngine | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_storage_engine() -> StorageEngine:
    """Return the explicit default or the engine for this process's Runtime backend.

    Without an active Runtime, the engine follows the default Runtime backend,
    the same backend a Runtime started in this process would use by default.
    """
    engine = _DEFAULT
    if engine is not None:
        return engine
    from verl.runtime import current_runtime

    try:
        backend = current_runtime().backend
    except RuntimeError:
        backend = DEFAULT_BACKEND
    # Implicit selection must not outlive the Runtime that selects the backend.
    return storage_engine_for_runtime(backend)


def set_default_storage_engine(engine: StorageEngine | None) -> None:
    """Install ``engine`` as the module-level default (used by tests)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = engine
    _sync_default_engine_registry(engine)


def _sync_default_engine_registry(engine: StorageEngine | None) -> None:
    """Keep NeoProto's compatibility registry aligned with the selected default."""
    from verl.experimental.neoproto import neo

    if engine is None:
        neo.GLOBAL_ENGINE_DICT.pop("default", None)
        return
    neo.GLOBAL_ENGINE_DICT["default"] = engine
    backend = getattr(engine, "backend", None)
    if backend:
        neo.GLOBAL_ENGINE_DICT[backend] = engine
    if isinstance(engine, RayStorageEngine):
        neo.GLOBAL_ENGINE_DICT["ray_object_store"] = engine


def get_engine_for_backend(backend: str | None = None) -> StorageEngine:
    """Return a :class:`StorageEngine` able to resolve a ref of ``backend``.

    Engines are reconstructable on demand because the live handle travels in
    :attr:`Ref.dataptr`. A :class:`NeoProto` therefore does not need to carry a
    persistent engine; it derives one per ref at materialize time from
    ``ref.backend``.

    - ``"memory"`` -> a fresh :class:`InMemoryStorageEngine` for tests and
      offline debug.
    - ``"ray_object_store"`` -> the Ray adapter.
    - ``None`` / ``"default"`` -> the explicitly configured Runtime adapter.
    """
    if backend == "local":
        raise ValueError("LocalRef is inline and does not use a StorageEngine")
    if backend == "memory":
        return InMemoryStorageEngine()
    if backend == "ray_object_store":
        return RayStorageEngine()
    if backend == "torchstore":
        from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine

        return TorchStorageEngine()
    if backend in (None, "default"):
        return get_default_storage_engine()
    raise ValueError(f"unsupported NeoProto storage backend {backend!r}")


def storage_engine_for_runtime(backend: str) -> StorageEngine:
    """Construct the only NeoProto engine valid for ``backend``."""
    if backend == "ray":
        return RayStorageEngine()
    if backend == "monarch":
        from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine

        return TorchStorageEngine()
    raise ValueError(f"unsupported NeoProto Runtime backend {backend!r}")


def configure_storage_engine(backend: str) -> StorageEngine:
    """Bind NeoProto storage to one explicit Runtime backend."""
    engine = storage_engine_for_runtime(backend)
    set_default_storage_engine(engine)
    return engine
