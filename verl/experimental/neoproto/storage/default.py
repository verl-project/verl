# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Default storage backends for :class:`NeoProto`.

This module provides two concrete implementations of :class:`StorageEngine`:

- :class:`InMemoryStorageEngine` -- process-local dict. Mostly useful for
  unit tests and debugging; no Ray dependency.
- :class:`DefaultStorageEngine` -- stores payloads with Ray ``ray.put`` /
  ``ray.get`` (tensors are converted to numpy on the wire via
  :meth:`DefaultStorageEngine.to_wire`).

Users select a backend explicitly or rely on :func:`get_default_storage_engine`,
which returns ``DefaultStorageEngine`` by default.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterable, Optional

import numpy as np
import ray

try:  # pragma: no cover - optional dep
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    Ref,
    StorageEngine,
    _BaseStorageEngine,
    _infer_dtype,
    _infer_shape,
    new_uid,
)

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

    Not thread-safe across processes; used for tests, notebooks, and the
    "zero ray" debug path for :class:`NeoProto`.
    """

    backend = "local"

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._refcount: dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def put(
        self,
        value: Any,
        *,
        key_hint: Optional[str] = None,
        spec: Optional[FieldSpec] = None,
        backend: Optional[str] = None,
    ) -> Ref:
        del key_hint, backend  # API parity with DefaultStorageEngine.put
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
        self._check_backend(ref)
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
                self._check_backend(r)
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

    def _check_backend(self, ref: Ref) -> None:
        if ref.backend != self.backend:
            raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")


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


class DefaultStorageEngine(_BaseStorageEngine):
    """Ray-based default storage engine.

    All non-local payloads go through ``ray.put`` / ``ray.get``. Tensor values
    are converted with :meth:`to_wire` / :meth:`from_wire` so stored buffers
    are not corrupted by later in-place mutation. ``backend="local"`` falls
    back to an in-process :class:`InMemoryStorageEngine`.
    """

    backend = "ray"  # namespace; individual refs record ``object_store`` / ``local``

    def __init__(self) -> None:
        self._object_store = None  # type: ignore[assignment]
        self._lock = threading.Lock()
        # local fallback used for values that can't be sent to Ray (e.g. during tests)
        self._fallback = InMemoryStorageEngine()

    # ------------------------------------------------------------------
    def _ray_available(self) -> bool:
        """Return True only inside an initialized Ray process.

        Library-level ``DataProto`` construction must not auto-start a Ray
        cluster. Trainers and Ray actors initialize Ray before using the
        object-store path; standalone callers use the local fallback.
        """
        try:
            return ray.is_initialized()
        except Exception:  # pragma: no cover
            return False

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
        key_hint: Optional[str] = None,
        spec: Optional[FieldSpec] = None,
        backend: Optional[str] = "object_store",
    ) -> Ref:
        if not self._ray_available() or backend == "local":
            return self._fallback.put(value, key_hint=key_hint, spec=spec)

        ref = ray.put(self.to_wire(value))
        uid = ref.hex()
        return Ref(
            backend=backend,
            uid=uid,
            dataptr=ref,
            dtype=_infer_dtype(value, spec),
            shape=_infer_shape(value, spec),
        )

    def put_many(
        self,
        values: Iterable[Any],
        *,
        key_hint: Optional[str] = None,
        spec: Optional[FieldSpec] = None,
    ) -> list[Ref]:
        if not self._ray_available():
            return self._fallback.put_many(values, key_hint=key_hint, spec=spec)
        if not values:
            return []
        # tensor path: one ray.put per value so each tensor can travel by RDMA
        refs: list[Ref] = []
        for v in values:
            r = ray.put(self.to_wire(v))
            refs.append(
                Ref(
                    backend="object_store",
                    uid=r.hex(),
                    dataptr=r,
                    dtype=_infer_dtype(v, spec),
                    shape=_infer_shape(v, spec),
                )
            )
        return refs

    # ------------------------------------------------------------------
    def get(self, ref: Ref) -> Any:
        if ref.backend == "local":
            return self._fallback.get(ref)
        if ref.backend == "object_store":
            value = self.from_wire(ray.get(ref.dataptr))
        else:  # pragma: no cover
            raise ValueError(f"Unknown ref backend: {ref.backend}")
        slice_item = self.apply_slice(value, ref.slice_spec)
        slice_item = ref.apply_ops(slice_item)
        return slice_item

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]:
        backend = refs[0].backend
        if backend == "local":
            return [self.get(r) if r is not None else None for r in refs]
        elif backend == "object_store":
            values = ray.get([r.dataptr for r in refs])
            for i in range(len(values)):
                r = refs[i]
                value = self.from_wire(values[i])
                if apply_ops:
                    value = self.apply_slice(value, r.slice_spec)
                    value = r.apply_ops(value)
                values[i] = value

            return values

    def consolidate(
        self,
        old_engine,
        refs: list[Ref],
    ) -> Ref:
        raise NotImplementedError("consolidate not implemented for DefaultStorageEngine")

    def release(self, refs: Ref | list[Ref]) -> None:
        if isinstance(refs, Ref):
            refs = [refs]
        os_refs: list[Ref] = [r.dataptr for r in refs if r.backend == "object_store"]
        if len(os_refs) > 0:
            # print(f"[NEO] release refs: {os_refs}")
            ray.internal.free(os_refs)
            # gc.collect()


# ---------------------------------------------------------------------------
# Module-level default engine
# ---------------------------------------------------------------------------


_DEFAULT: Optional[StorageEngine] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_storage_engine() -> StorageEngine:
    """Return the module-level default :class:`StorageEngine`.

    Creates a :class:`DefaultStorageEngine` on first access.
    """
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = DefaultStorageEngine()
    return _DEFAULT


def set_default_storage_engine(engine: Optional[StorageEngine]) -> None:
    """Install ``engine`` as the module-level default (used by tests)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = engine
    # Keep NeoProto._engine() in sync for unit tests that inject a counting engine.
    from verl.experimental.neoproto import neo

    if engine is None:
        neo.GLOBAL_ENGINE_DICT.pop("default", None)
        return
    neo.GLOBAL_ENGINE_DICT["default"] = engine
    backend = getattr(engine, "backend", None)
    if backend:
        neo.GLOBAL_ENGINE_DICT[backend] = engine


def get_engine_for_backend(backend: Optional[str] = None) -> StorageEngine:
    """Return a :class:`StorageEngine` able to resolve a ref of ``backend``.

    Engines are reconstructable on demand because the live handle travels in
    :attr:`Ref.dataptr` (the in-memory value for ``local`` refs, the Ray
    ``ObjectRef`` for ``object_store`` refs). A :class:`NeoProto` therefore does
    not need to carry a persistent engine; it derives one per ref at
    materialize time from ``ref.backend``.

    - ``"local"`` -> a fresh :class:`InMemoryStorageEngine` (no Ray needed; the
      payload is read straight back from ``ref.dataptr``).
    - ``"object_store"`` or ``None`` for the write path -> the shared
      :class:`DefaultStorageEngine`.
    """
    if backend == "local":
        return InMemoryStorageEngine()
    elif backend == "object_store":
        return DefaultStorageEngine()
    return get_default_storage_engine()
