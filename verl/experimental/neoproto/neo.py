# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
:class:`NeoProto` -- ref/index-only base container.

This is the backbone behind the public NeoProto-backed
:class:`~verl.experimental.neoproto.views.data_proto.DataProto` compatibility
view used by the V0 trainer.

Design principles
-----------------

1. Driver holds only **metadata + refs + indices**; it never pulls tensor
   bytes into its own process.
2. Workers call :meth:`NeoProto.materialize` to fetch only the keys /
   token ranges they need.
3. Index operations (``select``, ``chunk``, ``concat``, ``repeat``,
   ``slice_tokens``, ``tokens_at``) are all pure metadata rewrites.
4. Serialization splits into two modes (see :meth:`save_to_disk`):
   - default: refs + schema + meta only (debuggable snapshot, tiny);
   - ``include_data=True``: materialize and persist full tensor bytes.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

try:  # pragma: no cover - optional
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    Ref,
    RefTable,
    SliceSpec,
    StorageEngine,
    compose_slice,
)

# Granularity tag for a batch-level ``ref_table`` entry whose single value lives
# in ``Ref.dataptr`` (the old ``meta`` dict is stored this way). It is treated
# like any other ref by the metadata ops; only ``materialize`` consults the
# granularity (alongside ``"shards"``) to skip sample-axis indexing.
FULL_GRANULARITY = "full"


GLOBAL_ENGINE_DICT = {}

# ---------------------------------------------------------------------------
# IndexView
# ---------------------------------------------------------------------------


# All inputs must share the same ref identities per key for a cheap concat.
# ``RefTable.copy()`` produces a new wrapper but keeps the underlying
# ``Ref`` objects so ``is`` / value comparison per key is sufficient.
def same_refs(a: NeoProto, b: NeoProto) -> bool:
    if set(a.ref_table.keys()) != set(b.ref_table.keys()):
        return False
    for k in a.ref_table.keys():
        va, vb = a.ref_table[k], b.ref_table[k]
        if va is vb:
            continue
        if isinstance(va, Ref) and isinstance(vb, Ref):
            if va == vb:
                continue
        elif isinstance(va, np.ndarray | list) and isinstance(vb, np.ndarray | list):
            if len(va) == len(vb) and all(x is y or x == y for x, y in zip(va, vb, strict=False)):
                continue
        return False
    return True


def _payloads_equal(left: Any, right: Any) -> bool:
    """Strict equality used for DataProto-compatible union conflicts."""
    if type(left) is not type(right):
        return False
    if _HAVE_TORCH and isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        from verl.protocol import _deep_equal

        return bool(_deep_equal(left, right, visited=set()))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_payloads_equal(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(_payloads_equal(a, b) for a, b in zip(left, right, strict=True))
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, bool | np.bool_) else False


def is_valid_indices(indices: list[int]) -> bool:
    # Metrics-only worker outputs are empty batches; ``min([])`` must not crash.
    if not indices:
        return False
    if min(indices) == 0 and max(indices) == len(indices) - 1 and len(set(indices)) == len(indices):
        return True
    return False


@dataclass
class IndexView:
    """A sample+token view over a :class:`NeoProto`.

    The driver composes ``IndexView``s to reorder / slice / repeat samples
    without touching payloads. :meth:`NeoProto.materialize` consults the
    view when resolving each ref.

    Attributes
    ----------
    sample_indices:
        1-D ndarray of ints describing the current sample order. ``None``
        means "identity over ``[0, batch_size)``" (cheap default).
    token_slices:
        Per-key override token :class:`SliceSpec`. Applied in addition to each
        ref's intrinsic slice. Absent key => no token slicing beyond the ref.
    """

    sample_indices: Optional[np.ndarray] = None
    token_slices: dict[str, SliceSpec] = field(default_factory=dict)

    def __copy__(self):
        return self.copy()

    def __eq__(self, other: IndexView):
        if self.sample_indices is None and other.sample_indices is None:
            return True
        if self.sample_indices is not None and other.sample_indices is not None:
            return np.all(self.sample_indices == other.sample_indices)
        return False

    def copy(self) -> IndexView:
        return IndexView(
            sample_indices=None if self.sample_indices is None else self.sample_indices.copy(),
            token_slices=dict(self.token_slices),
        )

    # ------------------------------------------------------------------
    def identity(self, n: int) -> IndexView:
        return IndexView(sample_indices=np.arange(n, dtype=np.int64), token_slices=dict(self.token_slices))

    def compose_samples(self, indices: np.ndarray) -> IndexView:
        if self.sample_indices is None:
            return IndexView(sample_indices=indices, token_slices=dict(self.token_slices))
        sample_indices = np.concatenate((self.sample_indices, indices), axis=0)
        return IndexView(
            sample_indices=sample_indices,
            token_slices=dict(self.token_slices),
        )

    def with_token_slice(self, key: str, slice_spec: SliceSpec) -> IndexView:
        ts = dict(self.token_slices)
        prev = ts.get(key)
        ts[key] = compose_slice(prev or (), slice_spec) if prev else slice_spec
        return IndexView(
            sample_indices=None if self.sample_indices is None else self.sample_indices.copy(),
            token_slices=ts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_indices": None if self.sample_indices is None else self.sample_indices.tolist(),
            "token_slices": {k: _slice_spec_dump(v) for k, v in self.token_slices.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexView:
        si = data.get("sample_indices")
        return cls(
            sample_indices=None if si is None else np.asarray(si, dtype=np.int64),
            token_slices={k: _slice_spec_load(v) for k, v in (data.get("token_slices") or {}).items()},
        )


def _slice_spec_dump(spec):
    from verl.experimental.neoproto.storage.engine import _slice_spec_to_primitive

    return _slice_spec_to_primitive(spec)


def _slice_spec_load(raw):
    from verl.experimental.neoproto.storage.engine import _slice_spec_from_primitive

    return _slice_spec_from_primitive(raw)


# ---------------------------------------------------------------------------
# NeoProto base
# ---------------------------------------------------------------------------


@dataclass
class NeoProto:
    """Ref/index-only base container.

    Every method that is "driver-side" returns a new :class:`NeoProto`
    whose :class:`RefTable` is either the same or derived purely via metadata
    manipulation. No materialization happens on the driver.

    Concrete subclasses (``DataProto``, ``Query``, ``Datum``) only add
    typed helpers and *do not* change this contract.
    """

    _materialize_time_s: float = 0.0
    _materialize_calls: int = 0

    @classmethod
    def reset_materialize_stats(cls) -> None:
        cls._materialize_time_s = 0.0
        cls._materialize_calls = 0

    @classmethod
    def pop_materialize_stats(cls) -> dict[str, float]:
        stats = {
            "dataplane/materialize": cls._materialize_time_s,
            "dataplane/materialize_calls": float(cls._materialize_calls),
        }
        cls.reset_materialize_stats()
        return stats

    # Per-key type/shape descriptor; the driver reasons about sizes from here
    # without ever materializing payloads.
    schema: dict[str, FieldSpec] = field(default_factory=dict)
    # Per-key handles (shared ``Ref`` or per-sample ``ndarray[Ref]``). Metadata
    # only -- the actual bytes live in the storage engine. Batch-level metadata
    # (the old ``meta`` dict) is also kept here as ``granularity="meta"`` local
    # refs; use the ``*_meta`` helpers below rather than touching it directly.
    ref_table: RefTable = field(default_factory=lambda: RefTable(batch_size=0))
    # Sample-order + token-slice view; all index ops rewrite this, never refs.
    dim0_index: IndexView = field(default_factory=IndexView)

    def __init__(
        self,
        schema: dict[str, FieldSpec],
        ref_table: RefTable,
        dim0_index: IndexView = None,
    ):
        self.schema = schema
        self.ref_table = ref_table
        self.dim0_index = dim0_index
        self.dirty_cache = {}

    # ------------------------------------------------------------------
    # Basic protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.dim0_index.sample_indices is not None:
            return int(self.dim0_index.sample_indices.shape[0])
        return int(self.ref_table.batch_size)

    def __post_init__(self) -> None:
        self.dirty_cache = {}
        if self.dim0_index is None:
            n = int(self.ref_table.batch_size)
            self.dim0_index = IndexView().identity(n) if n else IndexView()

    def __getstate__(self):
        ref_table = self.ref_table
        if hasattr(self, "_NEO_DUMP"):
            if "ref_table" not in self._NEO_DUMP:
                import ray

                self._NEO_DUMP["ref_table"] = ray.put(self.ref_table)
            ref_table = self._NEO_DUMP["ref_table"]
        if hasattr(self, "OBJ_REF") and hasattr(self, "LOCAL_REF"):
            dump_start = time.time()
            ref_table = (self.OBJ_REF, self.LOCAL_REF)
            dump_end = time.time()
            logging.info(f"[neo-ser] dump ref_table ({type(ref_table)}) cost: {dump_end - dump_start}")
        return self.schema, ref_table, self.dim0_index

    def __setstate__(self, data):
        import ray

        self.schema, self.ref_table, self.dim0_index = data
        ref_type = type(self.ref_table)
        if isinstance(self.ref_table, ray.ObjectRef):
            self.ref_table = ray.get(self.ref_table)
        elif isinstance(self.ref_table, tuple):
            dump_start = time.time()
            self.OBJ_REF, self.LOCAL_REF = self.ref_table
            obj_ref = ray.get(self.OBJ_REF)
            local_ref = ray.get(self.LOCAL_REF)
            obj_ref.update(local_ref)
            self.ref_table = obj_ref
            dump_end = time.time()
            logging.info(f"[neo-ser] load ref_table ({ref_type}) cost: {dump_end - dump_start}")
        self.__post_init__()

    def __copy__(self):
        new = type(self).__new__(type(self))
        state = dict(self.__dict__)
        state["schema"] = dict(self.schema)
        state["ref_table"] = self.ref_table.copy()
        state["dim0_index"] = deepcopy(self.dim0_index)
        state["dirty_cache"] = dict(self.dirty_cache)
        new.__dict__.update(state)
        return new

    def __del__(self):
        import ray

        if hasattr(self, "_NEO_DUMP") and "ref_table" in self._NEO_DUMP:
            ray.internal.free([self._NEO_DUMP["ref_table"]])
            del self._NEO_DUMP

        for _attr in ("OBJ_REF", "LOCAL_REF"):
            if hasattr(self, _attr):
                delattr(self, _attr)

    def _cal_bs_size(self, value: Any) -> int:
        if isinstance(value, Ref):
            return value.estimated_shape[0]
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            if isinstance(value[0], Ref):
                ref_size = 0
                for ref in value:
                    if isinstance(ref, Ref):
                        ref_size += ref.estimated_shape[0]
                    else:
                        ref_size += 1
                return ref_size
            else:
                return len(value)
        elif torch.is_tensor(value):
            return len(value)
        else:  # pragma: no cover
            raise ValueError(f"Unknown value type: {type(value)}")

    # ------------------------------------------------------------------
    # Storage wiring
    # ------------------------------------------------------------------
    def _engine(self, backend: Optional[str] = None) -> StorageEngine:
        """Resolve a storage engine for ``backend``.

        ``NeoProto`` no longer holds an engine; one is constructed on demand
        from a ref's ``backend`` (the live handle lives in ``Ref.dataptr``).
        ``backend=None`` returns the default engine, used by the write path
        (``put``/``add_ref``) where the backend is decided by the engine itself.
        """
        global GLOBAL_ENGINE_DICT
        from verl.experimental.neoproto.storage.default import get_engine_for_backend

        if backend is None:
            backend = "default"

        engine = GLOBAL_ENGINE_DICT.get(backend, None)
        if engine is None:
            engine = get_engine_for_backend(backend)
            GLOBAL_ENGINE_DICT[backend] = engine

        return engine

    def finalize(self):
        dirty_keys = list(self.dirty_cache.keys())
        for k in dirty_keys:
            if k in self.dirty_cache and self.dirty_cache[k]:
                cache_name = self._get_cache_name(k)
                v = getattr(self, cache_name, None)
                setattr(self, k, v)

    def to_refs(self):
        obj = NeoProto(
            schema=self.schema,
            ref_table=self.ref_table,
            dim0_index=self.dim0_index,
        )
        # TODO: handle functional fields
        for k in getattr(self, "_EXPLICIT_FUNC_FIELDS", {}):
            v = getattr(self, k)
        for k, v in self.ref_table.items():
            if v.is_local():
                v = v.dataptr
            setattr(obj, k, v)
        return obj

    def add_ref(self, key: str, ref: Ref, granularity: str = "sample"):
        assert key not in self.ref_table and key not in self.schema, f"Ref {key} already exists"
        self.ref_table[key] = ref
        self.schema[key] = fieldspec_from_ref(ref, granularity=granularity)

    def update(self, other: NeoProto) -> NeoProto:
        if self.ref_table is None:
            self.ref_table = other.ref_table
        else:
            for k, v in other.ref_table.items():
                self.ref_table[k] = v
            self.ref_table.batch_size = other.ref_table.batch_size
        if self.schema is None:
            self.schema = other.schema
        else:
            for k, v in other.schema.items():
                self.schema[k] = v
        if other.dim0_index is not None:
            self.dim0_index = other.dim0_index
        # meta entries live in ref_table/schema, so the loops above already
        # merged them (other overrides self for shared keys).

    # ------------------------------------------------------------------
    # Index-only operations
    # ------------------------------------------------------------------
    def select(self, keys: Iterable[str], *, deepcopy_meta: bool = False) -> NeoProto:
        keys = list(keys)
        new_refs = RefTable(
            {k: self.ref_table[k] for k in keys if k in self.ref_table},
            batch_size=self.ref_table.batch_size,
        )
        new_schema = {k: self.schema[k] for k in keys if k in self.schema}
        return type(self)(
            schema=new_schema,
            ref_table=new_refs,
            dim0_index=deepcopy(self.dim0_index),
        )

    def select_idxs(self, idxs: list[int], *, deepcopy_meta: bool = False) -> NeoProto:
        # Match verl.protocol.DataProto.select_idxs: idxs are into the *current*
        # logical batch order, not the underlying ref_table row space.
        del deepcopy_meta
        if _HAVE_TORCH and isinstance(idxs, torch.Tensor):
            logical_indices = idxs.detach().cpu().numpy()
        elif isinstance(idxs, np.ndarray):
            logical_indices = idxs
        else:
            logical_indices = np.asarray(list(idxs))
        if logical_indices.dtype == np.bool_:
            if logical_indices.ndim != 1 or len(logical_indices) != len(self):
                raise IndexError(
                    f"Boolean index must be one-dimensional with length {len(self)}, got shape {logical_indices.shape}"
                )
            logical_indices = np.flatnonzero(logical_indices)
        else:
            logical_indices = logical_indices.astype(np.int64, copy=False)
        base = (
            self.dim0_index.sample_indices
            if self.dim0_index.sample_indices is not None
            else np.arange(self.ref_table.batch_size, dtype=np.int64)
        )
        return self._with_sample_indices(base[logical_indices])

    def reorder(self, indices: Iterable[int]):
        # Accept torch/numpy/list indices into the current logical order.
        if _HAVE_TORCH and isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
        if self.dim0_index is None or self.dim0_index.sample_indices is None:
            sample_indices = indices
        else:
            sample_indices = self.dim0_index.sample_indices[indices]
        self.dim0_index.sample_indices = sample_indices
        self.clear_cache()

    def clear_cache(self):
        pass

    def pop(self, keys: Iterable[str]) -> NeoProto:
        keys = list(keys)
        popped_refs = RefTable({}, batch_size=self.ref_table.batch_size)
        popped_schema: dict[str, FieldSpec] = {}
        for k in keys:
            if k in self.ref_table:
                popped_refs[k] = self.ref_table.pop(k)
            if k in self.schema:
                popped_schema[k] = self.schema.pop(k)
        return type(self)(
            schema=popped_schema,
            ref_table=popped_refs,
            dim0_index=deepcopy(self.dim0_index),
        )

    def rename(self, mapping: dict[str, str], deep_copy: bool = False) -> NeoProto:
        new_refs: dict[str, Any] = {}
        for k, v in self.ref_table.items():
            new_refs[mapping.get(k, k)] = v
        new_schema = {mapping.get(k, k): v for k, v in self.schema.items()}
        if deep_copy:
            return type(self)(
                schema=new_schema,
                ref_table=RefTable(new_refs, batch_size=self.ref_table.batch_size),
                dim0_index=deepcopy(self.dim0_index),
            )
        # In-place: replace the table (shares Ref objects; old RefTable only
        # releases entries that were explicitly GC-queued, not live ``_refs``).
        self.schema = new_schema
        self.ref_table = RefTable(new_refs, batch_size=self.ref_table.batch_size)
        return self

    def slice(self, start=None, end=None, step=None) -> NeoProto:
        return self._with_sample_indices(self.dim0_index.sample_indices[slice(start, end, step)])

    # ------------------------------------------------------------------
    # Chunking / concat / repeat / copy
    # ------------------------------------------------------------------
    def chunk(self, chunks: int) -> list[NeoProto]:
        n = len(self)
        if chunks <= 0:
            raise ValueError("chunks must be > 0")
        if n == 0:
            return [self] if chunks == 1 else [self.empty_like() for _ in range(chunks)]
        assert n % chunks == 0, f"batch size {n} not divisible by {chunks}"
        step = n // chunks
        base = (
            self.dim0_index.sample_indices
            if self.dim0_index.sample_indices is not None
            else np.arange(self.ref_table.batch_size, dtype=np.int64)
        )
        out: list[NeoProto] = []
        # _NEO_DUMP = {}
        for i in range(chunks):
            sub = base[i * step : (i + 1) * step]
            out.append(self._with_sample_indices(sub))
            # out[-1]._NEO_DUMP = _NEO_DUMP
        return out

    def deep_chunk(self, chunks: int) -> list[NeoProto]:
        n = len(self)
        assert n == chunks, f"deep_chunk only supports same size chunks: batch size {n} != chunks {chunks}"
        out: list[NeoProto] = []
        refs = self.ref_table.copy()
        for i in range(chunks):
            indices = np.arange(1, dtype=np.int64)
            refs = self.ref_table.select(i)
            new_index = IndexView(
                sample_indices=indices,
                token_slices=dict(self.dim0_index.token_slices),
            )
            chunked_proto = type(self)(
                schema=dict(self.schema),
                ref_table=refs,
                dim0_index=new_index,
            )
            out.append(chunked_proto)

        return out

    @staticmethod
    def concat(items: list[NeoProto], new_index: bool = False) -> NeoProto:
        if not items:
            raise ValueError("concat called with empty list")
        head = items[0]

        if all(same_refs(x, head) for x in items):
            indices = np.concatenate(
                [
                    (
                        x.dim0_index.sample_indices
                        if x.dim0_index.sample_indices is not None
                        else np.arange(x.ref_table.batch_size, dtype=np.int64)
                    )
                    for x in items
                ]
            )
            return head._with_sample_indices(indices)

        # Generic path: stitch per-sample refs into a new table.
        keys = set(head.ref_table.keys())
        for it in items[1:]:
            keys &= set(it.ref_table.keys())
        new_table: dict[str, Any] = {}
        total_len = 0
        expanded_shared = False
        for k in keys:
            pieces: list[Any] = []
            bs = 0
            for it in items:
                v = it.ref_table[k]
                n_it = len(it)
                if isinstance(v, np.ndarray):
                    sel = it.dim0_index.sample_indices if it.dim0_index.sample_indices is not None else slice(None)
                    pieces.append(v[sel] if not isinstance(sel, slice) else v[:n_it])
                elif isinstance(v, list):
                    sel_idx = (
                        it.dim0_index.sample_indices.tolist()
                        if it.dim0_index.sample_indices is not None
                        else list(range(n_it))
                    )
                    pieces.append([v[i] for i in sel_idx])
                elif isinstance(v, Ref):
                    # Shared column: expand to one sample-view ref per selected row so
                    # independent tables (different workers / batches) concat correctly.
                    if head.schema.get(k) is not None and head.schema[k].granularity == FULL_GRANULARITY:
                        pieces.append([v])
                    else:
                        sel_idx = (
                            it.dim0_index.sample_indices
                            if it.dim0_index.sample_indices is not None
                            else np.arange(n_it, dtype=np.int64)
                        )
                        pieces.append([v.with_slice((slice(int(i), int(i) + 1),)) for i in sel_idx])
                        expanded_shared = True
                bs += n_it
                # else:  # shared ref
                #     pieces.append([v] * n_it)
            merged = np.empty(sum(len(p) for p in pieces), dtype=object)
            cursor = 0
            for p in pieces:
                merged[cursor : cursor + len(p)] = p
                cursor += len(p)
            # Merge the same refs
            if len(merged) > 0 and isinstance(merged[0], Ref):
                if len(set([ref.uid for ref in merged])) == 1 and not expanded_shared:
                    merged = Ref.from_ref(merged[0])
                # Meta Info
                elif head.schema.get(k) is not None and head.schema[k].granularity == FULL_GRANULARITY:
                    merged = Ref.from_ref(merged[0])
            # TODO: Fix schema shape for "shards"
            new_table[k] = merged
            total_len = bs
        # build index
        index = IndexView().identity(total_len)
        if items[0].dim0_index is not None and not new_index and not expanded_shared:
            sample_indices = []
            for it in items:
                if it.dim0_index.sample_indices is None:
                    sample_indices = None
                    break
                sample_indices += it.dim0_index.sample_indices.tolist()
            # validate concat sample indices
            if sample_indices is not None and is_valid_indices(sample_indices):
                index = IndexView(sample_indices=np.array(sample_indices))
        return type(head)(
            schema=dict(head.schema),
            ref_table=RefTable(new_table, batch_size=total_len),
            dim0_index=index,
        )

    def consolidate(
        self,
        storage: Optional[StorageEngine] = None,
    ) -> NeoProto:
        """Store the logical batch as one full-column ref per field.

        The operation is non-mutating: views and source refs remain valid.
        With the Ray default engine each consolidated column is one object.
        """
        target_keys = [
            key
            for key in self.ref_table.keys()
            if self.schema.get(key) is None or self.schema[key].granularity != FULL_GRANULARITY
        ]

        new_schema = dict(self.schema)
        stored_keys: list[str] = []
        local_values: dict[str, Any] = {}

        def classify(key: str, value: Any) -> bool:
            if key in new_schema and hasattr(value, "shape"):
                new_schema[key] = new_schema[key].with_shape(tuple(value.shape))
                # TODO: Decide whether consolidated tensor fields should always use shared granularity.
            return (_HAVE_TORCH and torch.is_tensor(value)) or (isinstance(value, np.ndarray) and value.dtype != object)

        engine = storage if storage is not None else self._engine()
        stored_refs: dict[str, Ref] = {}
        created_refs: list[Ref] = []
        fragmented_values: dict[str, Any] = {}
        prepare_fragmented = getattr(engine, "prepare_fragmented", None)
        sample_indices = self.dim0_index.sample_indices
        identity_order = sample_indices is None or np.array_equal(
            sample_indices,
            np.arange(self.ref_table.batch_size, dtype=sample_indices.dtype),
        )
        if callable(prepare_fragmented) and identity_order and len(self) == self.ref_table.batch_size:
            for key in target_keys:
                prepared = prepare_fragmented(
                    self.ref_table[key],
                    batch_size=self.ref_table.batch_size,
                )
                if prepared is not None:
                    fragmented_values[key] = prepared
            if fragmented_values:
                target_keys = list(fragmented_values) + [key for key in target_keys if key not in fragmented_values]
        try:

            def iter_stored_values():
                for key in target_keys:
                    if key in fragmented_values:
                        value = fragmented_values[key]
                        if key in new_schema:
                            new_schema[key] = new_schema[key].with_shape(tuple(value.shape))
                        stored_keys.append(key)
                        yield value
                        continue
                    value = self.materialize(keys=[key])[key]
                    if classify(key, value):
                        stored_keys.append(key)
                        yield value
                        del value
                    else:
                        local_values[key] = value

            created_refs = engine.put_many(
                iter_stored_values(),
                key_hint="neo_batch",
                spec=None,
            )

            if len(created_refs) != len(stored_keys):
                raise RuntimeError(f"put_many returned {len(created_refs)} refs for {len(stored_keys)} values")
            if created_refs:
                stored_refs = dict(zip(stored_keys, created_refs, strict=False))
        except Exception:
            if created_refs:
                discard = getattr(engine, "discard", None)
                if callable(discard):
                    discard(created_refs)
                else:
                    engine.release(created_refs)
            raise

        # TODO: Define ownership transfer so callers can explicitly release source refs after commit.
        new_refs = dict(self.ref_table.copy().items())
        new_refs.update(stored_refs)
        for key, value in local_values.items():
            ref = Ref.from_local(value)
            ref.dtype = new_schema[key].dtype if key in new_schema else "object"
            ref.shape = tuple(getattr(value, "shape", ()))
            new_refs[key] = ref

        batch_size = len(self)
        return type(self)(
            schema=new_schema,
            ref_table=RefTable(new_refs, batch_size=batch_size),
            dim0_index=IndexView().identity(batch_size),
        )

    def repeat(self, repeat_times: int = 2, interleave: bool = True) -> NeoProto:
        if repeat_times <= 0:
            raise ValueError("repeat_times must be > 0")
        base = (
            self.dim0_index.sample_indices
            if self.dim0_index.sample_indices is not None
            else np.arange(self.ref_table.batch_size, dtype=np.int64)
        )
        if interleave:
            indices = np.repeat(base, repeat_times)
        else:
            indices = np.tile(base, repeat_times)
        return self._with_sample_indices(indices)

    def _is_permutation_indices(self, indices: Optional[np.ndarray], n: int) -> bool:
        if indices is None:
            return True
        if len(indices) != n:
            return False
        return bool(indices.min() == 0 and indices.max() == n - 1 and len(set(indices.tolist())) == n)

    def _rebind_to_logical_identity(self) -> NeoProto:
        """Push the current sample view into refs and return an identity-indexed copy."""
        sample_indices = self.dim0_index.sample_indices
        if sample_indices is None:
            return self
        n = int(len(sample_indices))
        if np.array_equal(sample_indices, np.arange(n, dtype=sample_indices.dtype)):
            return self
        new_refs: dict[str, Any] = {}
        for key, entry in self.ref_table.items():
            spec = self.schema.get(key)
            if spec is not None and spec.granularity == FULL_GRANULARITY:
                new_refs[key] = entry
                continue
            if isinstance(entry, Ref):
                gathered = np.empty(n, dtype=object)
                for i, src in enumerate(sample_indices):
                    gathered[i] = entry.with_slice((slice(int(src), int(src) + 1),))
                new_refs[key] = gathered
            elif isinstance(entry, np.ndarray):
                new_refs[key] = entry[sample_indices]
            elif isinstance(entry, list):
                new_refs[key] = [entry[int(i)] for i in sample_indices]
            else:
                new_refs[key] = entry
        return type(self)(
            schema=dict(self.schema),
            ref_table=RefTable(new_refs, batch_size=n),
            dim0_index=IndexView().identity(n),
        )

    def union(self, other: NeoProto) -> NeoProto:
        if len(self) != len(other):
            raise ValueError(f"union requires equal batch size, got {len(self)} vs {len(other)}")
        sample_indices = self.dim0_index.sample_indices
        other_sample_indices = other.dim0_index.sample_indices
        n = len(self)
        # Inverse-permutation alignment only works for true permutations. After
        # repeat()/gather views, rebind both sides to logical identity first.
        needs_rebind = False
        if sample_indices is not None and other_sample_indices is not None:
            if not np.array_equal(sample_indices, other_sample_indices):
                if not (
                    self._is_permutation_indices(sample_indices, n)
                    and self._is_permutation_indices(other_sample_indices, n)
                ):
                    needs_rebind = True
        if needs_rebind:
            left = self._rebind_to_logical_identity()
            right = other._rebind_to_logical_identity()
            return left.union(right)

        new_refs = RefTable(dict(self.ref_table.items()), batch_size=self.ref_table.batch_size)
        for k, v in other.ref_table.items():
            if k in new_refs:
                left_ref = new_refs[k]
                left_spec = self.schema.get(k)
                right_spec = other.schema.get(k)
                if (
                    left_spec is not None
                    and right_spec is not None
                    and left_spec.granularity == FULL_GRANULARITY
                    and right_spec.granularity == FULL_GRANULARITY
                    and isinstance(left_ref, Ref)
                    and isinstance(v, Ref)
                    and left_ref.backend == "local"
                    and v.backend == "local"
                    and not left_ref.apply_funcs
                    and not v.apply_funcs
                ):
                    if not _payloads_equal(left_ref.dataptr, v.dataptr):
                        raise AssertionError(f"`{k}` differs between NeoProto operands")
                    continue
                left_value = self.materialize([k])[k]
                right_value = other.materialize([k])[k]
                if not _payloads_equal(left_value, right_value):
                    raise AssertionError(f"`{k}` differs between NeoProto operands")
                continue
            if other.schema.get(k) is not None and other.schema[k].granularity == FULL_GRANULARITY:
                new_refs[k] = v
                continue
            if (
                sample_indices is not None
                and other_sample_indices is not None
                and not np.array_equal(sample_indices, other_sample_indices)
            ):
                inv1 = np.empty_like(sample_indices)
                inv1[sample_indices] = np.arange(len(sample_indices))  # inverse permutation
                new_indices = other_sample_indices[inv1]
                if isinstance(v, np.ndarray):
                    v = v[new_indices]
                elif isinstance(v, Ref):
                    # Align shared column into self's sample-index space without materializing.
                    aligned = np.empty(len(sample_indices), dtype=object)
                    for physical, src in enumerate(new_indices):
                        aligned[physical] = v.with_slice((slice(int(src), int(src) + 1),))
                    v = aligned
                else:
                    raise TypeError(f"Unsupported ref type in union: {type(v)}")
            new_refs[k] = v
        new_schema = dict(self.schema)
        for k, v in other.schema.items():
            new_schema.setdefault(k, v)
        return type(self)(
            schema=new_schema,
            ref_table=new_refs,
            dim0_index=deepcopy(self.dim0_index),
        )

    def copy(self) -> NeoProto:
        """
        Deep-copy the batch.
        """
        cloned_batch = self._with_sample_indices(self.dim0_index.sample_indices)
        # Create a new uid for deep-copy
        for k, refs in cloned_batch.ref_table.items():
            if isinstance(refs, Ref):
                refs.uid = copy.deepcopy(refs.uid)
            else:
                for r in refs:
                    r.uid = copy.deepcopy(r.uid)
        return cloned_batch

    def push_down_indices(self):
        """Push down all sample indices to ``key``."""
        for key in self.ref_table.keys():
            v = self.ref_table[key]
            if isinstance(v, Ref):
                self.ref_table[key] = v.with_slice((self.dim0_index.sample_indices,))
            elif isinstance(v, np.ndarray):
                start = 0
                for i in range(len(v)):
                    size = v[i].estimated_shape[0]
                    v_sample_indices = np.array([j for j in range(size) if j + start in self.dim0_index.sample_indices])
                    self.ref_table[key] = v[i].with_slice(v_sample_indices)
                    start += size
            else:
                raise ValueError(f"Unsupported ref type: {type(v)}")
        self.dim0_index = IndexView().identity()

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(
        self,
        key: int | slice | list[int] | np.ndarray | torch.Tensor,
    ) -> NeoProto:
        base = (
            self.dim0_index.sample_indices
            if self.dim0_index.sample_indices is not None
            else np.arange(self.ref_table.batch_size, dtype=np.int64)
        )
        if isinstance(key, int | np.integer):
            return self._with_sample_indices(base[int(key) : int(key) + 1])
        if isinstance(key, slice):
            return self._with_sample_indices(base[key])
        if _HAVE_TORCH and isinstance(key, torch.Tensor):
            return self._with_sample_indices(base[key.cpu().numpy().astype(np.int64)])
        if isinstance(key, list | tuple | np.ndarray):
            return self._with_sample_indices(base[np.asarray(list(key), dtype=np.int64)])
        raise TypeError(f"Unsupported key type for NeoProto: {type(key)}")

    # ------------------------------------------------------------------
    # Token-level indexing
    # ------------------------------------------------------------------
    def slice_tokens(
        self,
        key: str,
        start: int,
        stop: Optional[int] = None,
        step: int = 1,
        *,
        token_axis: Optional[int] = None,
    ) -> NeoProto:
        """Return a view that restricts ``key`` to tokens ``[start, stop)``."""
        axis = token_axis
        if axis is None:
            spec = self.schema.get(key)
            axis = spec.token_axis if (spec is not None and spec.token_axis is not None) else 1
        dims: list[Any] = [None] * (axis + 1)
        dims[axis] = slice(start, stop, step)
        new_index = self.dim0_index.with_token_slice(key, tuple(dims))
        return type(self)(
            schema=dict(self.schema),
            ref_table=self.ref_table.copy(),
            dim0_index=new_index,
        )

    def tokens_at(
        self,
        key: str,
        token_indices: Iterable[int],
        *,
        token_axis: Optional[int] = None,
    ) -> NeoProto:
        """Return a view where ``key`` selects specific ``token_indices``."""
        axis = token_axis
        if axis is None:
            spec = self.schema.get(key)
            axis = spec.token_axis if (spec is not None and spec.token_axis is not None) else 1
        dims: list[Any] = [None] * (axis + 1)
        dims[axis] = list(token_indices)
        new_index = self.dim0_index.with_token_slice(key, tuple(dims))
        return type(self)(
            schema=dict(self.schema),
            ref_table=self.ref_table.copy(),
            dim0_index=new_index,
        )

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------
    def get_refs(self, key: Optional[str] = None) -> dict[str, Ref]:
        refs = self.ref_table[key]
        if isinstance(refs, Ref):
            return refs
        else:
            if len(self.dim0_index.sample_indices) == self.ref_table.batch_size:
                return refs
            else:
                # TODO: handle this case properly, e.g., not single sample case
                refs_to_get = [refs[idx] for idx in self.dim0_index.sample_indices]
            return refs_to_get

    def materialize(
        self,
        keys: Optional[Iterable[str]] = None,
        *,
        device: Optional[Any] = None,
        as_: str = "dict",
        auto_padding: bool = False,
    ) -> dict[str, Any]:
        """Resolve refs and return a ``{key: tensor-or-object}`` dict.

        ``keys`` defaults to the entire schema; pass a subset to avoid
        transferring bytes you don't need.
        """
        keys = list(keys) if keys is not None else list(self.ref_table.keys())
        sample_idx = self.dim0_index.sample_indices
        out: dict[str, Any] = {}
        _t0 = time.time()

        # No engine is stored on the proto: build one per distinct ``backend``
        # from the refs themselves and cache it for the duration of this call.
        engine_cache: dict[Optional[str], StorageEngine] = {}

        def _engine_for(ref: Ref) -> StorageEngine:
            backend = ref.backend
            eng = engine_cache.get(backend)
            if eng is None:
                eng = self._engine(backend)
                engine_cache[backend] = eng
            return eng

        # Engines with a batched row-read capability can resolve all selected
        # shared columns together instead of reopening/fetching once per key.
        batched_rows: dict[str, Any] = {}
        row_read_groups: dict[int, dict[str, Any]] = {}
        if sample_idx is not None:
            for key in keys:
                ref = self.ref_table.get(key)
                spec = self.schema.get(key)
                if isinstance(ref, Ref) and spec is not None and spec.granularity != FULL_GRANULARITY:
                    eng = _engine_for(ref)
                    if hasattr(eng, "get_rows_many"):
                        # TODO: Promote batched row reads into the StorageEngine capability contract.
                        group = row_read_groups.setdefault(id(eng), {"engine": eng, "keys": [], "refs": []})
                        group["keys"].append(key)
                        group["refs"].append(ref)
            for group in row_read_groups.values():
                values = group["engine"].get_rows_many(group["refs"], sample_idx)
                batched_rows.update(zip(group["keys"], values, strict=False))

        # Ray can resolve a list of ObjectRefs in one call. Group shared
        # object-store columns that cannot use a backend-specific row reader,
        # then apply the logical sample/token views below exactly as before.
        batched_shared: dict[str, Any] = {}
        shared_get_groups: dict[int, dict[str, Any]] = {}
        for key in keys:
            ref = self.ref_table.get(key)
            if not isinstance(ref, Ref) or ref.backend == "local" or key in batched_rows:
                continue
            spec = self.schema.get(key)
            eng = _engine_for(ref)
            do_index = sample_idx is not None and spec is not None and spec.granularity != FULL_GRANULARITY
            if do_index and hasattr(eng, "get_rows"):
                continue
            group = shared_get_groups.setdefault(id(eng), {"engine": eng, "keys": [], "refs": []})
            group["keys"].append(key)
            group["refs"].append(ref)
        for group in shared_get_groups.values():
            if len(group["refs"]) < 2:
                continue
            values = group["engine"].get_many(group["refs"])
            batched_shared.update(zip(group["keys"], values, strict=False))

        for k in keys:
            if k not in self.ref_table:
                continue
            ref_or_array = self.ref_table[k]
            token_slice = self.dim0_index.token_slices.get(k)
            if isinstance(ref_or_array, Ref):
                # shared ref: fetch then apply sample index + token slice
                spec = self.schema.get(k)
                eng = _engine_for(ref_or_array)
                do_index = sample_idx is not None and spec is not None and spec.granularity not in (FULL_GRANULARITY,)
                # Row-addressable backends can read only the selected rows
                # instead of fetching the whole [N, ...] block.
                if k in batched_rows:
                    value = batched_rows[k]
                elif k in batched_shared:
                    value = batched_shared[k]
                    if do_index:
                        value = self._index_along(value, 0, sample_idx)
                elif do_index and hasattr(eng, "get_rows"):
                    value = eng.get_rows(ref_or_array, sample_idx)
                else:
                    value = eng.get(ref_or_array)
                    if do_index:
                        value = self._index_along(value, 0, sample_idx)
                if token_slice:
                    from verl.experimental.neoproto.storage.engine import _BaseStorageEngine

                    value = _BaseStorageEngine.apply_slice(value, token_slice)
                out[k] = _move_to_device(value, device)
                continue

            # per-sample refs: gather only the ones the index selects
            # number of refs
            n = len(ref_or_array)
            # number of dim0 index
            if sample_idx is not None:
                positions = sample_idx
            else:
                positions = np.arange(n, dtype=np.int64)

            selected: list[Any] = []
            refs_to_get: list[Ref] = []
            ref_positions: list[int] = []
            shuffle_indices = None
            batch_size = self.ref_table.batch_size
            # Per sample refs
            if n == batch_size:
                for out_i, src_i in enumerate(positions):
                    entry = ref_or_array[int(src_i)]
                    if entry is None:
                        selected.append(None)
                    elif isinstance(entry, Ref):
                        selected.append(None)
                        refs_to_get.append(entry)
                        ref_positions.append(out_i)
                    else:  # raw value already resident
                        selected.append(entry)
            else:
                refs_to_get = []
                # Full batch
                if len(positions) == batch_size:
                    refs_to_get = ref_or_array
                    shuffle_indices = sample_idx
                else:
                    offset = 0
                    miss_samples = 0
                    shuffle_indices = sample_idx.tolist()
                    for idx, ref in enumerate(ref_or_array):
                        is_selected = False
                        nr_samples = ref.shape[0]
                        for inner_idx in range(nr_samples):
                            offset_idx = offset + inner_idx
                            if offset_idx in positions:
                                is_selected = True
                                s_idx = np.where(positions == offset_idx)[0][0]
                                shuffle_indices[s_idx] = offset_idx - miss_samples
                        if is_selected:
                            refs_to_get.append(ref)
                        else:
                            miss_samples += nr_samples
                        offset += nr_samples
                ref_positions = list(range(len(refs_to_get)))
                selected = [None] * len(refs_to_get)

            if len(refs_to_get) > 0:
                values = self.get_many(refs_to_get)
                for pos, val in zip(ref_positions, values, strict=False):
                    if token_slice:
                        from verl.experimental.neoproto.storage.engine import _BaseStorageEngine

                        val = _BaseStorageEngine.apply_slice(val, token_slice)
                    selected[pos] = _move_to_device(val, device)

            # stack for data item
            out[k] = selected if as_ == "list" else _concat_if_possible(selected)

            # select item
            if shuffle_indices is not None:
                out[k] = out[k][shuffle_indices]

        type(self)._materialize_time_s += time.time() - _t0
        type(self)._materialize_calls += 1
        return out

    def get_many(
        self,
        refs: list[Ref],
    ) -> list[Any]:
        # Preserve caller order. Do not key by uid alone: the same payload can
        # appear multiple times with different slice_specs after concat/union.
        results = [None for _ in refs]
        ref_group_by_backend: dict[str, list[Ref]] = {}
        indices_by_backend: dict[str, list[int]] = {}
        for i, r in enumerate(refs):
            if r is None:
                continue
            ref_group_by_backend.setdefault(r.backend, []).append(r)
            indices_by_backend.setdefault(r.backend, []).append(i)
        for backend, ref_group in ref_group_by_backend.items():
            engine = self._engine(backend)
            ref_results = engine.get_many(ref_group)
            for idx, val in zip(indices_by_backend[backend], ref_results, strict=True):
                results[idx] = val
        return results

    def propogate_cache(self, data: NeoProto):
        pass

    def release(self):
        ref_group_by_backend = dict()
        for entry in self.ref_table.values():
            refs = entry if isinstance(entry, list | np.ndarray) else [entry]
            for r in refs:
                if isinstance(r, Ref):
                    if r.backend not in ref_group_by_backend:
                        ref_group_by_backend[r.backend] = []
                    ref_group_by_backend[r.backend].append(r)
        for backend, ref_group in ref_group_by_backend.items():
            if backend == "None":
                continue
            self._engine(backend).release(ref_group)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_to_disk(self, path: str, *, include_data: bool = False) -> None:
        """Persist refs + schema + meta (optionally tensor bytes).

        ``include_data=False``: fast, tiny manifest useful for debug.
        ``include_data=True``: call ``materialize`` and pickle the result
        alongside the manifest (``<path>.data.pkl``).
        """
        manifest = {
            "schema": {k: _fieldspec_to_dict(v) for k, v in self.schema.items()},
            "ref_table": self.ref_table.to_dict(),
            "dim0_index": self.dim0_index.to_dict(),
            "type": type(self).__name__,
        }
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, default=_json_default, indent=2)
        if include_data:
            materialized = self.materialize()
            with open(path + ".data.pkl", "wb") as f:
                pickle.dump(materialized, f)

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        *,
        storage: Optional[StorageEngine] = None,
    ) -> NeoProto:
        with open(path) as f:
            manifest = json.load(f)
        obj = cls(
            schema={k: _fieldspec_from_dict(v) for k, v in manifest["schema"].items()},
            ref_table=RefTable.from_dict(manifest["ref_table"]),
            dim0_index=IndexView.from_dict(manifest["dim0_index"]),
        )
        return obj

    # ------------------------------------------------------------------
    # Init functions
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        batch_dict: dict,
        *,
        storage: Optional[StorageEngine] = None,
    ) -> NeoProto:
        schema = {}
        ref_table = RefTable(batch_size=1)
        for k, v in batch_dict.items():
            if isinstance(v, Ref):
                schema[k] = fieldspec_from_ref(v)
                ref_table[k] = v
            else:
                schema[k] = fieldspec_from_local(v)
                ref_table[k] = Ref.from_local(v)
        obj = cls(
            schema={k: _fieldspec_from_dict(v) for k, v in batch_dict["schema"].items()},
            ref_table=ref_table,
        )
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _with_sample_indices(self, indices: np.ndarray) -> NeoProto:
        new_index = IndexView(
            sample_indices=indices,
            token_slices=dict(self.dim0_index.token_slices),
        )
        refs = self.ref_table.copy()
        new_neo_proto = type(self)(
            schema=dict(self.schema),
            ref_table=refs,
            dim0_index=new_index,
        )
        # self.propogate_cache(new_neo_proto)
        return new_neo_proto

    def _get_cache_name(self, key: str) -> str:
        return f"_{key}_cache"

    def empty_like(self) -> NeoProto:
        return type(self)(
            schema=dict(self.schema),
            ref_table=RefTable(batch_size=0),
            dim0_index=IndexView(),
        )

    @staticmethod
    def _index_along(value: Any, axis: int, indices: np.ndarray) -> Any:
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            writable_indices = np.asarray(indices).copy()
            return value.index_select(axis, torch.as_tensor(writable_indices, dtype=torch.long, device=value.device))
        if isinstance(value, np.ndarray):
            return np.take(value, indices, axis=axis)
        if isinstance(value, list):
            return [value[int(i)] for i in indices]
        return value


def _move_to_device(value: Any, device: Optional[Any], *, non_blocking: bool = False) -> Any:
    if device is None:
        return value
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    return value


def _concat_if_possible(values: list[Any]) -> Any:
    if not values:
        return values
    # CASE 1: all values are torch.Tensor
    if _HAVE_TORCH and all(isinstance(v, torch.Tensor) for v in values):
        # check shape of element
        if len(values[0].shape) == 0:
            if len(values) == 1:
                return values[0]
            else:
                return torch.stack(values, dim=0)
        else:
            return torch.cat(values, dim=0)
    # CASE 2: all values are np.ndarray
    if all(isinstance(v, np.ndarray) for v in values):
        # check shape of element
        if len(values[0].shape) == 0:
            if len(values) == 1:
                return values[0]
            else:
                return np.stack(values, axis=0)
        else:
            return np.concatenate(values, axis=0)
    # CASE 3: all values are list
    if len(values) == 1:
        return values[0]
    else:
        arr = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            arr[i] = v
        return arr


def _fieldspec_to_dict(spec: FieldSpec) -> dict[str, Any]:
    return {
        "dtype": spec.dtype,
        "shape": list(spec.shape),
        "granularity": spec.granularity,
        "token_axis": spec.token_axis,
    }


def _fieldspec_from_dict(data: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        dtype=data.get("dtype", "object"),
        shape=tuple(data.get("shape") or ()),
        granularity=data.get("granularity", "sample"),
        token_axis=data.get("token_axis"),
    )


def fieldspec_from_ref(ref: Ref, granularity: str = "sample") -> FieldSpec:
    token_axis = 0 if len(ref.shape) <= 1 else 1
    return FieldSpec(
        dtype=ref.dtype,
        shape=ref.estimated_shape,
        granularity=granularity,
        token_axis=token_axis,
    )


def fieldspec_from_local(ref: dict) -> FieldSpec:
    return FieldSpec(
        dtype=None,
        shape=None,
        granularity=FULL_GRANULARITY,
        token_axis=-1,
    )


def fieldspec_from_shared(ref: Ref) -> FieldSpec:
    return FieldSpec(
        dtype=None,
        shape=None,
        granularity=FULL_GRANULARITY,
        token_axis=-1,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
