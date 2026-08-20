# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
:class:`DataProto` -- verl-compatible view over :class:`NeoProto`.

This subclass preserves the **public** API of ``verl.protocol.DataProto``
(``batch`` / ``non_tensor_batch`` / ``meta_info``, plus ``from_single_dict``,
``from_dict``, ``select``, ``pop``, ``union``, ``chunk``, ``concat``,
``repeat``, ``__getitem__``) but underneath it is a ref/index-only
:class:`NeoProto`.

Materialization is lazy: ``batch`` and ``non_tensor_batch`` are accessors that
call :meth:`NeoProto.materialize` the first time they're read. Callers
that are part of the pilot path should prefer ``materialize(keys=[...])``
directly to avoid pulling more than they need.

For interop with untouched paths, use :meth:`DataProto.to_verl` at the edge.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from tensordict import TensorDict

from verl.protocol import DataProto as RLDataProto
from verl.protocol import DataProtoConfig

try:  # pragma: no cover
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from verl.experimental.neoproto.neo import FULL_GRANULARITY, IndexView, NeoProto, fieldspec_from_local
from verl.experimental.neoproto.storage.engine import FieldSpec, Ref, RefTable, StorageEngine
from verl.utils.device import get_device_name


def normalize_dtype(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        return str(v.dtype).replace("torch.", "") if _HAVE_TORCH else "float32"
    elif isinstance(v, np.ndarray):
        return str(v.dtype)
    else:
        return "object"


def is_refcontainer(v: Any) -> bool:
    if isinstance(v, Ref):
        return True
    elif isinstance(v, np.ndarray) or isinstance(v, list):
        return isinstance(v[0], Ref)
    else:
        return False


def is_identity(arr):
    return arr is None or arr.size == 0 or np.array_equal(arr, np.arange(arr.shape[0]))


def _is_permutation(arr, n: int) -> bool:
    """True iff ``arr`` is a permutation of ``0..n-1`` (safe for inverse shuffle)."""
    if arr is None or len(arr) != n:
        return False
    return bool(arr.min() == 0 and arr.max() == n - 1 and len(set(arr.tolist())) == n)


INDEX_KEY = "_sample_indices"
_INLINE_TENSOR_MAX_BYTES = 64 * 1024

# These payloads dominate AgentLoop RPC size and are naturally shared across
# chunk views. Small reward/control columns stay inline: reward workers access
# them per sample, where a Ray round-trip would cost more than serialization.
_OBJECT_STORE_NON_TENSOR_KEYS = frozenset(
    {
        "images",
        "videos",
        "audios",
        "image_data",
        "video_data",
        "audio_data",
    }
)


def _non_tensor_backend(key: str) -> str:
    return "object_store" if key in _OBJECT_STORE_NON_TENSOR_KEYS else "local"


def _batch_tensor_backend(value: Any) -> str:
    """Inline tiny constructor tensors; keep training-sized payloads in Ray."""
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        nbytes = value.numel() * value.element_size()
    elif isinstance(value, np.ndarray):
        nbytes = value.nbytes
    else:
        return "object_store"
    return "local" if nbytes <= _INLINE_TENSOR_MAX_BYTES else "object_store"


def _is_full_key(owner: NeoProto, key: str) -> bool:
    spec = owner.schema.get(key)
    return spec is not None and spec.granularity == FULL_GRANULARITY


class LocalProxy(MutableMapping):
    """Live, dict-like view over a proto's batch-level (``"full"``) ref entries.

    Reads/writes the underlying ``ref_table`` / ``schema`` directly, so in-place
    mutation (``meta_info[k] = v``, ``.update(...)``, ``.pop(...)``) persists.
    ``copy``/``deepcopy`` snapshot to a plain ``dict`` so callers that do
    ``meta_info = copy.deepcopy(meta_info)`` get detached data.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: NeoProto) -> None:
        self._owner = owner

    def __getitem__(self, key: str) -> Any:
        if not _is_full_key(self._owner, key):
            raise KeyError(key)
        return self._owner.ref_table[key].dataptr

    def __setitem__(self, key: str, value: Any) -> None:
        o = self._owner
        o.schema[key] = fieldspec_from_local(value)
        o.ref_table[key] = Ref.from_local(value)

    def __delitem__(self, key: str) -> None:
        if not _is_full_key(self._owner, key):
            raise KeyError(key)
        self._owner.schema.pop(key, None)
        self._owner.ref_table.pop(key, None)

    def __iter__(self):
        return iter([k for k in self._owner.ref_table.keys() if _is_full_key(self._owner, k)])

    def __len__(self) -> int:
        return sum(1 for k in self._owner.ref_table.keys() if _is_full_key(self._owner, k))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and _is_full_key(self._owner, key)

    def __copy__(self) -> dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo) -> dict[str, Any]:
        import copy as _copy

        return {k: _copy.deepcopy(v, memo) for k, v in self.items()}

    def __repr__(self) -> str:
        return f"LocalProxy({dict(self)!r})"


# ---------------------------------------------------------------------------
# DataProtoItem (kept for backward compat)
# ---------------------------------------------------------------------------


@dataclass
class DataProtoItem:
    batch: Any = None
    non_tensor_batch: dict[str, Any] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataProto
# ---------------------------------------------------------------------------


@dataclass
class DataProto(NeoProto, RLDataProto):
    """verl-compatible view over :class:`NeoProto`.

    The public surface matches ``verl.protocol.DataProto``; see module
    docstring for details.
    """

    _batch_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    _non_tensor_batch_cache: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        batch: Any = None,
        non_tensor_batch: Optional[dict[str, Any]] = None,
        meta_info: Optional[dict[str, Any]] = None,
        *,
        schema: Optional[dict[str, FieldSpec]] = None,
        ref_table: Optional[RefTable] = None,
        dim0_index: Optional[IndexView] = None,
    ) -> None:
        NeoProto.__init__(
            self,
            schema={} if schema is None else schema,
            ref_table=RefTable(batch_size=0) if ref_table is None else ref_table,
            dim0_index=dim0_index,
        )
        self.__post_init__()
        # schema / ref_table now exist -> apply RLDataProto-shaped fields safely.
        if batch is not None:
            self.batch = batch
        if non_tensor_batch:
            self.non_tensor_batch = non_tensor_batch
        if meta_info:
            self.meta_info = meta_info

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self, "_batch_cache"):
            self._batch_cache = dict(self._batch_cache or {})
        else:
            self._batch_cache = {}
        if hasattr(self, "_non_tensor_batch_cache"):
            self._non_tensor_batch_cache = dict(self._non_tensor_batch_cache or {})
        else:
            self._non_tensor_batch_cache = {}

    # ------------------------------------------------------------------
    # Public verl-compatible properties
    # ------------------------------------------------------------------
    @property
    def meta_info(self) -> dict[str, Any]:
        return LocalProxy(self)

    @meta_info.setter
    def meta_info(self, value: dict[str, Any]) -> None:
        if isinstance(value, LocalProxy):
            self._set_proxy(value)
            return
        ref_table = getattr(self, "ref_table", {})
        if value is None:
            for k in ref_table.keys():
                ref_table.pop(k, None)
            return
        for k, v in value.items():
            spec = self.schema.get(k, fieldspec_from_local(v))
            self.schema[k] = spec
            ref_table[k] = Ref.from_local(v)

    def _set_proxy(self, value: Any):
        # TODO: Add more assertion protection:
        # Handle the case where value is a _BatchProxy & real data case.
        if isinstance(value, _BatchProxy):
            check_consistency = True
            value_keys = value._owner.ref_table.keys()
        else:
            check_consistency = False
            value_keys = value.keys()

        if self.ref_table is None:
            self.ref_table = value._owner.ref_table
        else:
            self.ref_table.update(value._owner.ref_table, value_keys, check_consistency)
        if self.schema is None:
            self.schema = value._owner.schema
        else:
            for k in value_keys:
                self.schema[k] = value._owner.schema[k]
        if isinstance(value, _BatchProxy):
            self.dim0_index = value._owner.dim0_index

    @property
    def batch(self) -> Any:
        """Lazy view of tensor-typed fields as a dict-like.

        Returns a :class:`_BatchProxy` that defers to :meth:`materialize`.
        """
        if hasattr(self, "batch_proxy"):
            return self.batch_proxy
        proxy = _BatchProxy(self, tensor=True)
        self.batch_proxy = proxy
        return proxy

    @batch.setter
    def batch(self, value: Any) -> None:
        # Accept a TensorDict / dict and store each tensor via storage.
        if value is None:
            # TODO release real data
            return
        engine = self._engine()
        # value is a _BatchProxy, use its refs and schema
        if isinstance(value, _BatchProxy):
            self._set_proxy(value)
            return
        bs = -1
        items = value.items() if hasattr(value, "items") else value
        for k, v in items:
            v_bs = self._cal_bs_size(v)
            if bs == -1:
                bs = v_bs
            assert bs == v_bs, f"Batch size mismatch: {bs} != {v_bs}"
            spec = self.schema.get(
                k,
                FieldSpec(
                    dtype=normalize_dtype(v),
                    shape=tuple(v.shape),
                    token_axis=1 if v.dim() > 1 else None,
                ),
            )
            self.schema[k] = spec
            self.ref_table[k] = engine.put(
                v,
                key_hint=k,
                spec=spec,
                backend=_batch_tensor_backend(v),
            )
        if bs < 0 and hasattr(value, "batch_size") and len(value.batch_size) > 0:
            bs = int(value.batch_size[0])
        if bs >= 0:
            self.ref_table.batch_size = bs
        if self.dim0_index.sample_indices is None:
            self.dim0_index = self.dim0_index.identity(self.ref_table.batch_size)

    @property
    def non_tensor_batch(self) -> dict[str, Any]:
        """Lazy view of non-tensor fields."""
        if hasattr(self, "non_tensor_proxy"):
            return self.non_tensor_proxy
        proxy = _BatchProxy(self, tensor=False)
        self.non_tensor_proxy = proxy
        return proxy

    @non_tensor_batch.setter
    def non_tensor_batch(self, value: dict[str, Any]) -> None:
        if value is None:
            return
        # value is a _BatchProxy, use its refs and schema
        if isinstance(value, _BatchProxy):
            self._set_proxy(value)
            return
        # Special case: "_sample_indices" is a reserved key for index manipulation.
        if len(value) == 1 and INDEX_KEY in value:
            assert len(value[INDEX_KEY]) == len(self.dim0_index.sample_indices), (
                f"Dim0 index mismatch: {self.dim0_index.sample_indices} != {value[INDEX_KEY]}"
            )
        elif len(value) > 0:
            engine = self._engine()
            bs = -1
            for k, v in value.items():
                if not isinstance(v, np.ndarray):
                    v = np.asarray(v, dtype=object)
                granularity = "sample"
                v_bs = self._cal_bs_size(v)
                if bs == -1:
                    bs = v_bs
                assert bs == v_bs, f"Batch size mismatch: {bs} != {v_bs}"
                # Always object: agent_loop often builds non_tensors via
                # ``np.array([...])`` without dtype=object (e.g. ``<U*`` / int64).
                # Those must still be non-tensor keys for reward / uid lookups.
                spec = FieldSpec(
                    dtype="object",
                    shape=tuple(getattr(v, "shape", ())),
                    granularity=granularity,
                )
                self.schema[k] = spec
                self.ref_table[k] = engine.put(
                    v,
                    key_hint=k,
                    spec=spec,
                    backend=_non_tensor_backend(k),
                )
            if bs > 0:
                self.ref_table.batch_size = bs

    # ------------------------------------------------------------------
    # verl API parity
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors: Optional[dict[str, Any]] = None,
        meta_info: Optional[dict[str, Any]] = None,
        num_batch_dims: int = 1,
        auto_padding: bool = False,
        *,
        granularity: Optional[str] = "shared",
        dim0_index: Optional[IndexView] = None,
        storage: Optional[StorageEngine] = None,
    ) -> DataProto:
        assert num_batch_dims == 1, "NeoProto-backed DataProto supports exactly one batch dimension"
        tensors = tensors or {}
        non_tensors = non_tensors or {}
        meta_info = dict(meta_info or {})

        inst = cls()
        if storage is not None:
            engine = storage
        else:
            engine = inst._engine()

        batch_size: Optional[int] = None
        for k, v in tensors.items():
            if isinstance(v, Ref):
                inst.add_ref(k, v, granularity=granularity)
            else:
                # TODO: only materialize for token-based tensors
                if batch_size is None and hasattr(v, "shape") and len(v.shape) >= num_batch_dims:
                    batch_size = int(v.shape[0])
                elif hasattr(v, "shape") and len(v.shape) >= num_batch_dims:
                    assert batch_size == int(v.shape[0]), (
                        f"Not all tensors have the same batch size: {batch_size} != {int(v.shape[0])} for {k}"
                    )
                spec = FieldSpec(
                    dtype=normalize_dtype(v),
                    shape=tuple(v.shape) if hasattr(v, "shape") else (),
                    token_axis=1 if hasattr(v, "shape") and len(v.shape) > 1 else None,
                    granularity=granularity,
                )
                inst.schema[k] = spec
                if granularity == "sample" and v.shape[0] > 1:
                    ref_list = engine.put_many([v[i].unsqueeze(0) for i in range(v.shape[0])], key_hint=k, spec=spec)
                    refs = np.empty(len(v), dtype=object)
                    for i, r in enumerate(ref_list):
                        refs[i] = r
                else:
                    refs = engine.put(v, key_hint=k, spec=spec)
                inst.ref_table[k] = refs

        for k, v in non_tensors.items():
            if isinstance(v, Ref):
                inst.add_ref(k, v, granularity=granularity)
            else:
                if not isinstance(v, np.ndarray):
                    v = np.asarray(v, dtype=object)
                # TODO: only materialize for token-based np array
                if batch_size is None and hasattr(v, "shape") and len(v.shape) >= 1:
                    batch_size = int(v.shape[0])
                elif hasattr(v, "shape") and len(v.shape) >= 1:
                    assert batch_size == int(v.shape[0]), (
                        f"Tensor and non-tensor batch sizes differ: {batch_size} != {int(v.shape[0])} for {k}"
                    )
                spec = FieldSpec(
                    dtype="object",
                    shape=tuple(getattr(v, "shape", ())),
                    granularity=granularity,
                )
                inst.schema[k] = spec
                if granularity == "sample" and v.shape[0] > 1:
                    refs = np.empty(len(v), dtype=object)
                    for i in range(v.shape[0]):
                        refs[i] = engine.put(
                            v[i, ...],
                            key_hint=k,
                            spec=spec,
                            backend=_non_tensor_backend(k),
                        )
                else:
                    refs = engine.put(v, key_hint=k, spec=spec, backend=_non_tensor_backend(k))
                inst.ref_table[k] = refs

        for k, v in meta_info.items():
            inst.add_ref(k, Ref.from_local(v), granularity=FULL_GRANULARITY)

        inst.ref_table.batch_size = batch_size or 0
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        inst.meta_info = meta_info
        if dim0_index is None:
            inst.dim0_index = IndexView().identity(inst.ref_table.batch_size)
        else:
            inst.dim0_index = dim0_index
        return inst

    @classmethod
    def from_tensordict(cls, tensor_dict: TensorDict = None, meta_info=None, num_batch_dims: int = 1) -> DataProto:
        """Create a NeoProto-backed DataProto from a TensorDict (verl trainer edge)."""
        from tensordict import NonTensorData, NonTensorStack

        if meta_info is None:
            meta_info = {}
        tensors: dict[str, Any] = {}
        non_tensors: dict[str, Any] = {}
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, NonTensorStack):
                non_tensors[key] = np.array([elem.data for elem in val], dtype=object)
            elif isinstance(val, NonTensorData):
                meta_info[key] = val.data
        return cls.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=meta_info,
            num_batch_dims=num_batch_dims,
        )

    def to_tensordict(self) -> TensorDict:
        """Materialize into a TensorDict for trainer edges that still expect it."""
        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from verl.utils import tensordict_utils as tu

        tensors: dict[str, Any] = {k: self.batch[k] for k in self.batch.keys()}
        non_tensors = {k: self.non_tensor_batch[k] for k in self.non_tensor_batch.keys()}
        for key, val in non_tensors.items():
            if isinstance(val, list):
                arr = np.empty(len(val), dtype=object)
                for i, item in enumerate(val):
                    arr[i] = item
                val = arr
            assert isinstance(val, np.ndarray)
            tensors[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
        return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=dict(self.meta_info))

    def is_padding_enabled(self) -> bool:
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return bool(dataproto_specific_padding or DataProtoConfig.auto_padding)

    def padding(self, padding_size=0, padding_candidate=""):
        """Append repeated samples in-place (verl.protocol.DataProto.padding semantics)."""
        if padding_size == 0:
            return
        idx = 0 if padding_candidate == "first" else len(self) - 1
        padding_part = self.select_idxs([idx]).repeat(padding_size)
        padded = type(self).concat([self, padding_part])
        self.schema = padded.schema
        self.ref_table = padded.ref_table
        self.dim0_index = padded.dim0_index
        self.clear_cache()
        # Drop cached proxies so subsequent batch access reflects the new refs.
        for attr in ("batch_proxy", "non_tensor_proxy"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _rebind_to_logical_identity(self) -> DataProto:
        """Push current sample view into refs and reset index to identity.

        Needed when assigning a logical-length field after ``repeat``/gather
        views whose ``sample_indices`` are not a permutation of ``0..len-1``.
        Returns ``self`` so callers (e.g. ``NeoProto.union``) can chain.
        """
        sample_indices = self.dim0_index.sample_indices
        if sample_indices is None or is_identity(sample_indices):
            return self
        n = int(len(sample_indices))
        new_refs: dict[str, Any] = {}
        for key, entry in self.ref_table.items():
            if _is_full_key(self, key):
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
        self.ref_table = RefTable(new_refs, batch_size=n)
        self.dim0_index = IndexView().identity(n)
        self.clear_cache()
        for attr in ("batch_proxy", "non_tensor_proxy"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    @classmethod
    def from_single_dict(
        cls,
        data: dict[str, Any],
        meta_info: Optional[dict[str, Any]] = None,
        auto_padding: bool = False,
        *,
        storage: Optional[StorageEngine] = None,
    ) -> DataProto:
        tensors: dict[str, Any] = {}
        non_tensors: dict[str, Any] = {}
        for k, v in data.items():
            if _HAVE_TORCH and isinstance(v, torch.Tensor):
                tensors[k] = v
            elif isinstance(v, np.ndarray) and v.dtype != object:
                tensors[k] = v
            else:
                non_tensors[k] = v
        return cls.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=meta_info,
            auto_padding=auto_padding,
            storage=storage,
        )

    # ------------------------------------------------------------------
    # select / pop signatures mirror verl's keyword args
    # ------------------------------------------------------------------
    def select(  # type: ignore[override]
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
        deepcopy: bool = False,
    ) -> DataProto:
        tensor_keys = [
            k
            for k in self.ref_table
            if not _is_full_key(self, k) and self.schema.get(k) is not None and self.schema[k].dtype != "object"
        ]
        non_tensor_keys = [
            k
            for k in self.ref_table
            if not _is_full_key(self, k) and self.schema.get(k) is not None and self.schema[k].dtype == "object"
        ]
        meta_keys = [k for k in self.ref_table if _is_full_key(self, k)]
        keys = (
            (tensor_keys if batch_keys is None else list(batch_keys))
            + (non_tensor_keys if non_tensor_batch_keys is None else list(non_tensor_batch_keys))
            + (meta_keys if meta_info_keys is None else list(meta_info_keys))
        )
        view = NeoProto.select(self, keys, deepcopy_meta=deepcopy)
        return view  # type: ignore[return-value]

    def select_idxs(self, idxs, deepcopy: bool = False) -> DataProto:
        view = NeoProto.select_idxs(self, idxs, deepcopy_meta=deepcopy)
        return view

    def reorder(self, idxs: list[int]):
        NeoProto.reorder(self, idxs)

    def clear_cache(self):
        if hasattr(self, "_batch_cache"):
            self._batch_cache.clear()
        if hasattr(self, "_non_tensor_batch_cache"):
            self._non_tensor_batch_cache.clear()

    def prepare_dispatch(self, chunks) -> None:
        """Attach rank-local ref tables before Ray serializes chunked views."""
        import ray

        if not ray.is_initialized():
            return
        from verl.experimental.neoproto.dispatch import attach_preserialized_ref_tables

        attach_preserialized_ref_tables(self, chunks)

    def prepare_engine_batch(self, spec):
        """Materialize this ref-backed batch for a worker engine call."""
        from verl.experimental.neoproto.worker_bridge import prepare_neo_engine_batch

        return prepare_neo_engine_batch(self, spec)

    def prepare_worker_request(self, **control_fields: Any) -> DataProto:
        """Create an RPC-local ref view without mutating the long-lived batch."""
        request = self.select()
        request.set_control_fields(**control_fields)
        return request

    @staticmethod
    def collect_worker_output(output: Any, key_map: dict[str, str]) -> DataProto:
        """Select and rename ref-backed worker fields without driver materialization."""
        ref_table = getattr(output, "ref_table", None)
        source_keys = [key for key in key_map if ref_table is not None and key in ref_table]
        if not source_keys:
            raise RuntimeError(
                "NeoProto received an empty or non-Neo worker payload; "
                f"expected one of {sorted(key_map)}, got {type(output)!r}"
            )

        selected = output.select(batch_keys=source_keys, non_tensor_batch_keys=[], meta_info_keys=[])
        old_keys = [key for key in source_keys if key_map[key] != key]
        new_keys = [key_map[key] for key in old_keys]
        return selected.rename(old_keys=old_keys, new_keys=new_keys) if old_keys else selected

    def cpu(self) -> DataProto:
        """NeoProto payload refs already resolve to CPU tensors on workers."""
        return self

    def prefetch(self, keys: Optional[list[str]] = None, *, device: str = "cpu") -> dict[str, Any]:
        """Materialize ``keys`` once and seed the lazy ``.batch`` cache.

        Driver hot paths (adv / metrics) still read ``batch.batch[k]``; without
        prefetch each key pays a separate storage get. Prefer this over
        key-by-key access when several tensors are needed together.
        """
        if keys is None:
            keys = [
                k
                for k in self.ref_table.keys()
                if not _is_full_key(self, k) and self.schema.get(k) is not None and self.schema[k].dtype != "object"
            ]
        else:
            keys = [k for k in keys if k in self.ref_table]
        if not keys:
            return {}

        # Seed the same cache ``_BatchProxy.__getitem__`` consults (full-batch key).
        if not hasattr(self, "_batch_cache"):
            self._batch_cache = {}
        if not hasattr(self, "_non_tensor_batch_cache"):
            self._non_tensor_batch_cache = {}

        requested_device = torch.device(device) if _HAVE_TORCH else None
        materialized: dict[str, Any] = {}
        missing: list[str] = []
        for key in keys:
            is_non_tensor = self.schema.get(key) is not None and self.schema[key].dtype in ("object", "bytes")
            cache = self._non_tensor_batch_cache if is_non_tensor else self._batch_cache
            cache_key = self.batch.get_cache_key(key)
            cached = cache.get(cache_key)
            device_matches = (
                not (_HAVE_TORCH and isinstance(cached, torch.Tensor))
                or requested_device is None
                or cached.device == requested_device
            )
            if cached is not None and device_matches:
                materialized[key] = cached
            else:
                missing.append(key)

        if missing:
            fresh = self.materialize(keys=missing, device=device)
            for key, value in fresh.items():
                is_non_tensor = self.schema.get(key) is not None and self.schema[key].dtype in ("object", "bytes")
                cache = self._non_tensor_batch_cache if is_non_tensor else self._batch_cache
                cache[self.batch.get_cache_key(key)] = value
            materialized.update(fresh)
        return materialized

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Preserve the legacy iterator API while yielding Neo-backed batches."""
        legacy_iterator = super().make_iterator(
            mini_batch_size=mini_batch_size,
            epochs=epochs,
            seed=seed,
            dataloader_kwargs=dataloader_kwargs,
        )
        return (type(self).from_verl(batch) for batch in legacy_iterator)

    @staticmethod
    def concat(items: list[NeoProto], new_index: bool = False) -> DataProto:
        from verl.protocol import list_of_dict_to_dict_of_list

        output = NeoProto.concat(items, new_index=new_index)
        all_metrics = []
        merged_meta: dict[str, Any] = {}
        for item in items:
            meta = getattr(item, "meta_info", None) or {}
            for key, value in meta.items():
                if key == "metrics":
                    if value is None:
                        continue
                    if isinstance(value, list):
                        all_metrics.extend(value)
                    else:
                        all_metrics.append(value)
                elif key in merged_meta:
                    assert merged_meta[key] == value, f"Conflicting values for meta_info key '{key}'"
                else:
                    merged_meta[key] = value
        output.meta_info.update(merged_meta)
        if all_metrics:
            output.meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)
        return output

    def save_to_disk(self, filepath) -> None:
        """Persist using the historical DataProto pickle API."""
        import pickle

        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_disk(filepath) -> DataProto:
        """Load a batch written by :meth:`save_to_disk`."""
        import pickle

        with open(filepath, "rb") as file:
            return pickle.load(file)

    def dynamic_chunk(self, chunk_size: int) -> DataProto:
        return NeoProto.chunk(self, chunk_size)

    def propogate_cache(self, data: DataProto):
        if hasattr(self, "_batch_cache"):
            if not hasattr(data, "_batch_cache"):
                data._batch_cache = {}
            for k, v in self._batch_cache.items():
                data._batch_cache[k] = v
        if hasattr(self, "_non_tensor_batch_cache"):
            if not hasattr(data, "_non_tensor_batch_cache"):
                data._non_tensor_batch_cache = {}
            for k, v in self._non_tensor_batch_cache.items():
                data._non_tensor_batch_cache[k] = v

    def union(self, other: NeoProto) -> DataProto:  # type: ignore[override]
        """Union refs while retaining caches that remain logically aligned."""
        output = NeoProto.union(self, other)
        output_indices = output.dim0_index.sample_indices

        for source in (self, other):
            source_indices = source.dim0_index.sample_indices
            if source_indices is None or output_indices is None or not np.array_equal(source_indices, output_indices):
                continue
            for key in source.ref_table.keys():
                if key not in output.ref_table or _is_full_key(source, key):
                    continue
                is_non_tensor = source.schema.get(key) is not None and source.schema[key].dtype in ("object", "bytes")
                source_cache = (
                    getattr(source, "_non_tensor_batch_cache", {})
                    if is_non_tensor
                    else getattr(source, "_batch_cache", {})
                )
                output_cache = output._non_tensor_batch_cache if is_non_tensor else output._batch_cache
                source_key = source.batch.get_cache_key(key)
                if source_key in source_cache:
                    output_key = output.batch.get_cache_key(key)
                    output_cache[output_key] = source_cache[source_key]
        return output  # type: ignore[return-value]

    def pop(  # type: ignore[override]
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
    ) -> DataProto:
        keys = list(batch_keys or []) + list(non_tensor_batch_keys or []) + list(meta_info_keys or [])
        popped = NeoProto.pop(self, keys)
        return popped  # type: ignore[return-value]

    def rename(self, old_keys=None, new_keys=None) -> DataProto:  # type: ignore[override]
        if old_keys is None and new_keys is None:
            return self
        if isinstance(old_keys, str):
            old_keys = [old_keys]
        if isinstance(new_keys, str):
            new_keys = [new_keys]
        mapping = dict(zip(old_keys or [], new_keys or [], strict=False))
        return NeoProto.rename(self, mapping)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Mapping / TensorDict-ish helpers (needed by tu.get / membership tests)
    # ------------------------------------------------------------------
    def __contains__(self, key: object) -> bool:
        # Without this, ``"x" in data`` falls back to iterating samples via
        # ``__getitem__(i)`` (Sequence protocol) and can crash / hang.
        return isinstance(key, str) and key in self.ref_table

    def keys(self):
        return dict.fromkeys(self.ref_table.keys()).keys()

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.ref_table:
            return default
        if _is_full_key(self, key):
            return self.meta_info[key]
        if self.schema.get(key) is not None and self.schema[key].dtype in ("object", "bytes"):
            return self.non_tensor_batch[key]
        return self.batch[key]

    # ------------------------------------------------------------------
    # __getitem__: mirror verl's int -> DataProtoItem behaviour
    # ------------------------------------------------------------------
    def __getitem__(  # type: ignore[override]
        self,
        key: int | slice | list[int] | np.ndarray | torch.Tensor,
    ):
        # Single integer: return DataProtoItem (same contract as verl.protocol.DataProto).
        # Reward managers do ``data[0].batch["responses"].shape[-1]`` and
        # ``data[0].non_tensor_batch["reward_model"]["ground_truth"]``; the old
        # path over-squeezed tensor refs to 0-d scalars, which raised IndexError
        # and was swallowed as score=0 by the limited reward manager.
        if isinstance(key, int | np.integer):
            row_index = int(key)
            if row_index < 0:
                row_index += len(self)
            if row_index < 0 or row_index >= len(self):
                raise IndexError(f"DataProto index {int(key)} out of range for batch of size {len(self)}")
            # ``DataProtoItem`` eagerly exposes every visible column, and reward
            # managers iterate over all rows. Resolve the parent chunk once so
            # shared refs do not pay one storage get per field *per sample*.
            batch_keys = list(self.batch.keys())
            non_tensor_keys = list(self.non_tensor_batch.keys())
            self.prefetch(batch_keys + non_tensor_keys)
            row = NeoProto.__getitem__(self, row_index)  # length-1 sample view
            # Integer indexing must preserve any batch-level prefetch. Reward
            # workers slice a prefetched chunk into individual samples; without
            # inheriting the cache, every sample fetches the same columns again.
            self._inherit_caches(row, row_index)
            tensor_data = None
            if batch_keys:
                tensors = {k: row.batch[k][0] for k in batch_keys}
                tensor_data = TensorDict(source=tensors, batch_size=[])
            non_tensor_data: dict[str, Any] = {}
            for k in non_tensor_keys:
                v = row.non_tensor_batch[k]
                if isinstance(v, np.ndarray):
                    non_tensor_data[k] = v.item() if v.ndim == 0 else v[0]
                elif isinstance(v, list | tuple):
                    non_tensor_data[k] = v[0]
                else:
                    non_tensor_data[k] = v
            return DataProtoItem(
                batch=tensor_data,
                non_tensor_batch=non_tensor_data,
                meta_info=dict(self.meta_info) if self.meta_info is not None else {},
            )

        single = NeoProto.__getitem__(self, key) if isinstance(key, slice) else self.select_idxs(key)
        self._inherit_caches(single, key)
        return single

    def _inherit_caches(self, target: Any, key: Any) -> None:
        target_batch = getattr(target, "_batch_cache", None)
        target_non_tensor = getattr(target, "_non_tensor_batch_cache", None)
        if target_batch is None:
            target._batch_cache = {}
            target_batch = target._batch_cache
        if target_non_tensor is None:
            target._non_tensor_batch_cache = {}
            target_non_tensor = target._non_tensor_batch_cache

        if isinstance(key, int | np.integer):
            n = len(self)
            idx = int(key) if int(key) >= 0 else int(key) + n
            sel: Any = slice(idx, idx + 1)
        else:
            sel = key

        for key in self.ref_table.keys():
            if _is_full_key(self, key):
                continue
            is_non_tensor = self.schema.get(key) is not None and self.schema[key].dtype in ("object", "bytes")
            src = self._non_tensor_batch_cache if is_non_tensor else self._batch_cache
            dst = target_non_tensor if is_non_tensor else target_batch
            src_key = self.batch.get_cache_key(key)
            if src_key not in src:
                continue
            dst_key = target.batch.get_cache_key(key)
            dst[dst_key] = src[src_key][sel]

    # ------------------------------------------------------------------
    # Interop: verl <-> neo
    # ------------------------------------------------------------------
    @classmethod
    def from_verl(
        cls,
        data: Any,
        *,
        storage: Optional[StorageEngine] = None,
    ) -> DataProto:
        """Wrap an existing verl ``DataProto`` into a ref-only Neo view.

        The incoming payload is ``put`` into ``storage`` and only refs are
        retained going forward. Safe to call at the driver/worker boundary.
        """
        tensors: dict[str, Any] = {}
        non_tensors: dict[str, Any] = {}
        if getattr(data, "batch", None) is not None:
            for k in data.batch.keys():
                tensors[k] = data.batch[k]
        for k, v in (getattr(data, "non_tensor_batch", None) or {}).items():
            non_tensors[k] = v
        meta_info = dict(getattr(data, "meta_info", None) or {})
        return cls.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=meta_info,
            storage=storage,
        )

    def to_verl(self) -> Any:
        """Materialize into a classic verl ``DataProto`` (TensorDict-backed).

        Used at boundaries that still expect the legacy DataProto shape.
        """
        import tensordict as _td

        from verl.protocol import DataProto as MonoDataProto

        tensors: dict[str, Any] = {}
        for k in self.batch.keys(original=True):
            tensors[k] = self.batch[k]

        non_tensors: dict[str, Any] = {}
        for k in self.non_tensor_batch.keys(original=True):
            v = self.non_tensor_batch[k]
            # Classic verl DataProto requires non_tensor_batch values to be
            # object-dtype numpy arrays; wrap plain lists accordingly.
            if isinstance(v, list):
                arr = np.empty(len(v), dtype=object)
                for i, item in enumerate(v):
                    arr[i] = item
                v = arr
            non_tensors[k] = v

        meta_info: dict[str, Any] = {}
        for k in self.meta_info:
            meta_info[k] = self.meta_info[k]

        bsz = len(self)
        batch = None
        if tensors:
            batch = _td.TensorDict(source=tensors, batch_size=[bsz])
        return MonoDataProto(
            batch=batch,
            non_tensor_batch=non_tensors,
            meta_info=dict(self.meta_info),
        )


# ---------------------------------------------------------------------------
# _BatchProxy
# ---------------------------------------------------------------------------


class _BatchProxy:
    """Dict-like proxy that lazily materializes a subset of keys.

    This keeps the pilot path safe: the driver can still write code like
    ``batch.batch['input_ids']``, but doing so now pays for a ``get`` via the
    storage engine. Workers should prefer ``data.materialize(keys=...)``.
    """

    __slots__ = ("_owner", "_tensor", "_device")

    def __init__(self, owner: DataProto, tensor: bool, device: str = "cpu"):
        self._owner = owner
        self._tensor = tensor
        self._device = device

    def _is_tensor_key(self, key: str) -> bool:
        spec = self._owner.schema.get(key)
        if spec is None:
            return self._tensor  # unknown: trust the proxy's side
        return spec.dtype not in ("object", "bytes")

    def _visible_keys(self) -> list[str]:
        return [
            k
            for k in self._owner.ref_table.keys()
            if not _is_full_key(self._owner, k) and self._is_tensor_key(k) == self._tensor
        ]

    def keys(self, original: bool = True):
        # verl trainer paths use ``set & batch.keys()``; return a KeysView-like object.
        del original
        return dict.fromkeys(self._visible_keys()).keys()

    def items(self, original: bool = True):
        del original
        keys = self._visible_keys()
        if keys:
            self._owner.prefetch(keys, device=self._device)
        return [(k, self[k]) for k in keys]

    def values(self):
        return [value for _, value in self.items()]

    def update(self, other) -> None:
        """Dict-compatible bulk assign used by trainer paths (e.g. ``_get_gen_batch``)."""
        if other is None:
            return
        if isinstance(other, _BatchProxy):
            items = other.items()
        elif isinstance(other, dict):
            items = other.items()
        else:
            items = dict(other).items()
        for key, value in items:
            self[key] = value

    def contiguous(self) -> _BatchProxy:
        return self

    def cuda(self, device=None) -> _BatchProxy:
        device_name = get_device_name()
        self._device = device_name if device is None else f"{device_name}:{device}"
        return self

    def to(self, device: str = "cpu") -> _BatchProxy:
        self._device = device
        return self

    @property
    def device(self) -> str:
        return self._device

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._owner.dim0_index.sample_indices)

    def __contains__(self, key: str) -> bool:
        return key in self._visible_keys()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for key, expected in other.items():
            actual = self[key]
            if _HAVE_TORCH and isinstance(actual, torch.Tensor):
                if not (isinstance(expected, torch.Tensor) and torch.equal(actual, expected)):
                    return False
            elif isinstance(actual, np.ndarray):
                if not np.array_equal(actual, expected):
                    return False
            elif actual != expected:
                return False
        return True

    def get_cache_key(self, key: str) -> str:
        if len(self._owner.dim0_index.sample_indices) == self._owner.ref_table.batch_size:
            return key
        else:
            return key + str(self._owner.dim0_index.sample_indices.sum())

    def __getitem__(self, key) -> Any:
        if key == INDEX_KEY:
            return self._owner.dim0_index.sample_indices
        elif isinstance(key, slice):
            return self.slice(key.start, key.stop, key.step)
        elif key not in self._owner.ref_table:
            raise KeyError(key)
        is_tensor_key = self._is_tensor_key(key)
        if is_tensor_key != self._tensor:
            raise KeyError(f"Key {key} is not a tensor key ({is_tensor_key})")

        cache_key = self.get_cache_key(key)
        if self._tensor:
            if cache_key in self._owner._batch_cache:
                return self._owner._batch_cache[cache_key]
        else:
            if cache_key in self._owner._non_tensor_batch_cache:
                return self._owner._non_tensor_batch_cache[cache_key]
        value = self._owner.materialize([key], device=self._device)[key]
        # Save to cache if it's a tensor key or non-tensor key
        cache_key = self.get_cache_key(key)
        if self._tensor:
            self._owner._batch_cache[cache_key] = value
        else:
            self._owner._non_tensor_batch_cache[cache_key] = value
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if key == INDEX_KEY:
            self._owner.dim0_index.sample_indices = value
            return
        if value is None:
            self.pop(key, materialize=False)
            return
        engine = self._owner._engine()
        spec = self._owner.schema.get(key)
        bs = self._owner._cal_bs_size(value)
        sample_indices = self._owner.dim0_index.sample_indices
        if sample_indices is None:
            sample_indices = np.arange(bs, dtype=np.int64)
        assert bs == len(sample_indices), f"Batch size {bs} does not match sample indices size {len(sample_indices)}"
        # Storage is always in physical row order; ``materialize`` re-applies
        # ``sample_indices`` to present a logical view. The lazy cache must
        # store that same logical view — never the physical buffer — because
        # ``__getitem__`` returns cache hits as-is (no index re-apply).
        logical_value = value
        physical_value = value
        if not is_identity(sample_indices):
            if _is_permutation(sample_indices, bs):
                shuffle_indices = np.empty(bs, dtype=sample_indices.dtype)
                shuffle_indices[sample_indices] = np.arange(bs)
                physical_value = value[shuffle_indices]
            else:
                self._owner._rebind_to_logical_identity()
                sample_indices = self._owner.dim0_index.sample_indices
                logical_value = value
                physical_value = value

        # update batch_size if it's not set yet
        if self._owner.ref_table.batch_size == 0:
            self._owner.ref_table.batch_size = bs

        shape = tuple(getattr(physical_value, "shape", ()))

        is_refs = is_refcontainer(physical_value)
        if spec is None or not is_refs:
            granularity = "sample" if bs == self._owner.ref_table.batch_size else "shard"
            spec = FieldSpec(
                dtype=normalize_dtype(physical_value) if self._tensor else "object",
                shape=shape,
                granularity=granularity,
                token_axis=1 if len(getattr(physical_value, "shape", ())) > 1 and self._tensor else None,
            )
            self._owner.schema[key] = spec

        cache_key = self.get_cache_key(key)
        if self._tensor:
            # Clone before put/cache so later in-place mutation of the caller's
            # tensor cannot diverge cache from storage or corrupt Ray payloads.
            shared_logical_physical = physical_value is logical_value
            if _HAVE_TORCH and isinstance(physical_value, torch.Tensor):
                physical_value = physical_value.detach().contiguous().clone()
            stored_ref = engine.put(physical_value, key_hint=key, spec=spec)
            if shared_logical_physical and stored_ref.backend == "object_store":
                # Ray object-store ``put`` owns a separate wire copy, so the
                # cloned logical value can also seed the cache safely.
                logical_value = physical_value
            elif _HAVE_TORCH and isinstance(logical_value, torch.Tensor):
                # Local storage aliases its input; keep cache and storage isolated.
                logical_value = logical_value.detach().contiguous().clone()
            self._owner.ref_table[key] = stored_ref
            self._owner._batch_cache[cache_key] = logical_value
        else:
            self._owner.ref_table[key] = engine.put(
                physical_value,
                key_hint=key,
                spec=spec,
                backend=_non_tensor_backend(key),
            )
            self._owner._non_tensor_batch_cache[cache_key] = logical_value

    def __str__(self):
        return str(self.items())

    def pop(self, key: str, default: Any = ..., materialize: bool = True) -> Any:
        if key not in self._owner.ref_table:
            if default is ...:
                raise KeyError(key)
            return default
        value = None
        if materialize:
            value = self[key]
        self._owner.ref_table.pop(key)
        self._owner.schema.pop(key, None)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def slice(self, start=None, end=None, step=None):
        # Create a slice object
        owner = self._owner.slice(start, end, step)
        # Return a new DataProto object
        return type(self)(owner=owner, tensor=self._tensor, device=self._device)

    @property
    def batch_size(self):
        # mirror TensorDict's API for code that reads ``batch.batch_size[0]``
        if _HAVE_TORCH:
            return torch.Size([len(self._owner)])
        return (len(self._owner),)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Intercept torch.cat
        if func is torch.cat:
            # torch.cat expects a sequence of batches as the first argument (args[0])
            batches_to_cat = args[0]
            inst = batches_to_cat[0]
            # Create a concat object
            owner = inst._owner.concat([batch._owner for batch in batches_to_cat])
            # Return a new DataProto object
            return type(inst)(owner=owner, tensor=inst._tensor, device=inst.device)

        # Return NotImplemented for functions you haven't explicitly handled
        return NotImplemented(f"Unsupported function: {func}")
