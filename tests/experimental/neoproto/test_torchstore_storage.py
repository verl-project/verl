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

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest
import torch

from verl.experimental.neoproto import neo as neo_module
from verl.runtime.object_store import _close_object_store, _start_object_store
from verl.single_controller.monarch.object_store.store import TensorRowRange, TorchStoreObjectStore, TorchStoreReference


class StrictFakeTorchStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], object] = {}
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    async def put(self, key: str, value: object, *, store_name: str) -> None:
        self.calls.append(("put", store_name, (key,)))
        self.values[(store_name, key)] = value

    async def get_batch(self, keys: list[str], *, store_name: str) -> dict[str, object]:
        self.calls.append(("get_batch", store_name, tuple(keys)))
        return {key: self.values[(store_name, key)] for key in keys}

    async def delete_batch(self, keys: list[str], *, store_name: str) -> None:
        self.calls.append(("delete_batch", store_name, tuple(keys)))
        for key in keys:
            del self.values[(store_name, key)]


class RowSelectingObjectStore(TorchStoreObjectStore):
    """Honor the concrete row-read contract without starting native worker processes."""

    def __init__(self, values):
        super().__init__("reader")
        self.values = values
        self.requests = []

    def get_many(self, references, /, *, row_ranges=None):
        self.requests.append((list(references), row_ranges))
        fetched = {}
        for reference in references:
            if reference in fetched:
                continue
            value = self.values[reference]
            selection = (row_ranges or {}).get(reference)
            if selection is not None and reference.is_tensor and isinstance(value, torch.Tensor):
                value = value[selection.start : selection.stop]
            fetched[reference] = value
        return [fetched[reference] for reference in references]


@pytest.fixture
def row_store():
    from verl.experimental.neoproto.storage.engine import Ref
    from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine

    base = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    tensor_key = TorchStoreReference("producer", "tensor", is_tensor=True)
    array_key = TorchStoreReference("other-producer", "array")
    store = RowSelectingObjectStore({tensor_key: base, array_key: base.numpy().copy()})
    _start_object_store(store)
    tensor_ref = Ref(backend="torchstore", uid="tensor", dataptr=tensor_key, shape=tuple(base.shape))
    array_ref = Ref(backend="torchstore", uid="array", dataptr=array_key, shape=tuple(base.shape))
    try:
        yield TorchStorageEngine(), store, base, tensor_ref, array_ref
    finally:
        _close_object_store()


def test_torchstore_selected_rows_preserve_order_duplicates_objects_and_operations(row_store):
    engine, store, base, ref, array_ref = row_store
    transformed = ref.with_slice((slice(3, 6), slice(1, 7, 2)))
    transformed.apply_funcs = [("TO", {"dtype": torch.float64}), ("UNSQUEEZE", 0)]
    result = engine.get_many([ref.with_sample(5), transformed, array_ref.with_sample(2), ref.with_sample(5), None])
    assert torch.equal(result[0], base[5])
    assert torch.equal(result[1], base[3:6, 1:7:2].to(torch.float64).unsqueeze(0))
    np.testing.assert_array_equal(result[2], base.numpy()[2])
    assert torch.equal(result[3], base[5]) and result[4] is None
    assert store.requests[-1][1][ref.dataptr] == TensorRowRange((16, 8), 3, 6)


@pytest.mark.parametrize("selector", [slice(16, 16), slice(5, 5), slice(12, 3), slice(-7, -2), -1])
def test_torchstore_row_read_handles_empty_and_negative_bounds(row_store, selector):
    engine, _store, base, ref, _array_ref = row_store
    selected = ref.copy()
    selected.slice_spec = (selector, slice(None))
    result = engine.get(selected)
    assert torch.equal(result, base[selector]) and result.shape == base[selector].shape


def test_torchstore_full_advanced_and_raw_reads_keep_the_existing_path(row_store):
    engine, store, base, ref, _array_ref = row_store
    assert torch.equal(engine.get_many([ref, ref.with_sample(3)])[1], base[3])
    assert store.requests[-1][1] is None
    advanced = ref.copy()
    advanced.slice_spec = ([7, 2, 7], slice(None))
    assert torch.equal(engine.get(advanced), base[[7, 2, 7]])
    assert store.requests[-1][1] is None
    raw = engine.get_many([ref.with_sample(3)], apply_ops=False)[0]
    assert torch.equal(raw, base) and store.requests[-1][1] is None


def test_tensor_valued_object_and_legacy_references_do_not_use_shape_to_enable_row_reads(row_store):
    engine, store, base, ref, _array_ref = row_store
    # Native OBJECT decoding can produce a Tensor; only the producer knows
    # whether the storage volume applied a TensorSlice before deserialization.
    legacy = ref.copy()
    legacy.dataptr = TorchStoreReference(ref.dataptr.store_name, ref.dataptr.key)
    assert torch.equal(engine.get(legacy.with_sample(3)), base[3])
    assert store.requests[-1][1] is None
    # Equality/address identity is preserved when an old reference meets a hint.
    assert legacy.dataptr == ref.dataptr
    assert torch.equal(engine.get_many([ref.with_sample(4), legacy.with_sample(3)])[1], base[3])
    assert store.requests[-1][1] is None


def test_neoproto_round_trips_through_concrete_torchstore_batches(monkeypatch) -> None:
    native = StrictFakeTorchStore()
    monkeypatch.setitem(sys.modules, "torchstore", native)
    neo_module.GLOBAL_ENGINE_DICT.clear()
    try:
        from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine
        from verl.experimental.neoproto.views import DataProto
        from verl.single_controller.monarch.object_store.store import TorchStoreObjectStore

        store = TorchStoreObjectStore("neo-test-store")
        _start_object_store(store)
        source = DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(12).reshape(3, 4),
                "attention_mask": torch.ones((3, 4), dtype=torch.int64),
            },
            storage=TorchStorageEngine(),
        )
        transported = pickle.loads(pickle.dumps(source))
        stored_refs = [transported.ref_table[key] for key in ("input_ids", "attention_mask")]

        materialized = transported.materialize(["input_ids", "attention_mask"])

        assert torch.equal(materialized["input_ids"], torch.arange(12).reshape(3, 4))
        assert torch.equal(materialized["attention_mask"], torch.ones((3, 4), dtype=torch.int64))
        transported.release()
        store.close()
        assert native.values == {}

        with pytest.raises(KeyError):
            TorchStorageEngine().get_many(stored_refs)

        assert [call[0] for call in native.calls] == [
            "put",
            "put",
            "get_batch",
            "delete_batch",
            "get_batch",
        ]
    finally:
        neo_module.GLOBAL_ENGINE_DICT.clear()
        _close_object_store()
