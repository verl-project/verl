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

import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.neoproto import DataProto
from verl.experimental.neoproto import neo as neo_module
from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine
from verl.runtime.object_store import _close_object_store, _start_object_store
from verl.single_controller.monarch.object_store.store import TensorRowRange, TorchStoreObjectStore


@dataclass
class Slice:
    offsets: tuple
    coordinates: tuple
    global_shape: tuple
    local_shape: tuple
    mesh_shape: tuple


@dataclass
class Request:
    key: str
    tensor_slice: Slice | None
    tensor_val: torch.Tensor | None = None

    @classmethod
    def from_any(cls, key, value, tensor_slice=None):
        assert value is None
        return cls(key, tensor_slice)


class NativeStore:
    """Fake only the external native storage IO, including independent reads."""

    def __init__(self):
        self.values = {}
        self.reads = []

    async def put(self, key, value, *, store_name):
        self.values[store_name, key] = value

    async def get_batch(self, keys, *, store_name):
        values = await self.read_requests([Request(key, None) for key in keys], store_name=store_name)
        return dict(zip(keys, values, strict=True))

    async def client(self, store_name):
        async def locate(keys):
            return {key: {store_name: None} for key in keys}

        return SimpleNamespace(
            _locate_volumes=locate,
            strategy=SimpleNamespace(get_storage_volume=lambda volume: volume),
            _build_volume_requests=lambda requests, locations, buffers: (
                {store_name: requests},
                {r.key for r in requests},
            ),
            _assemble_results=self.assemble,
        )

    @staticmethod
    def assemble(requests, parts, whole):
        # This fixture stores unsharded values in one volume.
        return {request.key: value for request, value in parts}

    async def read_requests(self, requests, *, store_name):
        # One volume transport returns an ordered value for each request.
        self.reads.append((store_name, requests))
        values = []
        for request in requests:
            value = self.values[store_name, request.key]
            if isinstance(value, torch.Tensor):
                if request.tensor_slice is not None:
                    start = request.tensor_slice.offsets[0]
                    value = value[start : start + request.tensor_slice.local_shape[0]]
                value = value.clone()
            values.append(value)
        return values

    async def delete_batch(self, keys, *, store_name):
        for key in keys:
            self.values.pop((store_name, key), None)


@pytest.fixture
def native(monkeypatch):
    native = NativeStore()
    monkeypatch.setitem(sys.modules, "torchstore", native)

    def transport(store_name):
        async def get(requests):
            return await native.read_requests(requests, store_name=store_name)

        return SimpleNamespace(get_from_storage_volume=get)

    monkeypatch.setitem(
        sys.modules,
        "torchstore.transport",
        SimpleNamespace(Request=Request, TensorSlice=Slice, create_transport_buffer=transport),
    )

    def bounds(shapes, offsets):
        origin = min(offsets)
        shape = tuple(
            max(offset[axis] + size[axis] for offset, size in zip(offsets, shapes, strict=True)) - origin[axis]
            for axis in range(len(origin))
        )
        assert sum(math.prod(size) for size in shapes) >= math.prod(shape)
        return shape, origin

    monkeypatch.setitem(sys.modules, "torchstore.utils", SimpleNamespace(get_target_tensor_shape_and_offset=bounds))
    store = TorchStoreObjectStore("test")
    _start_object_store(store)
    neo_module.GLOBAL_ENGINE_DICT.clear()
    yield native, store
    neo_module.GLOBAL_ENGINE_DICT.clear()
    _close_object_store()


def test_materialize_batches_selected_shards_per_field(native):
    io, _store = native
    engine = TorchStorageEngine()
    values = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    parts = [
        DataProto.from_dict(
            tensors={"x": values[start : start + 8], "y": values[start : start + 8] + 100}, storage=engine
        )
        for start in (0, 8)
    ]
    batch = DataProto.concat(parts)
    batch.reorder(torch.tensor([3, 11, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]))
    selected = batch.slice(0, 2)
    result = selected.materialize(["x", "y"])
    assert torch.equal(result["x"], values[[3, 11]])
    assert torch.equal(result["y"], values[[3, 11]] + 100)
    assert len(io.reads) == 2
    assert sum(len(requests) for _, requests in io.reads) == 4
    assert all(request.tensor_slice.local_shape == (1, 4) for _, requests in io.reads for request in requests)


def test_shared_sparse_rows_read_exact_rows_in_unique_key_batches(native):
    io, _store = native
    engine = TorchStorageEngine()
    value = torch.arange(128 * 512, dtype=torch.float32).reshape(128, 512)
    batch = DataProto.from_dict(tensors={"x": value, "y": value + 1}, storage=engine)
    rows = np.array([19, 67, 127, 108, 32, 63, 91, 51])
    batch.dim0_index.sample_indices = rows
    result = batch.materialize(["x", "y"])
    assert torch.equal(result["x"], value[rows])
    assert torch.equal(result["y"], (value + 1)[rows])
    fetched_rows = sum(
        io.values[store_name, r.key].shape[0] if r.tensor_slice is None else r.tensor_slice.local_shape[0]
        for store_name, requests in io.reads
        for r in requests
    )
    assert fetched_rows == 16
    assert len(io.reads) == 1


def test_sparse_per_sample_views_preserve_order_duplicates_and_overlap(native):
    io, store = native
    engine = TorchStorageEngine()
    value = torch.arange(64).reshape(16, 4)
    ref = engine.put(value)
    left, far, repeated = engine.get_many(
        [ref.with_slice((slice(2, 4),)), ref.with_slice((slice(11, 13),)), ref.with_slice((slice(3, 4),))]
    )
    assert torch.equal(left, value[2:4])
    assert torch.equal(far, value[11:13])
    left[1, 0] = -99
    assert repeated[0, 0] == -99
    assert sum(r.tensor_slice.local_shape[0] for _, requests in io.reads for r in requests) == 4
    before = len(io.reads)
    gap = store.get_many([ref.dataptr], row_ranges={ref.dataptr: TensorRowRange((16, 4), 5, 6)})[0]
    assert torch.equal(gap, value[5:6])
    assert len(io.reads) == before + 1
