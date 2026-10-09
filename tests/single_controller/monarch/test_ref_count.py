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

"""Consumer isolation and final-reader cleanup for shared snapshots."""

import pickle
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("monarch")

from verl.single_controller.monarch.object_store.ref_count import RefCountRegistry
from verl.single_controller.monarch.object_store.store import TorchStoreObjectStore


@pytest.fixture
def native_store(monkeypatch):
    values = {}
    published = []
    deleted = []
    registry = RefCountRegistry("store")

    async def put(key, value, *, store_name):
        assert store_name == "store"
        values[key] = pickle.dumps(value)
        published.append(key)

    async def get(key, *, store_name):
        assert store_name == "store"
        return pickle.loads(values[key])

    async def delete(key, *, store_name):
        assert store_name == "store"
        del values[key]
        deleted.append(key)

    async def put_batch(entries, *, store_name):
        for key, value in entries.items():
            await put(key, value, store_name=store_name)

    async def get_batch(keys, *, store_name):
        return {key: await get(key, store_name=store_name) for key in keys}

    async def delete_batch(keys, *, store_name):
        for key in keys:
            await delete(key, store_name=store_name)

    async def native_delete(store_name, key):
        if key in values:
            await delete(key, store_name=store_name)

    async def select_owner(store_name, owners):
        assert store_name == "store" and owners == "volume-owners"
        return "selected-owner"

    async def register(owner, key, tokens):
        assert owner == "selected-owner"
        registry.register(key, tokens)

    async def release(owner, key, token):
        assert owner == "selected-owner"
        await registry.release(key, token)

    async def abort(owner, key, tokens):
        await registry.abort(key, tokens)

    monkeypatch.setitem(
        sys.modules,
        "torchstore",
        SimpleNamespace(
            put=put, get=get, delete=delete, put_batch=put_batch, get_batch=get_batch, delete_batch=delete_batch
        ),
    )
    # Model logical values after transport; exercise the real lease registry.
    monkeypatch.setitem(
        sys.modules,
        "verl.single_controller.monarch.patches.torchstore_compat",
        SimpleNamespace(_prepare_shared_snapshot=lambda value: value),
    )
    module = "verl.single_controller.monarch.object_store.ref_count."
    monkeypatch.setattr(module + "select_ref_count_actor", select_owner)
    monkeypatch.setattr(module + "_delete_stored_key", native_delete)
    monkeypatch.setattr(module + "register_ref_count", register)
    monkeypatch.setattr(module + "release_ref", release)
    monkeypatch.setattr(module + "abort_ref_count", abort)
    store = TorchStoreObjectStore("store", timeout_s=5, ref_count_actors="volume-owners")
    yield store, registry, values, published, deleted
    store.close()


def test_shared_reference_later_reader_survives_first_release(native_store):
    store, _registry, values, published, deleted = native_store
    payload = {"value": [1, 2]}
    references = store.put_shared("local", {"left": payload, "right": payload}, 2)
    assert len(published) == 1
    first_ref, second_ref = [pickle.loads(pickle.dumps(reference)) for reference in references]
    first = store.get(first_ref)
    first["left"]["value"].append(3)
    assert first["right"]["value"] == [1, 2, 3]
    store.delete(first_ref)
    store.close()
    assert values and not deleted
    second = store.get(second_ref)
    assert second["left"]["value"] == [1, 2]
    store.delete(second_ref)
    store.close()
    assert not values and deleted == published


def test_dispatch_transport_is_unique_releasable_and_retryable(native_store, monkeypatch):
    import numpy as np
    import torch

    from verl.experimental.neoproto import DataProto, InMemoryStorageEngine, set_default_storage_engine
    from verl.single_controller.base.decorator import _split_args_kwargs_data_proto

    store, _registry, values, published, deleted = native_store
    storage = InMemoryStorageEngine()
    set_default_storage_engine(storage)
    monkeypatch.setattr("verl.runtime.object_store._OBJECT_STORE", store)
    native = sys.modules["torchstore"]
    get_batch = native.get_batch
    fail_once = set()

    async def fail_get_batch_once(keys, *, store_name):
        failed = fail_once.intersection(keys)
        if failed:
            fail_once.difference_update(failed)
            raise RuntimeError("transient get failure")
        return await get_batch(keys, store_name=store_name)

    monkeypatch.setattr(native, "get_batch", fail_get_batch_once)
    try:
        first = DataProto.from_dict(
            tensors={"input_ids": torch.arange(8).view(4, 2)},
            non_tensors={"uid": np.asarray([f"first-{index}" for index in range(4)], dtype=object)},
            storage=storage,
        )
        chunks = _split_args_kwargs_data_proto(2, first)[0][0]
        payloads = [pickle.dumps(chunk) for chunk in chunks]
        fail_once.add(published[-1])
        with pytest.raises(RuntimeError, match="transient get failure"):
            pickle.loads(payloads[-1])
        restored = [pickle.loads(payload) for payload in payloads]
        issued_before_forward = len(published)
        forwarded = [pickle.loads(pickle.dumps(chunk)) for chunk in restored]
        assert len(published) == issued_before_forward
        assert torch.equal(torch.cat([chunk.batch["input_ids"] for chunk in forwarded]), first.batch["input_ids"])
        store.close()
        assert not values
        assert len(published) == len(set(published))
        assert sorted(deleted) == sorted(published)
    finally:
        set_default_storage_engine(None)
