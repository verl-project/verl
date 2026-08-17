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

import asyncio
import os
import threading
import uuid

import pytest
import torch
from tensordict import TensorDict

from verl.protocol import BatchData, DataProto, RolloutDataRef
from verl.utils import rollout_data_backend, transferqueue_utils
from verl.utils.mooncake_rollout_backend import MooncakeRolloutDataBackend
from verl.utils.transferqueue_utils import BatchMeta, tqbridge


@pytest.fixture(autouse=True)
def reset_backend():
    rollout_data_backend._backend = None
    os.environ.pop(rollout_data_backend.ROLLOUT_DATA_BACKEND_ENV, None)
    yield
    if rollout_data_backend._backend is not None:
        rollout_data_backend.close()
    rollout_data_backend._backend = None
    os.environ.pop(rollout_data_backend.ROLLOUT_DATA_BACKEND_ENV, None)


def test_transfer_queue_adapter_preserves_native_api():
    tq = pytest.importorskip("transfer_queue")
    ray = pytest.importorskip("ray")
    partition = f"test-{uuid.uuid4().hex}"
    keys = [f"key-{uuid.uuid4().hex}" for _ in range(2)]
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(include_dashboard=False)
    rollout_data_backend.configure_runtime({"name": "transfer_queue"})
    rollout_data_backend.init(host_catalog=True)
    try:
        ref = rollout_data_backend.batch_put(
            keys=keys,
            partition_id=partition,
            tags=[{"status": "success"}] * 2,
            fields=TensorDict({"value": torch.tensor([[1], [2]])}, batch_size=[2]),
        )
        assert isinstance(ref, RolloutDataRef)
        assert all(isinstance(chunk, BatchMeta) for chunk in BatchData(ref).chunk(2))
        result = rollout_data_backend.batch_get(
            keys=keys, partition_id=partition, select_fields=["value"]
        )
        assert torch.equal(result["value"], torch.tensor([[1], [2]]))
        assert rollout_data_backend.list_entries(partition)[partition][keys[0]]["status"] == "success"
    finally:
        remaining = tq.kv_list(partition_id=partition).get(partition, {})
        if remaining:
            tq.kv_clear(keys=list(remaining), partition_id=partition)
        rollout_data_backend.close()
        if started_ray:
            ray.shutdown()


def test_backend_close_can_retry_after_shutdown_failure():
    class Backend:
        calls = 0

        def shutdown(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary shutdown failure")

    backend = Backend()
    rollout_data_backend._backend = backend
    with pytest.raises(RuntimeError, match="temporary shutdown failure"):
        rollout_data_backend.close()
    assert rollout_data_backend._backend is backend
    rollout_data_backend.close()
    assert rollout_data_backend._backend is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rollout_ref_bridge_materializes_writes_and_releases(monkeypatch, asynchronous):
    source = TensorDict({"value": torch.tensor([[1], [2]])}, batch_size=[2])
    writes, releases = [], []
    monkeypatch.setattr(rollout_data_backend, "batch_get", lambda **_: source)
    monkeypatch.setattr(rollout_data_backend, "release_result", releases.append)

    def put(**kwargs):
        writes.append(kwargs["fields"])
        return RolloutDataRef(
            keys=kwargs["keys"], tags=[{}, {}], partition_id=kwargs["partition_id"]
        )

    monkeypatch.setattr(rollout_data_backend, "batch_put", put)
    ref = RolloutDataRef(keys=["a", "b"], tags=[{}, {}], partition_id="train")
    if asynchronous:

        @tqbridge()
        async def transform(batch):
            return batch.apply(lambda value: value * 2)

        result = asyncio.run(transform(ref))
    else:

        @tqbridge()
        def transform(batch):
            return batch.apply(lambda value: value * 2)

        result = transform(ref)

    assert isinstance(result, RolloutDataRef)
    assert torch.equal(writes[0]["value"], torch.tensor([[2], [4]]))
    assert releases == [source]


def test_materialized_batch_releases_on_exit(monkeypatch):
    source = TensorDict({"value": torch.tensor([1])}, batch_size=[1])
    releases = []
    monkeypatch.setattr(rollout_data_backend, "batch_get", lambda **_: source)
    monkeypatch.setattr(rollout_data_backend, "release_result", releases.append)
    with rollout_data_backend.materialized_batch(keys=["a"]) as result:
        assert result is source
    assert releases == [source]


def test_rollout_ref_chunk_and_concat_preserve_order():
    ref = RolloutDataRef(
        keys=["a", "b", "c"],
        tags=[{"i": 0}, {"i": 1}, {"i": 2}],
        partition_id="train",
        fields=["value"],
    )
    chunks = ref.chunk(2)
    chunks[0].extra_info = {"rank": 0}
    chunks[1].extra_info = {"rank": 2}
    merged = RolloutDataRef.concat(
        [chunks[0], RolloutDataRef(extra_info={"rank": 1}), chunks[1]]
    )
    assert merged.keys == ref.keys
    assert merged.extra_info == {"rank": [0, 1, 2]}


def test_cancelled_mooncake_put_waits_for_worker():
    started, finish = threading.Event(), threading.Event()
    backend = MooncakeRolloutDataBackend({})

    def put(**_kwargs):
        started.set()
        finish.wait()

    backend.put_batch = put

    async def run():
        task = asyncio.create_task(backend.put_batch_async(keys=["a"]))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        try:
            assert not task.done()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_rollout_ref_metadata_failure_releases_result(monkeypatch):
    source = TensorDict({"value": torch.tensor([1])}, batch_size=[1])
    releases = []
    monkeypatch.setattr(rollout_data_backend, "batch_get", lambda **_: source)
    monkeypatch.setattr(rollout_data_backend, "release_result", releases.append)

    def fail_metadata(**_kwargs):
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(transferqueue_utils.tu, "assign_non_tensor_data", fail_metadata)
    ref = RolloutDataRef(keys=["a"], tags=[{}], partition_id="train", extra_info={"meta": 1})
    with pytest.raises(RuntimeError, match="metadata failed"):
        transferqueue_utils._rollout_ref_to_realdata(ref)
    assert releases == [source]


def test_backend_specific_row_layout():
    rollout_data_backend.configure_runtime({"name": "mooncake", "config": {}})
    fields = rollout_data_backend.rows_to_fields(
        [
            {
                "responses": torch.tensor([1, 2]),
                "position_ids": torch.tensor([[0, 1], [0, 1]]),
            },
            {
                "responses": torch.tensor([3, 4]),
                "position_ids": torch.tensor([[2, 3], [2, 3]]),
            },
        ]
    )
    assert fields["responses"].is_nested
    assert torch.equal(fields["responses"].offsets().diff(), torch.tensor([2, 2]))
    assert not fields["position_ids"].is_nested
    assert fields["position_ids"].shape == (2, 2, 2)


class CatalogTransferStub:
    def __init__(self, plan):
        self.plan = plan
        self.released_reads = []
        self.discarded = []

    def resolve(self, *_args):
        return self.plan

    def release_read(self, token):
        self.released_reads.append(token)
        if getattr(self, "release_failures", 0):
            self.release_failures -= 1
            raise RuntimeError("release failed")

    def attach_results(self, output, results):
        output._mooncake_catalog_results = results

    def discard_results(self, results):
        self.discarded.extend(results)

    def release_result(self, output):
        self.discarded.extend(getattr(output, "_mooncake_catalog_results", []))

    def close(self):
        pass

    def drain(self):
        pass


class FragmentTransferStub:
    def __init__(self, fragments, fail_on=None):
        self.fragments = fragments
        self.fail_on = fail_on

    def get(self, handle, *, fields, rows, **_kwargs):
        name = handle["name"]
        if name == self.fail_on:
            raise RuntimeError(f"{name} failed")
        data = self.fragments[name].select(*fields)
        if rows is not None:
            data = data[rows]
        return DataProto.from_tensordict(data)


def _plan(*, second_field="score"):
    return {
        "fields": ["tokens", second_field],
        "field_groups": [
            {"fields": ["tokens"], "locations": [("base", 0), ("base", 1)]},
            {"fields": [second_field], "locations": [("second", 1), ("second", 0)]},
        ],
        "handles": {
            "base": {"name": "base", "batch_size": 2},
            "second": {"name": "second", "batch_size": 2},
        },
        "meta_info": {"stage": "rollout"},
        "read_token": 7,
    }


def test_mooncake_backend_reassembles_fragmented_fields():
    fragments = {
        "base": TensorDict(
            {"tokens": torch.nested.as_nested_tensor([torch.tensor([1, 2]), torch.tensor([3])], layout=torch.jagged)},
            batch_size=[2],
        ),
        "second": TensorDict({"score": torch.tensor([10, 20])}, batch_size=[2]),
    }
    backend = MooncakeRolloutDataBackend({})
    backend.catalog_transfer = CatalogTransferStub(_plan())
    backend.transfer = FragmentTransferStub(fragments)
    result = backend.get_batch(keys=["a", "b"], partition_id="train")
    assert result["tokens"].is_nested
    assert torch.equal(result["score"], torch.tensor([20, 10]))
    assert result["stage"] == "rollout"
    assert backend.catalog_transfer.released_reads == [7]
    backend.release_result(result)
    assert len(backend.catalog_transfer.discarded) == 2


def test_mooncake_read_pin_release_retries_rpc():
    backend = MooncakeRolloutDataBackend({})
    backend.catalog_transfer = CatalogTransferStub(_plan())
    backend.catalog_transfer.release_failures = 1
    backend._release_read(7)
    assert backend.catalog_transfer.released_reads == [7, 7]

    backend.catalog_transfer.release_failures = 2
    with pytest.raises(RuntimeError, match="release failed"):
        backend._release_read(8)
    assert backend._pending_read_tokens == {8}
    backend.shutdown()
    assert backend._pending_read_tokens == set()


def test_mooncake_shutdown_keeps_store_after_pool_failure():
    calls = []

    class Resource:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    backend = MooncakeRolloutDataBackend({})
    backend.catalog_transfer = CatalogTransferStub(_plan())
    backend.buffer_pool = Resource("pool", fail=True)
    backend.store = Resource("store")
    with pytest.raises(RuntimeError, match="pool failed"):
        backend.shutdown()
    assert calls == ["pool"]


def test_mooncake_shutdown_keeps_catalog_handle_when_kill_fails(monkeypatch):
    import ray

    catalog = object()
    backend = MooncakeRolloutDataBackend({}, host_catalog=True)
    backend.catalog = catalog
    backend.catalog_transfer = CatalogTransferStub(_plan())

    def fail_kill(*_args, **_kwargs):
        raise RuntimeError("kill failed")

    monkeypatch.setattr(ray, "kill", fail_kill)
    with pytest.raises(RuntimeError, match="kill failed"):
        backend.shutdown()
    assert backend.catalog is catalog

    monkeypatch.setattr(ray, "kill", lambda *_args, **_kwargs: None)
    backend.shutdown()
    assert backend.catalog is None


def test_mooncake_get_failure_discards_partial_result_and_read_pin():
    fragments = {
        "base": TensorDict({"tokens": torch.tensor([[1], [2]])}, batch_size=[2]),
        "second": TensorDict({"score": torch.tensor([10, 20])}, batch_size=[2]),
    }
    backend = MooncakeRolloutDataBackend({})
    backend.catalog_transfer = CatalogTransferStub(_plan())
    backend.transfer = FragmentTransferStub(fragments, fail_on="second")
    with pytest.raises(RuntimeError, match="second failed"):
        backend.get_batch(keys=["a", "b"], partition_id="train")
    assert len(backend.catalog_transfer.discarded) == 1
    assert backend.catalog_transfer.released_reads == [7]
