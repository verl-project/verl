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
"""CPU tests for verl.utils.transferqueue_utils against a real TransferQueue instance.

Covers the pure helpers (_find_meta, _postprocess_common, _compute_need_collect) and the
tqbridge decorator end-to-end: BatchMeta/KVBatchMeta in, TensorDict inside the wrapped
function, and updated metadata out. Each test uses a unique partition to isolate
TransferQueue state.
"""

import asyncio
import functools
import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from transfer_queue import BatchMeta, KVBatchMeta

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, collect_lazy_compute_data_proto
from verl.single_controller.base.worker import Worker
from verl.utils import transferqueue_utils as tqu


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    # tqbridge initializes TransferQueue lazily on first use; mark it initialized so the
    # decorated functions reuse the instance created here instead of calling tq.init() again.
    tqu.TQ_INITIALIZED = True
    yield
    tqu.TQ_INITIALIZED = False
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _make_batch_meta(partition_id: str, batch_size: int = 4, seq_len: int = 3) -> BatchMeta:
    """Insert a fresh batch into TransferQueue and return its metadata."""
    data = TensorDict(
        {"input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)},
        batch_size=(batch_size,),
    )
    client = tq.get_client()
    return client.put(data=data, partition_id=partition_id)


def _kv_put_samples(partition_id: str, num_samples: int = 2) -> tuple[list[str], list[dict]]:
    """Write per-key samples into the KV store and return their keys and tags."""
    keys, tags = [], []
    for i in range(num_samples):
        key = f"key-{uuid.uuid4().hex}-{i}"
        tag = {"idx": i}
        tq.kv_put(
            key=key,
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([i, i + 1])},
            tag=tag,
        )
        keys.append(key)
        tags.append(tag)
    return keys, tags


# --------------------------------------------------------------------------- #
# _find_meta
# --------------------------------------------------------------------------- #


def test_find_meta_in_args():
    meta = BatchMeta()
    assert tqu._find_meta(1, meta, "x") is meta


def test_find_meta_in_kwargs():
    meta = KVBatchMeta(keys=[], tags=[], partition_id="p")
    assert tqu._find_meta(1, batch=meta) is meta


def test_find_meta_returns_none_without_meta():
    assert tqu._find_meta(1, "x", batch=object()) is None


# --------------------------------------------------------------------------- #
# _postprocess_common
# --------------------------------------------------------------------------- #


def test_postprocess_put_without_collect_returns_empty_batch_meta():
    output = TensorDict({"a": torch.arange(3)}, batch_size=(3,))
    result = tqu._postprocess_common(output, put_data=True, need_collect=False)
    assert isinstance(result, BatchMeta)
    assert result.size == 0


def test_postprocess_no_put_no_collect_returns_empty_dataproto():
    output = DataProto()
    result = tqu._postprocess_common(output, put_data=False, need_collect=False)
    assert isinstance(result, DataProto)
    assert result is not output


def test_postprocess_no_put_no_collect_returns_empty_tensordict():
    output = TensorDict({"a": torch.arange(3)}, batch_size=(3,))
    result = tqu._postprocess_common(output, put_data=False, need_collect=False)
    assert isinstance(result, TensorDict)
    assert list(result.batch_size) == [0]


def test_postprocess_returns_output_otherwise():
    output = TensorDict({"a": torch.arange(3)}, batch_size=(3,))
    assert tqu._postprocess_common(output, put_data=True, need_collect=True) is output
    assert tqu._postprocess_common(output, put_data=False, need_collect=True) is output
    # non-DataProto/TensorDict output is returned unchanged even without collect
    assert tqu._postprocess_common("done", put_data=False, need_collect=False) == "done"


# --------------------------------------------------------------------------- #
# _compute_need_collect
# --------------------------------------------------------------------------- #


def test_compute_need_collect_default_true():
    assert tqu._compute_need_collect(None, []) is True
    assert tqu._compute_need_collect(Dispatch.ONE_TO_ALL, []) is True


def test_compute_need_collect_requires_collect_fn():
    with pytest.raises(AssertionError, match="collect_fn"):
        tqu._compute_need_collect({}, [])


def test_compute_need_collect_non_partial_collect_fn():
    dispatch_mode = {"collect_fn": collect_lazy_compute_data_proto}
    assert tqu._compute_need_collect(dispatch_mode, []) is True


def test_compute_need_collect_partial_other_func():
    dispatch_mode = {"collect_fn": functools.partial(len)}
    assert tqu._compute_need_collect(dispatch_mode, []) is True


def test_compute_need_collect_lazy_compute_without_worker():
    dispatch_mode = {"collect_fn": functools.partial(collect_lazy_compute_data_proto, "actor")}
    # no args, or args[0] is not a Worker: fall back to collecting
    assert tqu._compute_need_collect(dispatch_mode, []) is True
    assert tqu._compute_need_collect(dispatch_mode, [object()]) is True


def test_compute_need_collect_lazy_compute_without_mesh_name():
    dispatch_mode = {"collect_fn": functools.partial(collect_lazy_compute_data_proto)}
    worker = Worker.__new__(Worker)
    assert tqu._compute_need_collect(dispatch_mode, [worker]) is True


def test_compute_need_collect_lazy_compute_delegates_to_worker():
    dispatch_mode = {"collect_fn": functools.partial(collect_lazy_compute_data_proto, "actor")}

    worker = Worker.__new__(Worker)
    worker.query_collect_info = lambda mesh_name: True
    assert tqu._compute_need_collect(dispatch_mode, [worker]) is True

    worker.query_collect_info = lambda mesh_name: False
    assert tqu._compute_need_collect(dispatch_mode, [worker]) is False


# --------------------------------------------------------------------------- #
# tqbridge: passthrough when no meta is involved
# --------------------------------------------------------------------------- #


def test_tqbridge_passthrough_without_meta():
    @tqu.tqbridge()
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_tqbridge_passthrough_async_without_meta():
    @tqu.tqbridge()
    async def add(a, b):
        return a + b

    assert asyncio.run(add(1, 2)) == 3


# --------------------------------------------------------------------------- #
# tqbridge: BatchMeta <-> TensorDict bridge against a real TransferQueue
# --------------------------------------------------------------------------- #


def test_tqbridge_sync_round_trip(tq_init, partition_id):
    meta = _make_batch_meta(partition_id)
    meta.extra_info = {"global_steps": 7}
    expected_input = torch.arange(meta.size * 3).reshape(meta.size, 3)

    @tqu.tqbridge()
    def compute(batch: TensorDict) -> TensorDict:
        assert isinstance(batch, TensorDict)
        assert batch.batch_size[0] == meta.size
        # tensor fields come from TransferQueue storage
        assert torch.equal(batch["input_ids"], expected_input)
        # extra_info entries are injected as non-tensor data
        assert batch["global_steps"].data == 7

        output = TensorDict(
            {
                "input_ids": batch["input_ids"],
                "responses": batch["input_ids"] * 2,
            },
            batch_size=batch.batch_size,
        )
        output["prompt_text"] = NonTensorData("hello")
        return output

    updated = compute(meta)
    assert isinstance(updated, BatchMeta)
    assert updated.size == meta.size
    assert {"input_ids", "responses"}.issubset(set(updated.field_names))
    # NonTensorData entries of the output are carried in extra_info, not storage
    assert updated.extra_info == {"prompt_text": "hello"}

    data = tq.get_client().get_data(updated.select_fields(["input_ids", "responses"]))
    assert torch.equal(data["input_ids"], expected_input)
    assert torch.equal(data["responses"], expected_input * 2)


def test_tqbridge_async_round_trip(tq_init, partition_id):
    meta = _make_batch_meta(partition_id)
    expected_input = torch.arange(meta.size * 3).reshape(meta.size, 3)

    @tqu.tqbridge()
    async def compute(batch: TensorDict) -> TensorDict:
        assert isinstance(batch, TensorDict)
        return TensorDict({"responses": batch["input_ids"] + 1}, batch_size=batch.batch_size)

    updated = asyncio.run(compute(meta))
    assert isinstance(updated, BatchMeta)
    assert updated.size == meta.size

    data = tq.get_client().get_data(updated.select_fields(["responses"]))
    assert torch.equal(data["responses"], expected_input + 1)


def test_tqbridge_batch_size_mismatch_raises(tq_init, partition_id):
    meta = _make_batch_meta(partition_id, batch_size=4)

    @tqu.tqbridge()
    def compute(batch: TensorDict) -> TensorDict:
        return TensorDict({"responses": torch.zeros(2, 3)}, batch_size=(2,))

    with pytest.raises(AssertionError, match="output batch size"):
        compute(meta)


def test_tqbridge_non_tensordict_output_returned_as_is(tq_init, partition_id):
    meta = _make_batch_meta(partition_id)

    @tqu.tqbridge()
    def compute(batch: TensorDict):
        assert isinstance(batch, TensorDict)
        return "done"

    assert compute(meta) == "done"


def test_tqbridge_empty_tensordict_output_not_put(tq_init, partition_id):
    meta = _make_batch_meta(partition_id)

    @tqu.tqbridge()
    def compute(batch: TensorDict) -> TensorDict:
        return TensorDict({}, batch_size=(0,))

    result = compute(meta)
    assert isinstance(result, TensorDict)
    assert list(result.batch_size) == [0]


# --------------------------------------------------------------------------- #
# tqbridge: KVBatchMeta bridge
# --------------------------------------------------------------------------- #


def test_tqbridge_kv_batch_meta_round_trip(tq_init, partition_id):
    keys, tags = _kv_put_samples(partition_id, num_samples=2)
    kv_meta = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags, extra_info={})

    @tqu.tqbridge()
    def compute(batch: TensorDict) -> TensorDict:
        assert isinstance(batch, TensorDict)
        assert batch.batch_size[0] == len(keys)
        return TensorDict({"responses": batch["input_ids"] * 10}, batch_size=batch.batch_size)

    updated = compute(kv_meta)
    assert isinstance(updated, KVBatchMeta)
    assert set(updated.keys) == set(keys)
    assert updated.partition_id == partition_id
    # the original tags must be restored on the returned KVBatchMeta
    assert updated.tags == tags

    # the written field is retrievable through the KV path
    batch_meta = tqu.kv_batch_meta2batch_meta(updated)
    data = tq.get_client().get_data(batch_meta.select_fields(["responses"]))
    assert torch.equal(data["responses"], torch.tensor([[0, 10], [10, 20]]))


# --------------------------------------------------------------------------- #
# KVBatchMeta <-> BatchMeta conversion
# --------------------------------------------------------------------------- #


def test_kv_batch_meta2batch_meta_fields_filter(tq_init, partition_id):
    keys, tags = _kv_put_samples(partition_id, num_samples=2)
    kv_meta = KVBatchMeta(
        partition_id=partition_id,
        keys=keys,
        tags=tags,
        fields="input_ids",
        extra_info={"source": "rollout"},
    )

    batch_meta = tqu.kv_batch_meta2batch_meta(kv_meta)
    assert isinstance(batch_meta, BatchMeta)
    assert batch_meta.size == len(keys)
    assert set(batch_meta.field_names) == {"input_ids"}
    # extra_info is propagated from the KVBatchMeta
    assert batch_meta.extra_info == {"source": "rollout"}


def test_batch_meta2kv_batch_meta_round_trip(tq_init, partition_id):
    keys, tags = _kv_put_samples(partition_id, num_samples=2)
    kv_meta = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags, extra_info={})

    batch_meta = tqu.kv_batch_meta2batch_meta(kv_meta)
    restored = tqu.batch_meta2kv_batch_meta(batch_meta)

    assert isinstance(restored, KVBatchMeta)
    assert set(restored.keys) == set(keys)
    assert restored.partition_id == partition_id
    assert set(restored.fields) == set(batch_meta.field_names)
