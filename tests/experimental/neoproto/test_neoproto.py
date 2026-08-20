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

import pickle

import numpy as np
import pytest
import torch

from verl import LegacyDataProto as ClassicDataProto
from verl.experimental.neoproto import DataProto, InMemoryStorageEngine, set_default_storage_engine
from verl.protocol import BatchData, pad_dataproto_to_divisor
from verl.single_controller.base.decorator import _split_args_kwargs_data_proto_with_auto_padding


class CountingStorageEngine(InMemoryStorageEngine):
    backend = "counting"

    def __init__(self):
        super().__init__()
        self.get_count = 0
        self.get_many_count = 0

    def get(self, ref):
        self.get_count += 1
        return super().get(ref)

    def get_many(self, refs, apply_ops=True):
        self.get_many_count += 1
        return super().get_many(refs, apply_ops=apply_ops)


@pytest.fixture
def storage():
    engine = CountingStorageEngine()
    set_default_storage_engine(engine)
    yield engine
    set_default_storage_engine(None)


def _make_data(storage, offset=0, *, auto_padding=False):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(offset, offset + 12).reshape(4, 3),
            "attention_mask": torch.ones(4, 3, dtype=torch.int64),
        },
        non_tensors={
            "uid": np.asarray([f"id-{offset + index}" for index in range(4)], dtype=object),
            "local_indices": np.arange(offset, offset + 4),
        },
        meta_info={"temperature": 1.0},
        auto_padding=auto_padding,
        storage=storage,
    )


@pytest.mark.parametrize("implementation", ["classic", "neoproto"])
def test_common_data_proto_surface_matches(implementation, storage):
    tensors = {
        "input_ids": torch.arange(12).reshape(4, 3),
        "attention_mask": torch.ones(4, 3, dtype=torch.int64),
    }
    non_tensors = {"uid": np.asarray([f"id-{index}" for index in range(4)], dtype=object)}
    if implementation == "classic":
        data = ClassicDataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    else:
        data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, storage=storage)

    selected = data.select_idxs([3, 1])
    selected.batch["values"] = torch.tensor([[30.0], [10.0]])
    repeated = selected.repeat(2, interleave=True)
    chunks = repeated.chunk(2)
    merged = chunks[0].concat(chunks)
    merged.reorder(torch.tensor([3, 0, 2, 1]))
    output = merged.new_like(
        batch=merged.batch,
        non_tensor_batch=merged.non_tensor_batch,
        meta_info={"source": implementation},
    )

    assert type(output) is type(data)
    assert output.batch["input_ids"][:, 0].tolist() == [3, 9, 3, 9]
    assert output.batch["values"].squeeze(-1).tolist() == [10.0, 30.0, 10.0, 30.0]
    assert output.non_tensor_batch["uid"].tolist() == ["id-1", "id-3", "id-1", "id-3"]
    assert output.meta_info["source"] == implementation


def test_driver_operations_are_metadata_only(storage):
    data = _make_data(storage)

    first = data.select_idxs([3, 1])
    second = data.select_idxs([0, 2])
    combined = DataProto.concat([first, second])
    combined = combined.repeat(2, interleave=False)
    combined.reorder(torch.tensor([7, 0, 3, 4, 1, 6, 2, 5]))
    selected = combined.select(
        batch_keys=["input_ids"],
        non_tensor_batch_keys=["local_indices"],
        meta_info_keys=["temperature"],
    )

    assert storage.get_count == 0
    assert selected.batch["input_ids"][:, 0].tolist() == [6, 9, 6, 9, 3, 0, 0, 3]
    assert selected.non_tensor_batch["local_indices"].tolist() == [2, 3, 2, 3, 1, 0, 0, 1]
    assert storage.get_count > 0


def test_select_empty_key_lists_match_dataproto(storage):
    data = _make_data(storage)

    tensors_only = data.select(
        batch_keys=["input_ids"],
        non_tensor_batch_keys=[],
        meta_info_keys=[],
    )

    assert set(tensors_only.batch.keys()) == {"input_ids"}
    assert not set(tensors_only.non_tensor_batch.keys())
    assert not set(tensors_only.meta_info.keys())

    without_meta = data.select(meta_info_keys=[])

    assert set(without_meta.batch.keys()) == {"input_ids", "attention_mask"}
    assert set(without_meta.non_tensor_batch.keys()) == {"uid", "local_indices"}
    assert not set(without_meta.meta_info.keys())


def test_concat_from_independent_ref_tables_is_lazy(storage):
    left = _make_data(storage, offset=0).select_idxs([2, 0])
    right = _make_data(storage, offset=100).select_idxs([1, 3])

    output = DataProto.concat([left, right])

    assert storage.get_count == 0
    assert output.batch["input_ids"][:, 0].tolist() == [6, 0, 103, 109]
    assert output.non_tensor_batch["local_indices"].tolist() == [2, 0, 101, 103]


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True, False],
        np.asarray([True, False, True, False]),
        torch.tensor([True, False, True, False]),
    ],
)
def test_select_idxs_boolean_mask_matches_dataproto(storage, mask):
    data = _make_data(storage)

    selected = data.select_idxs(mask)

    assert selected.batch["input_ids"][:, 0].tolist() == [0, 6]
    assert selected.non_tensor_batch["local_indices"].tolist() == [0, 2]


def test_select_idxs_rejects_wrong_sized_boolean_mask(storage):
    data = _make_data(storage)

    with pytest.raises(IndexError, match="Boolean index"):
        data.select_idxs([True, False])


def test_union_aligns_independent_fields_after_reorder(storage):
    data = _make_data(storage)
    data.reorder(torch.tensor([2, 0, 3, 1]))
    generated = DataProto.from_dict(
        tensors={"responses": torch.tensor([[20], [0], [30], [10]])},
        meta_info={"temperature": 1.0},
        storage=storage,
    )

    output = data.union(generated)

    assert storage.get_count == 0  # equal inline metadata is inspected, tensor payloads are not
    assert output.batch["input_ids"][:, 0].tolist() == [6, 0, 9, 3]
    assert output.batch["responses"][:, 0].tolist() == [20, 0, 30, 10]


def test_assign_to_gather_view_preserves_existing_shared_fields(storage):
    data = _make_data(storage)
    view = data.select_idxs([3, 1])
    generated = DataProto.from_dict(
        tensors={"responses": torch.tensor([[30], [10]])},
        storage=storage,
    )

    view.batch.update(generated.batch)

    assert set(view.batch.keys()) >= {"input_ids", "responses"}
    assert view.batch["input_ids"][:, 0].tolist() == [9, 3]
    assert view.batch["responses"][:, 0].tolist() == [30, 10]


def test_union_rejects_conflicting_tensor_values(storage):
    left = _make_data(storage)
    right = DataProto.from_dict(
        tensors={"input_ids": torch.full((4, 3), -1, dtype=torch.long)},
        storage=storage,
    )

    with pytest.raises(AssertionError, match="input_ids"):
        left.union(right)


def test_union_accepts_equal_tensor_values(storage):
    left = _make_data(storage)
    right = DataProto.from_dict(
        tensors={"input_ids": torch.arange(12).reshape(4, 3)},
        storage=storage,
    )

    output = left.union(right)

    assert output.batch["input_ids"][:, 0].tolist() == [0, 3, 6, 9]


def test_union_rejects_different_token_views_of_same_ref(storage):
    data = _make_data(storage)
    left = data.slice_tokens("input_ids", 0, 1)
    right = data.slice_tokens("input_ids", 1, 2)

    with pytest.raises(AssertionError, match="input_ids"):
        left.union(right)


def test_dataproto_surface_and_tensordict_round_trip(storage):
    data = _make_data(storage)
    data.batch["advantages"] = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    data.non_tensor_batch["score"] = np.arange(4, dtype=np.float32)

    assert set(data.batch.keys()) == {"input_ids", "attention_mask", "advantages"}
    # Trainer paths do set & .keys() without wrapping keys() in set(...).
    assert {"input_ids", "missing"} & data.batch.keys() == {"input_ids"}
    assert {"uid", "missing"} & data.non_tensor_batch.keys() == {"uid"}
    assert "score" in data.non_tensor_batch
    assert "local_indices" in data.non_tensor_batch
    assert len(data.batch) == 4
    assert data.batch.batch_size == torch.Size([4])

    tensor_dict = data.to_tensordict()
    restored = DataProto.from_tensordict(tensor_dict)
    torch.testing.assert_close(restored.batch["input_ids"], data.batch["input_ids"])
    np.testing.assert_array_equal(restored.non_tensor_batch["uid"], data.non_tensor_batch["uid"])
    assert restored.meta_info["temperature"] == 1.0

    popped = restored.pop(batch_keys=["advantages"], non_tensor_batch_keys=["score"])
    assert set(popped.batch.keys()) == {"advantages"}
    assert set(popped.non_tensor_batch.keys()) == {"score"}
    assert "advantages" not in restored.batch
    assert "score" not in restored.non_tensor_batch


def test_padding_and_dispatch_support_neodataproto(storage):
    data = _make_data(storage, auto_padding=True)

    padded, pad_size = pad_dataproto_to_divisor(data, 3)
    assert pad_size == 2
    assert len(padded) == 6
    assert BatchData(padded).is_chunkable()
    assert BatchData([padded, padded]).is_concatable()

    args, kwargs = _split_args_kwargs_data_proto_with_auto_padding(3, data)
    assert [len(chunk) for chunk in args[0]] == [2, 2, 2]
    assert kwargs


def test_pickle_drops_materialized_cache(storage):
    data = _make_data(storage)
    expected = data.batch["input_ids"].clone()
    assert data._batch_cache

    restored = pickle.loads(pickle.dumps(data))

    assert restored._batch_cache == {}
    torch.testing.assert_close(restored.batch["input_ids"], expected)


def test_concat_aggregates_metrics(storage):
    first = DataProto.from_dict(
        tensors={"x": torch.ones(1, 1)},
        meta_info={"metrics": {"loss": 1.0}, "temperature": 1.0},
        storage=storage,
    )
    second = DataProto.from_dict(
        tensors={"x": torch.zeros(1, 1)},
        meta_info={"metrics": {"loss": 2.0}, "temperature": 1.0},
        storage=storage,
    )

    output = DataProto.concat([first, second])

    assert output.meta_info["metrics"] == {"loss": [1.0, 2.0]}
    assert output.meta_info["temperature"] == 1.0


def test_int_index_returns_dataproto_item_for_reward_path(storage):
    """Reward managers index ``data[0]`` and read response length + ground_truth.

    Over-squeezing tensor refs to 0-d used to raise IndexError on
    ``responses.shape[-1]``, which the limited reward manager swallowed as score=0.
    """
    from verl.experimental.neoproto import DataProtoItem

    data = DataProto.from_dict(
        tensors={
            "responses": torch.arange(16, dtype=torch.long).view(2, 8),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.asarray(["DigitalLearningGmbH/MATH-lighteval"] * 2, dtype=object),
            "reward_model": np.asarray(
                [{"ground_truth": "42", "style": "rule"}, {"ground_truth": "7", "style": "rule"}],
                dtype=object,
            ),
        },
        storage=storage,
    )

    item = data[-1:][0]
    assert isinstance(item, DataProtoItem)
    assert item.batch["responses"].shape == torch.Size([8])
    assert item.batch["attention_mask"].shape == torch.Size([16])
    assert item.non_tensor_batch["reward_model"]["ground_truth"] == "7"
    # Same accesses as NaiveRewardManager / LimitedRewardManager.
    response_length = item.batch["responses"].shape[-1]
    valid_response_length = item.batch["attention_mask"][-response_length:].sum()
    assert int(valid_response_length) == 8
    assert data[0].non_tensor_batch["reward_model"]["ground_truth"] == "42"
    assert data[-1].non_tensor_batch["reward_model"]["ground_truth"] == "7"
    with pytest.raises(IndexError):
        _ = data[2]


def test_int_index_inherits_prefetched_reward_cache(storage):
    data = DataProto.from_dict(
        tensors={
            "responses": torch.arange(16, dtype=torch.long).view(2, 8),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "42"}, {"ground_truth": "7"}], dtype=object),
        },
        storage=storage,
    )
    data.prefetch(["responses", "attention_mask", "data_source", "reward_model"])
    get_count = storage.get_count

    item = data[-1:][0]

    assert storage.get_count == get_count
    assert item.batch["responses"].shape == torch.Size([8])
    assert item.non_tensor_batch["reward_model"]["ground_truth"] == "7"


def test_int_index_prefetches_parent_chunk_only_once(storage):
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.arange(8, dtype=torch.long).view(2, 4),
            "responses": torch.arange(16, dtype=torch.long).view(2, 8),
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "42"}, {"ground_truth": "7"}], dtype=object),
        },
        storage=storage,
    )

    first = data[0]
    get_count = storage.get_count
    second = data[1]

    assert get_count > 0
    assert storage.get_count == get_count
    assert first.non_tensor_batch["reward_model"]["ground_truth"] == "42"
    assert second.non_tensor_batch["reward_model"]["ground_truth"] == "7"


def test_prefetch_skips_keys_already_cached_on_same_device(storage, monkeypatch):
    data = DataProto.from_dict(
        tensors={"responses": torch.arange(16, dtype=torch.long).view(2, 8)},
        storage=storage,
    )
    first = data.prefetch(["responses"])
    original_materialize = data.materialize
    calls = []

    def record_materialize(keys=None, **kwargs):
        calls.append(list(keys or []))
        return original_materialize(keys=keys, **kwargs)

    monkeypatch.setattr(data, "materialize", record_materialize)
    second = data.prefetch(["responses"])

    assert calls == []
    assert second["responses"] is first["responses"]


def test_prefetch_batches_shared_refs_by_backend(storage):
    data = DataProto.from_dict(
        tensors={
            "responses": torch.arange(16, dtype=torch.long).view(2, 8),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        },
        storage=storage,
    )

    materialized = data.prefetch(["responses", "attention_mask"])

    assert storage.get_many_count == 1
    torch.testing.assert_close(materialized["responses"], torch.arange(16, dtype=torch.long).view(2, 8))
    torch.testing.assert_close(materialized["attention_mask"], torch.ones(2, 16, dtype=torch.long))


def test_union_inherits_only_logically_aligned_caches(storage, monkeypatch):
    left = DataProto.from_dict(
        tensors={"responses": torch.arange(16, dtype=torch.long).view(2, 8)},
        storage=storage,
    )
    right = DataProto.from_dict(
        tensors={"old_log_probs": torch.zeros(2, 8)},
        storage=storage,
    )
    left.prefetch(["responses"])
    right.prefetch(["old_log_probs"])

    merged = left.union(right)

    def fail_materialize(*args, **kwargs):
        raise AssertionError("aligned union cache should avoid materialization")

    monkeypatch.setattr(merged, "materialize", fail_materialize)
    torch.testing.assert_close(merged.batch["responses"], torch.arange(16, dtype=torch.long).view(2, 8))
    torch.testing.assert_close(merged.batch["old_log_probs"], torch.zeros(2, 8))


def test_int_index_accepts_cached_single_ref_token_slice(storage):
    cached = torch.arange(50, dtype=torch.long).view(1, 50)
    data = DataProto.from_dict(tensors={"responses": cached}, storage=storage)
    data.ref_table["responses"].slice_spec = (slice(0, 50),)
    data._batch_cache["responses"] = cached

    item = data[0]

    torch.testing.assert_close(item.batch["responses"], cached[0])


def test_proxy_items_prefetches_visible_columns_together(storage, monkeypatch):
    data = DataProto.from_dict(
        non_tensors={
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "42"}, {"ground_truth": "7"}], dtype=object),
        },
        storage=storage,
    )
    original_materialize = data.materialize
    calls = []

    def record_materialize(keys=None, **kwargs):
        calls.append(list(keys or []))
        return original_materialize(keys=keys, **kwargs)

    monkeypatch.setattr(data, "materialize", record_materialize)
    first = dict(data.non_tensor_batch.items())
    second = dict(data.non_tensor_batch.items())

    assert calls == [["data_source", "reward_model"]]
    np.testing.assert_array_equal(first["data_source"], second["data_source"])
    np.testing.assert_array_equal(first["reward_model"], second["reward_model"])


def test_runtime_tensor_assignment_isolates_caller_cache_and_local_storage(storage):
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(2, 4)},
        storage=storage,
    )
    value = torch.arange(8, dtype=torch.float32).view(2, 4)
    expected = value.clone()

    data.batch["token_level_scores"] = value
    value.add_(100)
    torch.testing.assert_close(data.batch["token_level_scores"], expected)

    # Local test storage aliases the value passed to put(). Mutating the cached
    # tensor must still leave storage unchanged after a cache miss.
    data.batch["token_level_scores"].add_(200)
    data.clear_cache()
    torch.testing.assert_close(data.batch["token_level_scores"], expected)
