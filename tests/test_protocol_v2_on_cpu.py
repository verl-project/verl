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

"""
Replace DataProto with raw TensorDict
"""

import copy
import random

import numpy as np
import pytest
import torch
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.utils import tensordict_utils as tu


def test_union_tensor_dict():
    obs = torch.randn(100, 10)

    meta_info1 = {"top_p": 0.8}
    meta_info2 = {"top_p": 0.9}
    data1 = {"obs": obs, "act": torch.randn(100, 3), "data_sources": ["gsm8k"] * 100}
    data2 = {"obs": obs, "next_obs": torch.randn(100, 10), "rew": torch.randn(100), "data_sources": ["gsm8k"] * 100}

    data_with_copied_obs = {"obs": obs.clone(), "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}

    data1 = tu.get_tensordict(tensor_dict=data1)
    data2 = tu.get_tensordict(tensor_dict=data2)
    data_with_copied_obs = tu.get_tensordict(data_with_copied_obs)

    tu.union_tensor_dict(data1, data2)
    with pytest.raises(AssertionError):
        # conflict in tensor values
        tu.union_tensor_dict(data1, data_with_copied_obs)

    data1 = tu.assign_non_tensor(data1, **meta_info1)
    tu.union_tensor_dict(data1, data2)  # works ok

    data2 = tu.assign_non_tensor(data2, **meta_info2)

    with pytest.raises(AssertionError):
        # conflict in NonTensorData
        tu.union_tensor_dict(data1, data2)

    data1.pop("top_p")
    data2.pop("top_p")

    data2["data_sources"][0] = "math"
    with pytest.raises(AssertionError):
        # conflict in NonTensorData
        tu.union_tensor_dict(data1, data2)


def test_tensor_dict_constructor():
    obs = torch.ones(100, 10)
    act = torch.zeros(100, 10, 3)
    data_source = ["gsm8k"] * 100
    non_tensor_dict = {"name": "abdce"}

    data = tu.get_tensordict(
        tensor_dict={"obs": obs, "act": act, "data_source": data_source}, non_tensor_dict=non_tensor_dict
    )

    assert data.batch_size == torch.Size([100])

    # test slicing
    assert torch.all(torch.eq(data[0]["obs"], torch.ones(10))).item()
    assert torch.all(torch.eq(data[0]["act"], torch.zeros(10, 3))).item()
    assert data[0]["data_source"] == "gsm8k"

    assert torch.all(torch.eq(data[0:2]["obs"], torch.ones(2, 10))).item()
    assert torch.all(torch.eq(data[0:2]["act"], torch.zeros(2, 10, 3))).item()
    assert data[0:2]["data_source"] == ["gsm8k"] * 2

    # test non tensor data
    assert data["name"] == "abdce"


def test_index_select_tensor_dict():
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    c = torch.randint(low=0, high=vocab_size, size=(12,))
    d = torch.randint(low=0, high=vocab_size, size=(15,))
    input_ids = [a, b, c, d]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    padded_tensor = torch.randn(4, 10)
    non_tensor_dict = {"global_batch_size": "4"}

    data = tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "padded_tensor": padded_tensor,
        },
        non_tensor_dict=non_tensor_dict,
    )

    assert data.batch_size == torch.Size([4])

    # test index select
    indices = torch.tensor([1, 3])
    selected_data = tu.index_select_tensor_dict(data, indices)

    assert selected_data.batch_size == torch.Size([2])

    target_input_ids = torch.nested.as_nested_tensor([input_ids[idx] for idx in indices], layout=torch.jagged)
    target_select_data = tu.get_tensordict(
        tensor_dict={
            "input_ids": target_input_ids,
            "padded_tensor": padded_tensor[indices],
        },
        non_tensor_dict=non_tensor_dict,
    )
    tu.assert_tensordict_eq(selected_data, target_select_data)


def test_tensordict_with_images():
    # each sample contains a sequence with multiple images of different sizes
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    # must be numpy
    # TODO(vermouth1992). We may use nested tensor too. But this requires nested over nested
    a_images = [
        torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy(),
    ]
    b_images = [
        torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 64, 64), dtype=torch.uint8).numpy(),
    ]

    images = [a_images, b_images]

    data = tu.get_tensordict({"input_ids": input_ids, "images": images})

    assert np.all(np.equal(data[0]["images"][0], a_images[0]))
    assert torch.all(torch.eq(data[0]["input_ids"], a))


def test_tensordict_with_packing():
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    data = tu.get_tensordict({"input_ids": input_ids})

    # test cu_seqlens
    cu_seqlens = torch.tensor([0, 11, 24])
    assert torch.all(torch.eq(cu_seqlens, data["input_ids"].offsets()))

    # test index
    assert torch.all(torch.eq(data["input_ids"][0], a))
    assert torch.all(torch.eq(data["input_ids"][1], b))

    assert torch.all(torch.eq(data[0]["input_ids"], a))
    assert torch.all(torch.eq(data[1]["input_ids"], b))

    data_lst = data.chunk(2)

    assert torch.all(torch.eq(data_lst[0]["input_ids"][0], a))
    assert torch.all(torch.eq(data_lst[1]["input_ids"][0], b))


def test_tensordict_eq():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0}, "val_sample_kwargs": {"top_p": 0.7}}
    data = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0}, "val_sample_kwargs": {"top_p": 0.7}}
    data1 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    tu.assert_tensordict_eq(data, data1)

    data2 = copy.deepcopy(data1)
    data2["obs"][0] += 1

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)

    data2 = copy.deepcopy(data1)
    data2["data_sources"][0] = "math"

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)

    data2 = copy.deepcopy(data1)
    data2["train_sample_kwargs"]["top_p"] = 0.9

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)

    tensor_list = [
        torch.tensor([1, 2, 3, 3, 2]),
        torch.tensor([4, 5]),
        torch.tensor([7, 8, 10, 14]),
        torch.tensor([10, 11, 12]),
        torch.tensor([13, 14, 15, 18]),
        torch.tensor([16, 17]),
    ]
    obs = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0}, "val_sample_kwargs": {"top_p": 0.7}}
    data3 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    tensor_list[0] = torch.tensor([1, 2, 3, 3, 2])
    obs = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    data4 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)
    tu.assert_tensordict_eq(data3, data4)

    tensor_list[0] = torch.tensor([1, 2, 4])
    obs = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    data5 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)
    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data3, data5)

    tensor_list[0] = torch.tensor([4, 5])
    tensor_list[1] = torch.tensor([1, 2, 3, 3, 2])
    obs = torch.nested.as_nested_tensor(tensor_list, layout=torch.jagged)
    data6 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)
    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data3, data6)


def test_tensor_dict_make_iterator():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    input_ids = torch.nested.as_nested_tensor(
        [
            torch.tensor([0, 1]),
            torch.tensor([2]),
            torch.tensor([3, 4]),
            torch.tensor([5]),
            torch.tensor([6, 7, 8]),
            torch.tensor([9]),
        ],
        layout=torch.jagged,
    )
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0}, "val_sample_kwargs": {"top_p": 0.7}}
    dataset = tu.get_tensordict(
        {"obs": obs, "data_sources": data_sources, "input_ids": input_ids}, non_tensor_dict=non_tensor_dict
    )

    dataloader = tu.make_iterator(
        dataset, mini_batch_size=2, epochs=2, seed=0, dataloader_kwargs={"shuffle": False, "drop_last": False}
    )

    expected_tensor_dict = [
        tu.index_select_tensor_dict(dataset, indices=list(range(0, 2))),
        tu.index_select_tensor_dict(dataset, indices=list(range(2, 4))),
        tu.index_select_tensor_dict(dataset, indices=list(range(4, 6))),
        tu.index_select_tensor_dict(dataset, indices=list(range(0, 2))),
        tu.index_select_tensor_dict(dataset, indices=list(range(2, 4))),
        tu.index_select_tensor_dict(dataset, indices=list(range(4, 6))),
    ]

    i = 0

    for d in dataloader:
        tu.assert_tensordict_eq(d, expected_tensor_dict[i])
        i += 1

    data_iter_1 = tu.make_iterator(dataset, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": True})
    data_list_1 = []
    for data in data_iter_1:
        data_list_1.append(data)

    data_iter_2 = tu.make_iterator(dataset, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": True})
    data_list_2 = []
    for data in data_iter_2:
        data_list_2.append(data)

    for data1, data2 in zip(data_list_1, data_list_2, strict=True):
        tu.assert_tensordict_eq(data1, data2)


def test_reorder():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    non_tensor_dict = {"name": "abdce"}

    data = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict=non_tensor_dict)
    data = data[torch.tensor([3, 4, 2, 0, 1, 5])]

    assert torch.all(torch.eq(data["obs"], torch.tensor([4, 5, 3, 1, 2, 6])))
    assert np.all(data["labels"] == np.array(["d", "e", "c", "a", "b", "f"]))
    assert data["name"] == "abdce"


def test_chunk_concat():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    data = tu.get_tensordict({"obs": obs, "labels": labels}, non_tensor_dict={"name": "abcde"})

    data_split = data.tensor_split(indices_or_sections=5, dim=0)

    expected_idx_lst = [[0, 1], [2], [3], [4], [5]]

    for d, expected_idx in zip(data_split, expected_idx_lst, strict=False):
        tu.assert_tensordict_eq(d, data[expected_idx])

    data_split = data.chunk(2)
    assert len(data_split) == 2
    assert torch.all(torch.eq(data_split[0]["obs"], torch.tensor([1, 2, 3])))
    assert np.all(data_split[0]["labels"] == np.array(["a", "b", "c"]))
    assert data_split[0]["name"] == "abcde"

    assert torch.all(torch.eq(data_split[1]["obs"], torch.tensor([4, 5, 6])))
    assert np.all(data_split[1]["labels"] == np.array(["d", "e", "f"]))
    assert data_split[1]["name"] == "abcde"

    concat_data = torch.cat(data_split, dim=0)
    assert torch.all(torch.eq(concat_data["obs"], data["obs"]))
    assert np.all(concat_data["labels"] == data["labels"])
    assert concat_data["name"] == data["name"]

    data1 = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict={"name": "abcde"})
    data2 = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict={"name": "def"})
    data3 = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict={"name": "cfg"})

    output = torch.cat([data1, data2, data3], dim=0)

    # concat NonTensorData will keep the first one.
    assert output["name"] == "abcde"


def test_pop():
    obs = torch.randn(3, 10)
    act = torch.randn(3, 3)
    labels = ["a", ["b"], []]
    dataset = tu.get_tensordict({"obs": obs, "act": act, "labels": labels}, non_tensor_dict={"2": 2, "1": 1})

    dataset1 = copy.deepcopy(dataset)

    # test pop keys
    popped_dataset = tu.pop_keys(dataset, keys=["obs", "2"])

    assert popped_dataset.batch_size[0] == 3

    assert popped_dataset.keys() == {"obs", "2"}
    assert torch.all(torch.eq(popped_dataset["obs"], obs)).item()
    assert popped_dataset["2"] == 2

    assert dataset.keys() == {"act", "1", "labels"}

    # test pop non-exist key
    with pytest.raises(KeyError):
        tu.pop_keys(dataset, keys=["obs", "2"])

    # test single pop
    # NonTensorData
    assert tu.pop(dataset1, key="2") == 2
    # NonTensorStack
    assert tu.pop(dataset1, key="labels") == ["a", ["b"], []]
    # Tensor
    assert torch.all(torch.eq(tu.pop(dataset1, key="obs"), obs)).item()


def test_get():
    obs = torch.randn(3, 10)
    act = torch.randn(3, 3)
    labels = ["a", ["b"], []]
    dataset = tu.get_tensordict({"obs": obs, "act": act, "labels": labels}, non_tensor_dict={"2": 2, "1": 1})

    # test pop keys
    popped_dataset = tu.get_keys(dataset, keys=["obs", "2"])

    assert popped_dataset.batch_size[0] == 3

    assert torch.all(torch.eq(popped_dataset["obs"], dataset["obs"])).item()

    assert popped_dataset["2"] == dataset["2"]

    # test pop non-exist key
    with pytest.raises(KeyError):
        tu.get_keys(dataset, keys=["obs", "3"])

    # test single pop
    # NonTensorData
    assert tu.get(dataset, key="2") == 2
    # NonTensorStack
    assert tu.get(dataset, key="labels") == ["a", ["b"], []]
    # Tensor
    assert torch.all(torch.eq(tu.get(dataset, key="obs"), obs)).item()
    # Non-exist key
    assert tu.get(dataset, key="3", default=3) == 3


def test_repeat():
    # Create a DataProto object with some batch and non-tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = tu.get_tensordict({"obs": obs, "labels": labels}, non_tensor_dict={"info": "test_info"})

    # Test interleave=True
    repeated_data_interleave = data.repeat_interleave(repeats=2)
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ["a", "a", "b", "b", "c", "c"]

    assert torch.all(torch.eq(repeated_data_interleave["obs"], expected_obs_interleave))
    assert repeated_data_interleave["labels"] == expected_labels_interleave
    assert repeated_data_interleave["info"] == "test_info"

    # Test interleave=False
    repeated_data_no_interleave = data.repeat(2)
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
    expected_labels_no_interleave = ["a", "b", "c", "a", "b", "c"]

    assert torch.all(torch.eq(repeated_data_no_interleave["obs"], expected_obs_no_interleave))
    assert repeated_data_no_interleave["labels"] == expected_labels_no_interleave
    assert repeated_data_no_interleave["info"] == "test_info"


def test_dataproto_pad_unpad():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict={"info": "test_info"})

    padded_data, pad_size = tu.pad_to_divisor(data, size_divisor=2)

    assert pad_size == 1

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a"]

    assert torch.all(torch.eq(padded_data["obs"], expected_obs))
    assert padded_data["labels"] == expected_labels
    assert padded_data["info"] == "test_info"

    unpadd_data = tu.unpad(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data["obs"], obs))
    assert unpadd_data["labels"] == labels
    assert unpadd_data["info"] == "test_info"

    padded_data, pad_size = tu.pad_to_divisor(data, size_divisor=3)
    assert pad_size == 0

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    expected_labels = ["a", "b", "c"]

    assert torch.all(torch.eq(padded_data["obs"], expected_obs))
    assert padded_data["labels"] == expected_labels
    assert padded_data["info"] == "test_info"

    unpadd_data = tu.unpad(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data["obs"], obs))
    assert unpadd_data["labels"] == labels
    assert unpadd_data["info"] == "test_info"

    padded_data, pad_size = tu.pad_to_divisor(data, size_divisor=7)
    assert pad_size == 4

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a", "b", "c", "a"]
    assert torch.all(torch.eq(padded_data["obs"], expected_obs))
    assert padded_data["labels"] == expected_labels
    assert padded_data["info"] == "test_info"

    unpadd_data = tu.unpad(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data["obs"], obs))
    assert unpadd_data["labels"] == labels
    assert unpadd_data["info"] == "test_info"


def test_torch_save_data_proto():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = tu.get_tensordict({"obs": obs, "labels": labels}, non_tensor_dict={"info": "test_info"})

    filename = "test_data.pt"
    torch.save(data, filename)
    loaded_data = torch.load(filename, weights_only=False)

    assert torch.all(torch.eq(loaded_data["obs"], data["obs"]))
    assert loaded_data["labels"] == data["labels"]
    assert loaded_data["info"] == data["info"]

    import os

    os.remove(filename)


def test_len():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = np.array(["a", "b", "c"], dtype=object)

    data = tu.get_tensordict({"obs": obs, "labels": labels.tolist()}, non_tensor_dict={"info": "test_info"})
    assert len(data) == 3

    data = tu.get_tensordict({"labels": labels.tolist()}, non_tensor_dict={"info": "test_info"})
    assert len(data) == 3

    data_item = data[0]
    assert len(data_item) == 0

    data = tu.get_tensordict({}, non_tensor_dict={"info": "test_info"})
    assert len(data) == 0


def test_dataproto_index():
    data_len = 100
    idx_num = 10

    obs = torch.randn(data_len, 10)
    labels = [random.choice(["abc", "cde"]) for _ in range(data_len)]

    data = tu.get_tensordict({"obs": obs, "labels": labels})

    labels_np = np.array(labels)

    idx_np_int = np.random.randint(0, data_len, size=(idx_num,))
    result_np_int = data[idx_np_int]
    assert result_np_int.keys() == data.keys()
    assert result_np_int["obs"].shape[0] == idx_num
    assert len(result_np_int["labels"]) == idx_num
    assert np.array_equal(result_np_int["obs"].cpu().numpy(), obs[idx_np_int].numpy())
    assert np.array_equal(result_np_int["labels"], labels_np[idx_np_int])

    idx_torch_int = torch.randint(0, data_len, size=(idx_num,))
    result_torch_int = data[idx_torch_int]
    assert result_torch_int.keys() == data.keys()
    assert result_torch_int["obs"].shape[0] == idx_num
    assert len(result_torch_int["labels"]) == idx_num
    assert np.array_equal(result_torch_int["obs"].cpu().numpy(), obs[idx_torch_int].cpu().numpy())
    assert np.array_equal(result_torch_int["labels"], labels_np[idx_torch_int.cpu().numpy()])

    idx_list_int = [np.random.randint(0, data_len) for _ in range(idx_num)]
    result_list_int = data[idx_list_int]
    assert result_list_int.keys() == data.keys()
    assert result_list_int["obs"].shape[0] == idx_num
    assert len(result_list_int["labels"]) == idx_num
    assert np.array_equal(result_list_int["obs"].cpu().numpy(), obs[idx_list_int].cpu().numpy())
    assert np.array_equal(result_list_int["labels"], labels_np[idx_list_int])

    # idx_np_bool = np.random.randint(0, 2, size=(data_len,), dtype=bool)
    # result_np_bool = data[idx_np_bool]
    # assert result_np_bool.keys() == data.keys()
    # assert result_np_bool["obs"].shape[0] == idx_np_bool.sum()
    # assert len(result_np_bool["labels"]) == idx_np_bool.sum()
    # assert np.array_equal(result_np_bool["obs"].cpu().numpy(), obs[idx_np_bool].cpu().numpy())
    # assert np.array_equal(result_np_bool["labels"], labels_np[idx_np_bool])

    idx_torch_bool = torch.randint(0, 2, size=(data_len,), dtype=torch.bool)
    result_torch_bool = data[idx_torch_bool]
    assert result_torch_bool.keys() == data.keys()
    assert result_torch_bool["obs"].shape[0] == idx_torch_bool.sum().item()
    assert len(result_torch_bool["labels"]) == idx_torch_bool.sum().item()
    assert np.array_equal(result_torch_bool["obs"].cpu().numpy(), obs[idx_torch_bool].cpu().numpy())
    assert np.array_equal(result_torch_bool["labels"], labels_np[idx_torch_bool])

    # idx_list_bool = [np.random.randint(0, 2, dtype=bool) for _ in range(data_len)]
    # result_list_bool = data[idx_list_bool]
    # assert result_list_bool.keys() == data.keys()
    # assert result_list_bool["obs"].shape[0] == sum(idx_list_bool)
    # assert len(result_list_bool["labels"]) == sum(idx_list_bool)
    # assert np.array_equal(result_list_bool["obs"].cpu().numpy(), obs[idx_list_bool].cpu().numpy())
    # assert np.array_equal(result_list_bool["labels"], labels_np[idx_list_bool])


def test_select():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 3)
    dataset = tu.get_tensordict({"obs": obs, "act": act}, non_tensor_dict={"2": 2, "1": 1})

    subset = dataset.select("obs", "2")

    assert torch.all(torch.eq(subset["obs"], dataset["obs"]))
    assert subset["2"] == dataset["2"]
    assert "act" not in subset.keys()
    assert "1" not in subset.keys()


def test_dataproto_no_batch():
    labels = ["a", "b", "c"]
    data = tu.get_tensordict(tensor_dict={"labels": labels}, non_tensor_dict={"info": "test_info"})
    selected = data.select("labels")

    assert selected["labels"] == labels
    pop_data = tu.pop_keys(data, keys=["labels"])
    assert pop_data["labels"] == labels
    assert "labels" not in data


def test_sample_level_repeat():
    # Create a DataProto object with some batch and non-tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]

    data = tu.get_tensordict({"obs": obs, "labels": labels}, non_tensor_dict={"info": "test_info"})

    # list
    repeated_data_interleave = data.repeat_interleave(repeats=torch.tensor([3, 1, 2]))
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ["a", "a", "a", "b", "c", "c"]

    assert torch.all(torch.eq(repeated_data_interleave["obs"], expected_obs_interleave))
    assert repeated_data_interleave["labels"] == expected_labels_interleave
    assert repeated_data_interleave["info"] == "test_info"

    # torch.tensor
    repeated_data_no_interleave = data.repeat_interleave(repeats=torch.tensor([1, 2, 3]))
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]])
    expected_labels_no_interleave = ["a", "b", "b", "c", "c", "c"]

    assert torch.all(torch.eq(repeated_data_no_interleave["obs"], expected_obs_no_interleave))
    assert repeated_data_no_interleave["labels"] == expected_labels_no_interleave
    assert repeated_data_no_interleave["info"] == "test_info"


def test_dataproto_chunk_after_index():
    data_len = 4
    obs = torch.randn(data_len, 4)
    labels = [f"label_{i}" for i in range(data_len)]

    data = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict={"name": "abc"})
    # Test with boolean numpy array
    bool_mask = torch.tensor([True, False, True, False])
    selected = data[bool_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)  # int or List[int]

    # Test with integer numpy array
    int_mask = torch.tensor([0, 2])
    selected = data[int_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)

    # Test with boolean list
    list_mask = [True, False, True, False]
    selected = data[list_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)

    # Test with list
    list_mask = [0, 2]
    selected = data[list_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)

    # Test with torch tensor (bool)
    torch_bool_mask = torch.tensor([True, False, True, False])
    selected = data[torch_bool_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)

    # Test with torch tensor (int)
    torch_int_mask = torch.tensor([0, 2])
    selected = data[torch_int_mask]
    assert isinstance(selected.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch_size)


def test_concat_nested_tensor():
    # Test 2D nested tensors
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    c = torch.randint(low=0, high=vocab_size, size=(12,))
    d = torch.randint(low=0, high=vocab_size, size=(15,))

    nested_a_b = torch.nested.as_nested_tensor([a, b], layout=torch.jagged)
    nested_c_d = torch.nested.as_nested_tensor([c, d], layout=torch.jagged)

    output = tu.concat_nested_tensors([nested_a_b, nested_c_d])

    output_values = output.values()
    expected = torch.cat([a, b, c, d], dim=0)

    assert torch.all(torch.eq(output_values, expected)).item()

    # Test 3D nested tensors
    a_3d = torch.randint(low=0, high=vocab_size, size=(4, 4))
    b_3d = torch.randint(low=0, high=vocab_size, size=(4, 5))
    c_3d = torch.randint(low=0, high=vocab_size, size=(4, 6))
    d_3d = torch.randint(low=0, high=vocab_size, size=(4, 7))

    nested_a_b_3d = torch.nested.as_nested_tensor([a_3d, b_3d], layout=torch.jagged)
    nested_c_d_3d = torch.nested.as_nested_tensor([c_3d, d_3d], layout=torch.jagged)

    output_3d = tu.concat_nested_tensors([nested_a_b_3d, nested_c_d_3d])

    assert output_3d.shape[0] == 4
    output_3d_unbind = output_3d.unbind(0)
    assert torch.all(torch.eq(output_3d_unbind[0], a_3d)).item()
    assert torch.all(torch.eq(output_3d_unbind[1], b_3d)).item()
    assert torch.all(torch.eq(output_3d_unbind[2], c_3d)).item()
    assert torch.all(torch.eq(output_3d_unbind[3], d_3d)).item()

    # Test 4D nested tensors
    a_4d = torch.randint(low=0, high=vocab_size, size=(2, 3, 4))
    b_4d = torch.randint(low=0, high=vocab_size, size=(2, 3, 5))
    c_4d = torch.randint(low=0, high=vocab_size, size=(2, 3, 3))
    d_4d = torch.randint(low=0, high=vocab_size, size=(2, 3, 6))

    nested_a_b_4d = torch.nested.as_nested_tensor([a_4d, b_4d], layout=torch.jagged)
    nested_c_d_4d = torch.nested.as_nested_tensor([c_4d, d_4d], layout=torch.jagged)

    output_4d = tu.concat_nested_tensors([nested_a_b_4d, nested_c_d_4d])

    assert output_4d.shape[0] == 4
    output_4d_unbind = output_4d.unbind(0)
    assert torch.all(torch.eq(output_4d_unbind[0], a_4d)).item()
    assert torch.all(torch.eq(output_4d_unbind[1], b_4d)).item()
    assert torch.all(torch.eq(output_4d_unbind[2], c_4d)).item()
    assert torch.all(torch.eq(output_4d_unbind[3], d_4d)).item()


def test_concat_tensordict():
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    c = torch.randint(low=0, high=vocab_size, size=(12,))
    d = torch.randint(low=0, high=vocab_size, size=(15,))

    nested_a_b = torch.nested.as_nested_tensor([a, b], layout=torch.jagged)
    nested_c_d = torch.nested.as_nested_tensor([c, d], layout=torch.jagged)

    tensordict1 = tu.get_tensordict(
        tensor_dict={"input_ids": nested_a_b, "labels": ["a", "b"]}, non_tensor_dict={"temp": 1.0}
    )
    tensordict2 = tu.get_tensordict(
        tensor_dict={"input_ids": nested_c_d, "labels": ["c", "d"]}, non_tensor_dict={"temp": 2.0}
    )

    tensordict1_copy = copy.deepcopy(tensordict1)
    tensordict2_copy = copy.deepcopy(tensordict2)

    output = tu.concat_tensordict([tensordict1, tensordict2])

    assert torch.all(torch.eq(output["input_ids"].values(), torch.cat([a, b, c, d]))).item()
    assert output["labels"] == ["a", "b", "c", "d"]
    assert output["temp"] == 1.0

    # make sure tensordict1 and tensordict2 is untouched
    tu.assert_tensordict_eq(tensordict1, tensordict1_copy)
    tu.assert_tensordict_eq(tensordict2, tensordict2_copy)

    # test concat tensordict with only NonTensorStack and NonTensorData
    tensordict1 = tu.get_tensordict(tensor_dict={"labels": ["a", "b"]}, non_tensor_dict={"temp": 1.0})
    tensordict2 = tu.get_tensordict(tensor_dict={"labels": ["c", "d"]}, non_tensor_dict={"temp": 2.0})

    output = tu.concat_tensordict([tensordict1, tensordict2])

    assert output["labels"] == ["a", "b", "c", "d"]
    assert output["temp"] == 1.0

    assert output.batch_size[0] == 4

    # test concat tensordict with only NonTensorData
    tensordict1 = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"temp": 1.0})
    tensordict2 = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"temp": 2.0})

    output = tu.concat_tensordict([tensordict1, tensordict2])
    assert len(output.batch_size) == 0
    assert output["temp"] == 1.0


def test_chunk_tensordict():
    # Qwen-VL 3d position_ids
    position_ids = torch.nested.as_nested_tensor(
        [
            torch.arange(4).expand(4, 4),
            torch.arange(5).expand(4, 5),
            torch.arange(6).expand(4, 6),
            torch.arange(7).expand(4, 7),
        ],
        layout=torch.jagged,
    )
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(4), torch.arange(5), torch.arange(6), torch.arange(7)], layout=torch.jagged
    )
    attention_mask = torch.nested.as_nested_tensor(
        [
            torch.randint(low=0, high=2, size=[3, 4]),
            torch.randint(low=0, high=2, size=[3, 5]),
            torch.randint(low=0, high=2, size=[3, 6]),
            torch.randint(low=0, high=2, size=[3, 7]),
        ],
        layout=torch.jagged,
    )

    multi_modal_inputs = torch.stack(
        [
            NonTensorData({"pixel_values": torch.randn(3, 224, 224)}),
            NonTensorData(None),
            NonTensorData({"pixel_values": torch.randn(3, 128, 128)}),
            NonTensorData({"pixel_values": torch.randn(3, 128, 128)}),
        ]
    )
    td = tu.get_tensordict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "multi_modal_inputs": multi_modal_inputs,
        },
    )
    assert len(td) == 4
    chunks = tu.chunk_tensordict(td, chunks=2)

    for i, chunk in enumerate(chunks):
        assert len(chunk) == 2
        for key, val in chunk.items():
            if isinstance(val, torch.Tensor) and val.is_nested:
                tensors = td[key].unbind(dim=0)
                expected = torch.nested.as_nested_tensor(tensors[i * 2 : (i + 1) * 2], layout=torch.jagged)
                assert torch.all(torch.eq(val.values(), expected.values())).item()
            else:
                expected = td[key][i * 2 : (i + 1) * 2]
                for tensor, expect in zip(val, expected, strict=False):
                    if tensor.data is None:
                        assert expect is None
                    else:
                        assert torch.all(torch.eq(tensor.data["pixel_values"], expect["pixel_values"])).item()


def test_assign_non_tensor_stack_with_nested_lists():
    """Test assign_non_tensor_stack with lists of lists."""
    td = tu.get_tensordict({"obs": torch.randn(3, 4)}, non_tensor_dict={})

    # Lists of varying lengths (like turn_scores or tool_rewards)
    turn_scores = [[], [0.5, 0.8], [0.9]]
    tu.assign_non_tensor_stack(td, "turn_scores", turn_scores)

    # Verify data is accessible
    assert len(td["turn_scores"]) == 3
    assert list(td["turn_scores"][0]) == []
    assert list(td["turn_scores"][1]) == [0.5, 0.8]
    assert list(td["turn_scores"][2]) == [0.9]


def test_assign_non_tensor_stack_with_nested_dicts():
    """Test assign_non_tensor_stack with lists of dicts."""
    td = tu.get_tensordict({"obs": torch.randn(3, 4)}, non_tensor_dict={})

    # Lists of dicts (like reward_extra_info)
    reward_extra_info = [{"acc": 1.0, "loss": 0.1}, {"acc": 0.0, "loss": 0.9}, {"acc": 1.0, "loss": 0.05}]
    tu.assign_non_tensor_stack(td, "reward_extra_info", reward_extra_info)

    # Verify data is accessible
    assert len(td["reward_extra_info"]) == 3
    assert dict(td["reward_extra_info"][0]) == {"acc": 1.0, "loss": 0.1}
    assert dict(td["reward_extra_info"][1]) == {"acc": 0.0, "loss": 0.9}
    assert dict(td["reward_extra_info"][2]) == {"acc": 1.0, "loss": 0.05}


def test_assign_non_tensor_stack_with_complex_nested():
    """Test assign_non_tensor_stack with lists of lists of dicts."""
    td = tu.get_tensordict({"obs": torch.randn(2, 4)}, non_tensor_dict={})

    # Lists of lists of dicts (like raw_prompt)
    raw_prompt = [
        [{"content": "Question 1", "role": "user"}],
        [{"content": "Question 2", "role": "user"}, {"content": "Answer 2", "role": "assistant"}],
    ]
    tu.assign_non_tensor_stack(td, "raw_prompt", raw_prompt)

    # Verify data is accessible
    assert len(td["raw_prompt"]) == 2
    assert len(td["raw_prompt"][0]) == 1
    assert dict(td["raw_prompt"][0][0]) == {"content": "Question 1", "role": "user"}
    assert len(td["raw_prompt"][1]) == 2
    assert dict(td["raw_prompt"][1][0]) == {"content": "Question 2", "role": "user"}


def test_assign_non_tensor_handles_wrappers():
    td = tu.get_tensordict({"obs": torch.randn(3, 4)}, non_tensor_dict={})

    meta = {"top_p": 0.8}
    tu.assign_non_tensor(td, **meta)
    assert td["top_p"] == 0.8

    wrapped = NonTensorData(0.3)
    stack = NonTensorStack.from_list([NonTensorData(1.0), NonTensorData(2.0), NonTensorData(3.0)])
    tu.assign_non_tensor(td, wrapped=wrapped, stack=stack)

    assert td["wrapped"] == 0.3
    assert td["stack"] == [1.0, 2.0, 3.0]


def test_assign_non_tensor_stack_batch_size_check():
    td = tu.get_tensordict({"obs": torch.randn(3, 4)}, non_tensor_dict={})
    stack = NonTensorStack.from_list([NonTensorData(1.0), NonTensorData(2.0)])

    with pytest.raises(RuntimeError):
        tu.assign_non_tensor(td, stack=stack)


def test_assign_non_tensor_with_auto_detection():
    """Test assign_non_tensor automatically detects and handles nested structures."""
    td = tu.get_tensordict({"obs": torch.randn(3, 4)}, non_tensor_dict={})

    # Mix of simple and nested data
    tu.assign_non_tensor(
        td,
        metadata="experiment_1",  # Simple value
        turn_scores=[[], [0.5, 0.8], [0.9]],  # Nested list
        reward_extra_info=[{"acc": 1.0}, {"acc": 0.0}, {"acc": 1.0}],  # List of dicts
        simple_list=["a", "b", "c"],  # Simple list (also uses NonTensorStack for consistency)
    )

    # Verify all data is accessible
    assert td["metadata"] == "experiment_1"
    assert len(td["turn_scores"]) == 3
    assert list(td["turn_scores"][1]) == [0.5, 0.8]
    assert len(td["reward_extra_info"]) == 3
    assert dict(td["reward_extra_info"][0]) == {"acc": 1.0}
    assert len(td["simple_list"]) == 3
    assert td["simple_list"][0] == "a"


def test_get_tensordict_with_nested_lists():
    """Test get_tensordict automatically handles nested lists."""
    obs = torch.randn(3, 4)
    turn_scores = [[], [0.5, 0.8], [0.9]]

    # This should automatically convert turn_scores to NonTensorStack
    td = tu.get_tensordict({"obs": obs, "turn_scores": turn_scores})

    # Verify tensors and nested data are both accessible
    assert torch.all(torch.eq(td["obs"], obs))
    assert len(td["turn_scores"]) == 3
    assert list(td["turn_scores"][0]) == []
    assert list(td["turn_scores"][1]) == [0.5, 0.8]


def test_get_tensordict_with_nested_dicts():
    """Test get_tensordict automatically handles lists of dicts."""
    obs = torch.randn(3, 4)
    reward_extra_info = [{"acc": 1.0}, {"acc": 0.0}, {"acc": 1.0}]

    td = tu.get_tensordict({"obs": obs, "reward_extra_info": reward_extra_info})

    assert torch.all(torch.eq(td["obs"], obs))
    assert len(td["reward_extra_info"]) == 3
    assert dict(td["reward_extra_info"][0]) == {"acc": 1.0}


def test_get_tensordict_with_complex_nested_structures():
    """Test get_tensordict with lists of lists of dicts."""
    obs = torch.randn(2, 4)
    raw_prompt = [
        [{"content": "Q1", "role": "user"}],
        [{"content": "Q2", "role": "user"}, {"content": "A2", "role": "assistant"}],
    ]

    td = tu.get_tensordict({"obs": obs, "raw_prompt": raw_prompt})

    assert torch.all(torch.eq(td["obs"], obs))
    assert len(td["raw_prompt"]) == 2
    assert dict(td["raw_prompt"][0][0]) == {"content": "Q1", "role": "user"}


def test_get_tensordict_agent_loop_scenario():
    """Test the complete agent loop scenario with all nested types.

    This simulates the exact use case from agent loops with:
    - turn_scores: lists of lists
    - reward_extra_info: lists of dicts
    - raw_prompt: lists of lists of dicts
    - tool_rewards: lists of lists
    """
    prompts = torch.randn(2, 10)
    responses = torch.randn(2, 5)

    # Nested structures from agent loop
    data_source = ["lighteval/MATH", "lighteval/MATH"]
    uid = ["uuid-1", "uuid-2"]
    turn_scores = [[], [0.5, 0.8]]  # Lists of varying lengths
    reward_extra_info = [{"acc": 1.0, "loss": 0.1}, {"acc": 0.0, "loss": 0.9}]
    raw_prompt = [
        [{"content": "Compute 4 @ 2", "role": "user"}],
        [{"content": "Compute 8 @ 7", "role": "user"}],
    ]
    tool_rewards = [[0.0], []]  # List of lists

    # This should handle all nested structures automatically
    td = tu.get_tensordict(
        tensor_dict={
            "prompts": prompts,
            "responses": responses,
            "data_source": data_source,
            "uid": uid,
            "turn_scores": turn_scores,
            "reward_extra_info": reward_extra_info,
            "raw_prompt": raw_prompt,
            "tool_rewards": tool_rewards,
        },
        non_tensor_dict={"global_steps": 42},
    )

    # Verify all data types are accessible
    assert torch.all(torch.eq(td["prompts"], prompts))
    assert torch.all(torch.eq(td["responses"], responses))
    assert td["data_source"] == data_source
    assert td["uid"] == uid

    # Verify nested structures
    assert len(td["turn_scores"]) == 2
    assert list(td["turn_scores"][0]) == []
    assert list(td["turn_scores"][1]) == [0.5, 0.8]

    assert len(td["reward_extra_info"]) == 2
    assert dict(td["reward_extra_info"][0]) == {"acc": 1.0, "loss": 0.1}

    assert len(td["raw_prompt"]) == 2
    assert dict(td["raw_prompt"][0][0]) == {"content": "Compute 4 @ 2", "role": "user"}

    assert len(td["tool_rewards"]) == 2
    assert list(td["tool_rewards"][0]) == [0.0]
    assert list(td["tool_rewards"][1]) == []

    # Verify metadata
    assert td["global_steps"] == 42


def test_contiguous():
    # create a tensordict that contains normal tensor, nested tensor,
    # nontensorstack with numpy, nontensorstack with tensor, NonTensorData with numpy and NonTensorData with tensor

    a = torch.randn(3, 4)  # contiguous tensor
    b = torch.randn(3, 4)[:, :-1]  # non contiguous tensor
    c = torch.nested.as_nested_tensor([torch.randn(3), torch.randn(4), torch.randn(5)], layout=torch.jagged)

    d = torch.randn(10, 12)
    e = torch.randn(11, 12)
    f = torch.randn(13, 12)

    data = tu.get_tensordict(
        tensor_dict={"a": a, "b": b, "c": c, "nt": [{"pixel": d}, {"pixel": e}, {"pixel": f}]},
        non_tensor_dict={"ntd": a.clone()},
    )

    with pytest.raises(RuntimeError):
        # b is not contiguous
        data.consolidate()

    data1 = copy.deepcopy(data)
    data_cont = tu.contiguous(data1)

    tu.assert_tensordict_eq(data_cont, data)

    data_cont.consolidate()

    tu.assert_tensordict_eq(data_cont, data)
