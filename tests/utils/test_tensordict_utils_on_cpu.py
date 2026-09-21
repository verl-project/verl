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

import torch
from tensordict import NonTensorStack

from verl.utils.tensordict_utils import list_of_dict_to_tensordict


def _sample(seq_len: int) -> dict:
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "seq_len": torch.tensor(seq_len, dtype=torch.long),
        "data_source": "gsm8k",
    }


def test_single_element_list_keeps_ragged_fields_nested():
    # A length-1 list makes ``all(item.shape == val_list[0].shape ...)`` vacuously
    # true, which used to stack the single ragged tensor into a dense [1, L] and
    # break downstream ``.offsets()`` calls.
    td = list_of_dict_to_tensordict([_sample(5)])

    assert td["input_ids"].is_nested
    assert td["input_ids"].offsets().tolist() == [0, 5]


def test_same_length_ragged_fields_stay_nested():
    # Several genuinely-ragged items that coincidentally share a length (e.g. a
    # group of rollout responses all saturating max_response_length) must not be
    # collapsed to a dense tensor.
    td = list_of_dict_to_tensordict([_sample(7), _sample(7), _sample(7)])

    assert td["input_ids"].is_nested
    assert td["input_ids"].offsets().tolist() == [0, 7, 14, 21]


def test_variable_length_ragged_fields_stay_nested():
    td = list_of_dict_to_tensordict([_sample(3), _sample(6), _sample(4)])

    assert td["input_ids"].is_nested
    assert td["input_ids"].offsets().tolist() == [0, 3, 9, 13]
    assert td["input_ids"][1].tolist() == [0, 1, 2, 3, 4, 5]


def test_scalar_fields_are_stacked_dense():
    td = list_of_dict_to_tensordict([_sample(3), _sample(6)])

    assert not td["seq_len"].is_nested
    assert td["seq_len"].tolist() == [3, 6]


def test_non_tensor_fields_use_non_tensor_stack():
    td = list_of_dict_to_tensordict([_sample(3), _sample(6)])

    assert isinstance(td.get("data_source"), NonTensorStack)
    assert td.get("data_source").tolist() == ["gsm8k", "gsm8k"]
