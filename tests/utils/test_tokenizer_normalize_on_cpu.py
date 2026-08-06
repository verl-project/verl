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

import numpy as np
import pytest

from verl.utils.tokenizer import normalize_token_ids


class DummyBatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class DummyToList:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


@pytest.mark.parametrize(
    ("tokenized_output", "expected"),
    [
        # transformers v4-style direct token ids
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        # common list-like outputs with tolist()/ndarray paths
        (DummyToList([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3], dtype=np.int64), [1, 2, 3]),
        # transformers v5-like mapping / BatchEncoding-style outputs
        ({"input_ids": [1, 2, 3]}, [1, 2, 3]),
        ({"input_ids": DummyToList([1, 2, 3])}, [1, 2, 3]),
        ({"input_ids": [[1, 2, 3]]}, [1, 2, 3]),
        (DummyBatchEncoding([1, 2, 3]), [1, 2, 3]),
        (DummyBatchEncoding(DummyToList([[1, 2, 3]])), [1, 2, 3]),
        # scalar item() support
        ([np.int64(1), np.int32(2), np.int16(3)], [1, 2, 3]),
    ],
)
def test_normalize_token_ids_valid_outputs(tokenized_output, expected):
    assert normalize_token_ids(tokenized_output) == expected


@pytest.mark.parametrize(
    "tokenized_output",
    [
        "not-token-ids",
        {"attention_mask": [1, 1, 1]},
        [[1, 2], [3, 4]],  # ambiguous batched ids should fail fast
        [1, object(), 3],
    ],
)
def test_normalize_token_ids_invalid_outputs(tokenized_output):
    with pytest.raises(TypeError):
        normalize_token_ids(tokenized_output)
