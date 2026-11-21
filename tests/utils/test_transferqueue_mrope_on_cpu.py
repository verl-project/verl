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

import torch
from tensordict import TensorDict

from verl.utils.tensordict_utils import index_select_tensor_dict
from verl.utils.transferqueue_utils import _normalize_tq_position_ids


def test_repack_mrope_preserves_values_and_microbatch_sequence_axis():
    samples = [torch.arange(15).reshape(3, 5), torch.arange(21).reshape(3, 7)]
    data = TensorDict({"position_ids": torch.nested.nested_tensor(samples)}, batch_size=[])
    _normalize_tq_position_ids(data)
    positions = data["position_ids"]
    assert positions._ragged_idx == 2
    for actual, expected in zip(positions.unbind(0), samples, strict=True):
        torch.testing.assert_close(actual, expected)
    data.batch_size = [2]
    selected = index_select_tensor_dict(data, [1])
    torch.testing.assert_close(selected["position_ids"].unbind(0)[0], samples[1])
    _normalize_tq_position_ids(data)
    assert data["position_ids"] is positions


def test_dense_positions_are_unchanged():
    positions = torch.arange(30).reshape(2, 3, 5)
    data = TensorDict({"position_ids": positions}, batch_size=[])
    _normalize_tq_position_ids(data)
    assert data["position_ids"] is positions


def test_repack_true_axis_one_jagged_positions():
    samples = [torch.arange(21).reshape(3, 7), torch.arange(21, 42).reshape(3, 7)]
    positions = torch.nested.nested_tensor(samples, layout=torch.jagged)
    assert positions._ragged_idx == 1
    data = TensorDict({"position_ids": positions}, batch_size=[2])
    _normalize_tq_position_ids(data)
    assert data["position_ids"]._ragged_idx == 2
    selected = index_select_tensor_dict(data, [1])
    torch.testing.assert_close(selected["position_ids"].unbind(0)[0], samples[1])
