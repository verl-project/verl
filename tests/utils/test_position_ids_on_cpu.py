# Copyright 2026 Individual Contributor: Egan
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

import torch
from tensordict import TensorDict

from verl.utils.tensordict_utils import maybe_fix_3d_position_ids, nested_tensor_from_tensor_list


def test_position_ids_after_tensordict_serialization():
    expected = [torch.arange(24).reshape(3, 8), torch.arange(33).reshape(3, 11)]
    data = TensorDict({"position_ids": nested_tensor_from_tensor_list(expected, ragged_idx=2)}, batch_size=[2])
    data = pickle.loads(pickle.dumps(data.consolidate()))

    maybe_fix_3d_position_ids(data)

    for actual, sample in zip(data["position_ids"].unbind(), expected, strict=True):
        torch.testing.assert_close(actual, sample)
