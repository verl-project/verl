# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.model_merger.fsdp_model_merger import merge_non_dtensor_shards


def test_replicated_scalar_buffer_is_kept_as_is():
    # A 0-d buffer replicated by FSDP2: torch.cat would raise on it.
    shards = [torch.tensor(-6.375) for _ in range(8)]
    merged = merge_non_dtensor_shards("clamp_min", shards)
    assert merged.shape == ()
    assert merged.item() == -6.375


def test_replicated_1d_buffer_is_not_concatenated():
    # A (1,) buffer replicated by FSDP2: concatenating would silently give (8,).
    shards = [torch.tensor([0.061]) for _ in range(8)]
    assert merge_non_dtensor_shards("layer_scalar", shards).shape == (1,)


def test_single_rank_checkpoint_returns_the_tensor():
    # world_size == 1 (the case the old torch.cat fallback was added for): one plain copy.
    tensor = torch.arange(6.0).reshape(2, 3)
    assert torch.equal(merge_non_dtensor_shards("weight", [tensor]), tensor)


def test_ranks_disagreeing_is_an_error_naming_the_key():
    same_shape = [torch.full((2, 3), float(rank)) for rank in range(4)]
    with pytest.raises(ValueError, match="'suspicious'.*rank 1"):
        merge_non_dtensor_shards("suspicious", same_shape)

    different_shape = [torch.zeros(rank + 1) for rank in range(4)]
    with pytest.raises(ValueError, match="'ragged'"):
        merge_non_dtensor_shards("ragged", different_shape)
