# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

_TENSOR_1MB = torch.zeros(512, 512)
_BYTES_1MB = 1 << 20


@pytest.mark.parametrize(
    "named_tensors, bucket_size_mb, gt_groups",
    [
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            0.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            2 * _BYTES_1MB,
            [["a", "b"]],
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_named_tensor_buckets(named_tensors, bucket_size_mb, gt_groups: list[list[str]]):
    named_tensors_iter = iter(named_tensors)
    groups = [g async for g in get_named_tensor_buckets(named_tensors_iter, bucket_size_mb)]
    assert len(groups) == len(gt_groups)
    for group, gt_group in zip(groups, gt_groups, strict=True):
        assert len(group) == len(gt_group)
        for (name, _), (gt_name) in zip(group, gt_group, strict=True):
            assert name == gt_name
