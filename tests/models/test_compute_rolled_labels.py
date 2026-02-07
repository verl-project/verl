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

import pytest
import torch

from verl.models.transformers.dense_common import compute_rolled_labels


def test_rolled_labels_from_labels():
    """When labels is provided, rolled_labels should be rolled from labels."""
    labels = torch.tensor([[10, 20, 30, 40, 50]])
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    result = compute_rolled_labels(input_ids, labels, "test_backend")
    expected = torch.roll(labels, shifts=-1, dims=-1)
    torch.testing.assert_close(result, expected)


def test_rolled_labels_from_input_ids():
    """When labels is None, rolled_labels should be rolled from input_ids."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    result = compute_rolled_labels(input_ids, labels=None, backend_name="test_backend")
    expected = torch.roll(input_ids, shifts=-1, dims=-1)
    torch.testing.assert_close(result, expected)


def test_rolled_labels_both_none_raises():
    """When both input_ids and labels are None, should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="test_backend"):
        compute_rolled_labels(input_ids=None, labels=None, backend_name="test_backend")


def test_rolled_labels_shape_preserved():
    """Output shape should match input shape (no SP slicing when sp_size=1)."""
    input_ids = torch.randint(0, 1000, (2, 128))
    result = compute_rolled_labels(input_ids, labels=None, backend_name="test_backend")
    assert result.shape == input_ids.shape


def test_rolled_labels_roll_correctness():
    """Verify roll shifts left by 1: [a, b, c, d] -> [b, c, d, a]."""
    labels = torch.tensor([[1, 2, 3, 4]])
    result = compute_rolled_labels(None, labels, "test_backend")
    expected = torch.tensor([[2, 3, 4, 1]])
    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
