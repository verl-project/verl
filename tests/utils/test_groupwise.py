# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import os

os.environ.setdefault("VERL_FORCE_DEVICE", "cpu")  # ensure CPU for tests

import numpy as np
import pytest
import torch

from verl.utils import as_torch_index, group_mean_std


def test_as_torch_index_basic_integers():
    g = as_torch_index([2, 2, 5, 7, 5, 2])
    assert g.dtype == torch.long
    assert g.device.type == "cpu"
    # Values should be contiguous 0..G-1, keeping equal labels equal
    assert g.tolist()[0] == g.tolist()[1]
    assert len(torch.unique(g)) == 3  # {2,5,7} -> 3 groups
    assert g.tolist() == [0, 0, 1, 2, 1, 0]


def test_as_torch_index_near_integer_floats():
    arr = np.array([1.0000001, 2.0, 1.0, 3.0000000001], dtype=np.float64)
    g = as_torch_index(arr)  # should round to integers then factorize
    assert g.dtype == torch.long
    assert len(torch.unique(g)) == 3  # {1,2,3}
    assert g.tolist() == [0, 1, 0, 2]


@pytest.mark.parametrize(
    "label,index",
    [
        ("python int list", [2, 2, 5, 7, 5, 2]),
        ("numpy int array", np.array([10, 10, 20, 20, 30])),
        ("torch int tensor", torch.tensor([3, 3, 9, 9, 12])),
        ("torch bool tensor", torch.tensor([True, True, False, False, True])),
        ("near-integer floats", np.array([1.0000001, 2.0, 1.0, 3.0])),
        ("numeric strings", np.array(["7", "7", "9", "9"], dtype=object)),
        ("uuid-like strings", np.array(["uid-a", "uid-a", "uid-b", "uid-b"], dtype=object)),
        ("negative ints", np.array([-1, -1, 3, 3])),
        ("sparse ints", np.array([1000, 1000, 1001, 1001])),
    ],
)
def test_as_torch_index_always_returns_dense_ids(label, index):
    """Every recognized label kind must satisfy the documented [0..G-1] contract.

    The integer / near-integer-float / numeric-string fast paths used to return the raw
    label values, which are not usable as the positional group indices that
    ``group_mean_std`` indexes against.
    """
    g = as_torch_index(index)
    values = g.tolist()
    n_groups = len(set(values))
    assert g.dtype == torch.long, label
    assert sorted(set(values)) == list(range(n_groups)), f"{label}: {values} is not contiguous [0..G-1]"


def test_as_torch_index_preserves_label_equality():
    g = as_torch_index(np.array([-5, 7, -5, 7, 100]))
    assert g[0] == g[2]
    assert g[1] == g[3]
    assert g[4] not in (g[0].item(), g[1].item())


def test_group_mean_std_output_is_sized_by_group_count_not_label_magnitude():
    """G must be the number of groups, not max(label) + 1."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    gidx = as_torch_index([1000, 1000, 1001, 1001])

    mean_g, std_g, cnt_g = group_mean_std(scores, gidx)

    assert mean_g.numel() == 2
    assert std_g.numel() == 2
    assert torch.allclose(mean_g, torch.tensor([1.5, 3.5]))
    assert torch.equal(cnt_g, torch.tensor([2.0, 2.0]))


def test_group_mean_std_handles_negative_labels_via_as_torch_index():
    """Negative raw labels used to raise IndexError from index_add_."""
    scores = torch.tensor([1.0, 3.0, 10.0, 20.0], dtype=torch.float32)
    gidx = as_torch_index(np.array([-1, -1, 3, 3]))

    mean_g, _, cnt_g = group_mean_std(scores, gidx)

    assert torch.allclose(mean_g, torch.tensor([2.0, 15.0]))
    assert torch.equal(cnt_g, torch.tensor([2.0, 2.0]))


def test_group_mean_std_rejects_raw_negative_indices():
    """A caller bypassing as_torch_index gets an actionable error, not IndexError."""
    scores = torch.tensor([1.0, 2.0], dtype=torch.float32)
    gidx = torch.tensor([-1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="as_torch_index"):
        group_mean_std(scores, gidx)


def test_as_torch_index_factorization_mixed():
    labels = ["a", "b", "a", "c", "0042", 42]
    g = as_torch_index(labels)
    # "0042" and 42 should NOT be the same group (strings are not coerced here)
    assert g.tolist()[4] != g.tolist()[5]
    assert len(torch.unique(g)) == 5


def test_group_mean_std_simple():
    # groups: 0 -> [1, 3], 1 -> [2]
    scores = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    gidx = as_torch_index([0, 1, 0])

    mean_g, std_g, cnt_g = group_mean_std(scores, gidx)
    # group 0: mean = (1+3)/2 = 2
    # sample std (unbiased) = sqrt( (sum(x^2) - (sum(x)^2)/n) / (n-1) )
    # = sqrt( (1^2+3^2) - (1+3)^2/2 ) / (2-1) = sqrt(10 - 16/2) = sqrt(2)
    assert torch.allclose(mean_g, torch.tensor([2.0, 0.0]))
    assert torch.allclose(cnt_g, torch.tensor([2.0, 1.0]))
    # singleton group -> std = 1.0
    assert mean_g[1].item() == 0.0
    assert std_g[1].item() == 1.0
    assert pytest.approx(std_g[0].item(), rel=1e-6) == (2.0**0.5)


def test_group_mean_std_low_variance_matches_torch_std():
    scores = torch.tensor([1.0, 1.00001, 2.0, 2.00001], dtype=torch.float32)
    gidx = as_torch_index(["prompt-a", "prompt-a", "prompt-b", "prompt-b"])

    mean_g, std_g, cnt_g = group_mean_std(scores, gidx, eps=0.0)

    assert torch.allclose(mean_g, torch.tensor([1.000005, 2.000005]), rtol=1e-5, atol=1e-6)
    assert torch.equal(cnt_g, torch.tensor([2.0, 2.0]))
    assert torch.allclose(std_g[0], torch.std(scores[:2]), rtol=1e-5, atol=1e-6)
    assert torch.allclose(std_g[1], torch.std(scores[2:]), rtol=1e-5, atol=1e-6)


def test_group_mean_std_empty():
    scores = torch.tensor([], dtype=torch.float32)
    gidx = torch.tensor([], dtype=torch.long)
    mean_g, std_g, cnt_g = group_mean_std(scores, gidx)
    assert mean_g.numel() == 0 and std_g.numel() == 0 and cnt_g.numel() == 0


def test_group_mean_std_default_device_no_force_env(monkeypatch):
    """
    Regression test:
    - group_mean_std(device=None) must not pass a device *module* (e.g., torch.cuda)
      into Tensor.to(device=...), which crashes with:
      TypeError: to() received an invalid combination of arguments - got (..., device=module, ...)
    """
    # Simulate a non-pytest environment (training code path) while keeping the test CPU-only.
    monkeypatch.delenv("VERL_FORCE_DEVICE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    # Force device selection to CPU even if CUDA is available on the test machine.
    # Must patch the reference in groupwise module directly (it uses `from ... import get_device_name`).
    import verl.utils.groupwise as groupwise_mod

    monkeypatch.setattr(groupwise_mod, "get_device_name", lambda: "cpu")

    scores = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    gidx = torch.tensor([0, 1, 0], dtype=torch.long)

    mean_g, std_g, cnt_g = group_mean_std(scores, gidx)
    assert mean_g.device.type == "cpu"
    assert std_g.device.type == "cpu"
    assert cnt_g.device.type == "cpu"
