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


def test_as_torch_index_near_integer_floats():
    arr = np.array([1.0000001, 2.0, 1.0, 3.0000000001], dtype=np.float64)
    g = as_torch_index(arr)  # should round to integers then factorize
    assert g.dtype == torch.long
    assert len(torch.unique(g)) == 3  # {1,2,3}


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
    import verl.utils.device as device_mod

    monkeypatch.setattr(device_mod, "is_cuda_available", False)
    monkeypatch.setattr(device_mod, "is_npu_available", False)

    scores = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    gidx = torch.tensor([0, 1, 0], dtype=torch.long)

    mean_g, std_g, cnt_g = group_mean_std(scores, gidx)
    assert mean_g.device.type == "cpu"
    assert std_g.device.type == "cpu"
    assert cnt_g.device.type == "cpu"
