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

import os

import pytest
import torch

from verl.workers.engine.utils import enable_full_determinism


@pytest.fixture
def restore_determinism_state():
    env = dict(os.environ)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    yield
    os.environ.clear()
    os.environ.update(env)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled = cudnn


def test_full_determinism_does_not_fill_uninitialized_memory(restore_determinism_state):
    enable_full_determinism(seed=0)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory is False
