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
from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine_workers import _release_trainer_grads


@pytest.mark.parametrize("with_optimizer", [True, False], ids=["optimizer", "no-optimizer"])
def test_release_trainer_grads_drops_grads_and_returns_nothing(with_optimizer):
    a = torch.nn.Parameter(torch.ones(2))
    a.grad = torch.ones(2)
    b = torch.nn.Parameter(torch.ones(2))
    b.main_grad = torch.ones(2)
    opt = torch.optim.SGD([a, b], lr=0.1) if with_optimizer else None
    assert _release_trainer_grads(SimpleNamespace(handle=SimpleNamespace(_optimizer=opt))) is None
    assert (a.grad is None and b.main_grad is None) is with_optimizer
