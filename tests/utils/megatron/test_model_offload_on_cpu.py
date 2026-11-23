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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils import megatron_utils  # noqa: E402


def test_offload_keeps_storage_already_on_cpu(monkeypatch):
    monkeypatch.setattr(megatron_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.grad = torch.ones_like(module.weight)
    weight, grad = module.weight.detach().clone(), module.weight.grad.clone()
    weight_storage = module.weight.untyped_storage()
    grad_storage = module.weight.grad.untyped_storage()

    for _ in range(2):
        megatron_utils.offload_megatron_model_to_cpu([module])
        assert module.weight.untyped_storage() is weight_storage
        assert module.weight.grad.untyped_storage() is grad_storage
        assert weight_storage.nbytes() == weight.nbytes
        assert grad_storage.nbytes() == grad.nbytes
        torch.testing.assert_close(module.weight, weight)
        torch.testing.assert_close(module.weight.grad, grad)
