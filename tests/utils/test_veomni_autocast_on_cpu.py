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

from verl.workers.engine.fsdp import transformer_impl as fsdp
from verl.workers.engine.veomni import transformer_impl as veomni


@pytest.mark.parametrize("enabled", [True, False])
def test_veomni_controls_inherited_autocast(monkeypatch, enabled):
    monkeypatch.setattr(fsdp, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(veomni, "get_device_name", lambda: "cpu")
    engine = object.__new__(veomni.VeOmniEngineWithLMHead)
    engine.engine_config = SimpleNamespace(enable_autocast=enabled)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with engine._get_forward_autocast_context():
            assert torch.is_autocast_enabled("cpu") == enabled
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_plain_fsdp_uses_its_own_dtype(monkeypatch, dtype):
    monkeypatch.setattr(fsdp, "get_device_name", lambda: "cpu")
    engine = object.__new__(fsdp.FSDPEngineWithLMHead)
    engine._autocast_dtype = dtype
    engine.engine_config = SimpleNamespace(enable_autocast=False)
    with engine._get_forward_autocast_context():
        assert torch.is_autocast_enabled("cpu") == (dtype == torch.bfloat16)
