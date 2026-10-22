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
import sys
import types

import pytest
import torch

from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine.utils import enable_batch_invariance


def test_batch_invariant_is_off_by_default():
    assert FSDPEngineConfig().batch_invariant is False
    assert FSDPEngine._batch_invariant is False


def test_batch_invariant_rejects_fused_kernels(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(transformer_impl.FSDPEngine, "_init_device_mesh", lambda self: None)
    monkeypatch.setattr("verl.workers.engine.fsdp.utils.apply_npu_fsdp_patches", lambda model_config: None)
    model_config = types.SimpleNamespace(use_remove_padding=True, use_fused_kernels=True)

    with pytest.raises(ValueError, match="use_fused_kernels"):
        FSDPEngine(model_config, FSDPEngineConfig(batch_invariant=True), None, None)


def test_enable_batch_invariance_installs_vllm_overrides_and_reverts_its_nccl_profile(monkeypatch):
    calls = []

    def init_batch_invariance():
        calls.append(os.environ["VLLM_BATCH_INVARIANT"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["NCCL_MAX_NCHANNELS"] = "1"
        os.environ["NCCL_PROTO"] = "Simple"

    fake = types.ModuleType("vllm.model_executor.determinism.batch_invariant")
    fake.init_batch_invariance = init_batch_invariance
    monkeypatch.setitem(sys.modules, "vllm.model_executor.determinism.batch_invariant", fake)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("NCCL_MAX_NCHANNELS", raising=False)
    monkeypatch.setenv("NCCL_PROTO", "LL128")

    enable_batch_invariance()

    assert calls == ["1"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert "NCCL_MAX_NCHANNELS" not in os.environ
    assert os.environ["NCCL_PROTO"] == "LL128"
