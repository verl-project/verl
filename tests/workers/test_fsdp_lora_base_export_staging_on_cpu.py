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

"""Which FSDP2 LoRA weight exports stage the whole module on the device first."""

from types import SimpleNamespace

import pytest

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


@pytest.fixture
def staging_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(transformer_impl, "load_fsdp_model_to_gpu", lambda module: calls.append("load"))
    monkeypatch.setattr(transformer_impl, "offload_fsdp_model_to_cpu", lambda module: calls.append("offload"))
    monkeypatch.setattr(transformer_impl, "collect_lora_params", lambda **kwargs: {})
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    return calls


def _lora_engine():
    peft_config = SimpleNamespace(to_dict=lambda: {})
    return SimpleNamespace(
        module=SimpleNamespace(peft_config={"default": peft_config}),
        model_config=SimpleNamespace(lora={"merge": False}, hf_config=SimpleNamespace(model_type="llama")),
        _uses_fsdp2_cpu_offload_policy=False,
        _is_offload_param=True,
        _qat_enabled=False,
    )


def test_lora_base_export_streams_shards_without_staging(staging_calls):
    # Staging would leave the streamed DTensor refs holding the GPU copy of the shard after the offload.
    FSDPEngine.get_per_tensor_param(_lora_engine(), layered_summon=False, base_sync_done=False)
    assert staging_calls == []


def test_lora_adapter_export_still_stages(staging_calls):
    FSDPEngine.get_per_tensor_param(_lora_engine(), layered_summon=False, base_sync_done=True)
    assert staging_calls == ["load", "offload"]
