# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU wiring test: Megatron-Bridge provider path defers DDP for Muon layer-wise."""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest
import torch.nn as nn

from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module


class _DummyConfig:
    def __init__(self):
        self.fp16 = False
        self.bf16 = False


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.config = _DummyConfig()


@dataclasses.dataclass
class _FakeDDPConfig:
    use_distributed_optimizer: bool = True
    use_layer_wise_param_layout: bool = False


def _minimal_hf_config():
    cfg = types.SimpleNamespace()
    cfg.hidden_size = 4
    cfg.rope_theta = 10000.0
    return cfg


@pytest.fixture
def megatron_bridge_stubs(monkeypatch):
    """Stub megatron.bridge imports used by the provider branch."""
    create_ddp_config_calls = []
    provide_calls = []
    layerwise_wrap_calls = []

    def fake_create_ddp_config(**kwargs):
        create_ddp_config_calls.append(kwargs)
        return _FakeDDPConfig(use_distributed_optimizer=kwargs.get("use_distributed_optimizer", True))

    class FakeProvider:
        fp16 = False
        bf16 = False
        sequence_parallel = False

        def register_pre_wrap_hook(self, _hook):
            return None

        def provide_distributed_model(self, **kwargs):
            provide_calls.append(kwargs)
            return [_DummyModel()]

    bridge_mod = types.ModuleType("megatron.bridge.training.utils.config_utils")
    bridge_mod.create_ddp_config = fake_create_ddp_config
    peft_mod = types.ModuleType("megatron.bridge.peft.utils")
    peft_mod.create_peft_hook = lambda *args, **kwargs: (lambda model: model)
    peft_mod.load_peft_adapter_checkpoint = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "megatron.bridge.training.utils.config_utils", bridge_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", peft_mod)

    def fake_layerwise_wrap(model_chunks, tfconfig, **kwargs):
        layerwise_wrap_calls.append(
            {
                "model_chunks": model_chunks,
                "tfconfig": tfconfig,
                "kwargs": kwargs,
            }
        )
        wrapped = []
        for idx, chunk in enumerate(model_chunks):
            shell = types.SimpleNamespace(module=chunk, config=chunk.config)
            wrapped.append(shell)
        return wrapped

    monkeypatch.setattr(
        "verl.utils.megatron_utils.wrap_model_chunks_with_layerwise_aware_ddp",
        fake_layerwise_wrap,
    )
    monkeypatch.setattr(
        "verl.utils.megatron_utils._assert_muon_layer_wise_ddp_supported",
        lambda: None,
    )
    monkeypatch.setattr(
        "verl.models.mcore.config_converter.get_hf_rope_theta",
        lambda hf_config: hf_config.rope_theta,
    )
    bridge_helpers = types.ModuleType("verl.models.mcore.bridge")
    bridge_helpers.freeze_moe_router = lambda model: model
    bridge_helpers.make_value_model = lambda *args, **kwargs: (lambda model: model)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.bridge", bridge_helpers)

    return {
        "create_ddp_config_calls": create_ddp_config_calls,
        "provide_calls": provide_calls,
        "layerwise_wrap_calls": layerwise_wrap_calls,
        "provider": FakeProvider(),
    }


def test_megatron_bridge_provider_defers_ddp_for_layerwise_muon(megatron_bridge_stubs):
    wrap_config = McoreModuleWrapperConfig(
        wrap_with_ddp=True,
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=True,
    )
    model, _ = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=_DummyConfig(),
        hf_config=_minimal_hf_config(),
        bridge=object(),
        provider=megatron_bridge_stubs["provider"],
        override_ddp_config={"use_layer_wise_param_layout": True},
    )

    assert megatron_bridge_stubs["create_ddp_config_calls"] == [
        {
            "wrap_with_ddp": False,
            "use_distributed_optimizer": True,
            "use_megatron_fsdp": False,
            "overrides": {"use_layer_wise_param_layout": True},
        }
    ]
    provide_call = megatron_bridge_stubs["provide_calls"][0]
    assert provide_call["wrap_with_ddp"] is False
    assert provide_call["fp16"] is False
    assert provide_call["bf16"] is False
    assert provide_call["use_megatron_fsdp"] is False
    assert len(megatron_bridge_stubs["layerwise_wrap_calls"]) == 1
    wrap_call = megatron_bridge_stubs["layerwise_wrap_calls"][0]
    assert wrap_call["kwargs"]["use_layer_wise_distributed_optimizer"] is True
    assert wrap_call["kwargs"]["override_ddp_config"] == {"use_layer_wise_param_layout": True}
    assert model[0].module is megatron_bridge_stubs["layerwise_wrap_calls"][0]["model_chunks"][0]


def test_megatron_bridge_provider_keeps_standard_ddp_without_layerwise(megatron_bridge_stubs):
    wrap_config = McoreModuleWrapperConfig(
        wrap_with_ddp=True,
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=False,
    )
    model, _ = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=_DummyConfig(),
        hf_config=_minimal_hf_config(),
        bridge=object(),
        provider=megatron_bridge_stubs["provider"],
    )

    assert megatron_bridge_stubs["create_ddp_config_calls"][-1]["wrap_with_ddp"] is True
    assert megatron_bridge_stubs["provide_calls"][-1]["wrap_with_ddp"] is True
    assert megatron_bridge_stubs["layerwise_wrap_calls"] == []
    assert isinstance(model[0], _DummyModel)


def test_megatron_bridge_layerwise_rejects_megatron_fsdp(megatron_bridge_stubs):
    wrap_config = McoreModuleWrapperConfig(
        wrap_with_ddp=True,
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=True,
        use_megatron_fsdp=True,
    )
    with pytest.raises(ValueError, match="incompatible with Megatron FSDP"):
        make_megatron_module(
            wrap_config=wrap_config,
            tf_config=_DummyConfig(),
            hf_config=_minimal_hf_config(),
            bridge=object(),
            provider=megatron_bridge_stubs["provider"],
        )
