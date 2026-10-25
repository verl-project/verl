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

import sys
import types
from types import SimpleNamespace

import pytest

from verl.workers.engine.megatron.transformer_impl import _use_qat_layer_spec


def _default_spec(config):
    return "te-spec"


def _modelopt_spec(config):
    return "modelopt-spec"


def _bridge_spec(config):
    return "bridge-spec"


@pytest.fixture(autouse=True)
def _fake_gpt_provider(monkeypatch):
    module = types.ModuleType("megatron.bridge.models.gpt_provider")
    module.default_layer_spec = _default_spec
    module.modelopt_transformer_layer_spec = _modelopt_spec
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.gpt_provider", module)


def test_qat_spec_defaults_arbitrary_attention_mask_off():
    provider = SimpleNamespace(transformer_layer_spec=_default_spec)
    overrides = {"context_parallel_size": 1}

    _use_qat_layer_spec(provider, overrides)

    assert provider.transformer_layer_spec is _modelopt_spec
    # Unset, the ModelOpt spec would pick an arbitrary mask at CP=1, which TE rejects for packed THD inputs.
    assert overrides["use_arbitrary_attention_mask"] is False


def test_qat_spec_keeps_an_explicit_override():
    provider = SimpleNamespace(transformer_layer_spec=_default_spec)
    overrides = {"use_arbitrary_attention_mask": True}

    _use_qat_layer_spec(provider, overrides)

    assert overrides["use_arbitrary_attention_mask"] is True


def test_qat_spec_keeps_a_bridge_specific_spec():
    # e.g. OLMoE: its bridge loads the full-width QK norm into modules only its own spec builds.
    provider = SimpleNamespace(transformer_layer_spec=_bridge_spec)
    overrides = {}

    _use_qat_layer_spec(provider, overrides)

    assert provider.transformer_layer_spec is _bridge_spec
    assert overrides == {}
