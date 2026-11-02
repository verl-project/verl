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
"""A quantized refit must leave the attention q/k/v/prob scales at 1.0 when the config has no KV scheme.

verl's compressed-tensors rollout config sets ``kv_cache_scheme: None``, yet vLLM still gives every
attention layer a ``CompressedTensorsKVCacheMethod``. ``load_format="dummy"`` fills its scale
placeholders with random values and the method adopts them as the live scales, so with
``kv_cache_dtype="fp8"`` the rollout quantized Q/K/V with a scale of about -7.5e-4 and produced
garbage (NaN rollout log-probs). Nothing in the refit re-processed the attention layers.

``vllm_quant_utils`` is loaded by path against stub vLLM modules, so these run without vLLM. The
stub ``set_default_quant_scales`` mirrors vLLM's ``register_buffer=False`` branch.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DUMMY_SCALE = -7.5e-4  # what vLLM's dummy loader wrote into every scale (seeded uniform(-1e-3, 1e-3))
_SCALES = ("_q_scale", "_k_scale", "_v_scale", "_prob_scale")


class _BaseKVCacheMethod:
    def __init__(self, quant_config):
        self.quant_config = quant_config


class _CompressedTensorsKVCacheMethod(_BaseKVCacheMethod):
    pass


def _set_default_quant_scales(layer, register_buffer=False):
    assert not register_buffer
    for name in _SCALES:
        getattr(layer, name).fill_(1.0)
    layer._q_scale_float = layer._k_scale_float = layer._v_scale_float = layer._prob_scale_float = 1.0
    layer._k_scale_cpu = torch.tensor(1.0)
    layer._v_scale_cpu = torch.tensor(1.0)


@pytest.fixture
def quant_utils(monkeypatch):
    attention = types.ModuleType("vllm.model_executor.layers.attention.attention")
    attention.set_default_quant_scales = _set_default_quant_scales
    kv_cache = types.ModuleType("vllm.model_executor.layers.quantization.kv_cache")
    kv_cache.BaseKVCacheMethod = _BaseKVCacheMethod
    # A bare package stub, so ``verl.utils.vllm.vllm_fp*_utils`` load from their files without
    # running ``verl/utils/vllm/__init__.py``, which imports vLLM.
    package = types.ModuleType("verl.utils.vllm")
    package.__path__ = [str(_REPO_ROOT / "verl/utils/vllm")]
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setitem(sys.modules, kv_cache.__name__, kv_cache)
    monkeypatch.setitem(sys.modules, package.__name__, package)

    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "vllm_quant_utils_under_test", _REPO_ROOT / "verl/utils/vllm/vllm_quant_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    # Drop the submodules loaded under the stub package so later tests import the real package.
    for name in set(sys.modules) - before:
        if name.startswith("verl.utils.vllm."):
            sys.modules.pop(name, None)


def _attention_layer(kv_cache_scheme=None, as_parameters=False, method_cls=_CompressedTensorsKVCacheMethod):
    """An attention layer as vLLM leaves it after a dummy load with a compressed-tensors config."""
    layer = nn.Module()
    for name in _SCALES:
        value = torch.tensor(_DUMMY_SCALE)
        if as_parameters:
            # vLLM <= 0.29 rebinds ``_k_scale`` etc. to the placeholder ``nn.Parameter``s, which require grad.
            setattr(layer, name, nn.Parameter(value))
        else:
            layer.register_buffer(name, value)
        setattr(layer, f"{name}_float", _DUMMY_SCALE)
    layer.quant_method = method_cls(SimpleNamespace(kv_cache_scheme=kv_cache_scheme))
    return layer


def _scales(layer):
    return [getattr(layer, name).item() for name in _SCALES] + [
        layer._q_scale_float,
        layer._k_scale_float,
        layer._v_scale_float,
        layer._prob_scale_float,
    ]


@pytest.mark.parametrize("as_parameters", [False, True], ids=["buffers", "grad_parameters"])
def test_resets_scales_without_kv_cache_scheme(quant_utils, as_parameters):
    model = nn.ModuleDict({"attn": _attention_layer(as_parameters=as_parameters)})

    quant_utils.reset_uncalibrated_kv_cache_scales(model)

    assert _scales(model["attn"]) == [1.0] * 8
    assert model["attn"]._k_scale_cpu.item() == 1.0 and model["attn"]._v_scale_cpu.item() == 1.0


def test_keeps_scales_of_a_kv_cache_scheme(quant_utils):
    scheme = {"type": "float", "num_bits": 8, "strategy": "tensor", "symmetric": True}
    model = nn.ModuleDict({"attn": _attention_layer(kv_cache_scheme=scheme)})

    quant_utils.reset_uncalibrated_kv_cache_scales(model)

    assert _scales(model["attn"]) == pytest.approx([_DUMMY_SCALE] * 8)


def test_ignores_layers_without_a_kv_cache_method(quant_utils):
    model = nn.ModuleDict(
        {
            "linear": nn.Linear(4, 4),
            "other": _attention_layer(method_cls=lambda config: SimpleNamespace(quant_config=config)),
        }
    )

    quant_utils.reset_uncalibrated_kv_cache_scales(model)

    assert _scales(model["other"]) == pytest.approx([_DUMMY_SCALE] * 8)


def test_refit_post_processing_resets_the_scales(quant_utils):
    model = nn.ModuleDict({"attn": _attention_layer()})

    quant_utils.process_quanted_weights_after_loading(model, reload_state={})

    assert _scales(model["attn"]) == [1.0] * 8
