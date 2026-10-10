# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Individual Contributor: Lirui Luo
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

"""Packed-boundary capability detection, including the optional TRL value head.

Exercise the real engine builder/input preparation without distributed startup.
The recording model tests metadata delivery, not a Qwen kernel's numerics.
"""

import builtins
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import import_utils
from verl.workers.engine.fsdp import transformer_impl


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.received = None

    def forward(self, input_ids=None, attention_mask=None, cu_seqlens=None, cu_seqlens_cpu=None, **kwargs):
        self.received = (cu_seqlens, cu_seqlens_cpu)
        hidden = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(hidden,), logits=hidden, loss=None)


class UnsupportedModel(torch.nn.Module):
    def forward(self, input_ids=None, **kwargs):
        return input_ids


class TransparentValueHead(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model

    def forward(self, input_ids=None, **kwargs):
        return self.pretrained_model(input_ids=input_ids, **kwargs)


@pytest.fixture(params=["experimental", "legacy"])
def trl_class(request, monkeypatch):
    """Test both optional import locations even in CPU CI images without TRL."""
    module_name = (
        "trl.experimental.ppo.modeling_value_head"
        if request.param == "experimental"
        else "trl.models.modeling_value_head"
    )
    value_head_cls = type("AutoModelForCausalLMWithValueHead", (TransparentValueHead,), {"__module__": module_name})
    trl = ModuleType("trl")
    trl.__path__ = []
    monkeypatch.setitem(sys.modules, "trl", trl)
    if request.param == "experimental":
        experimental = ModuleType("trl.experimental")
        experimental.__path__ = []
        ppo = ModuleType("trl.experimental.ppo")
        ppo.AutoModelForCausalLMWithValueHead = value_head_cls
        monkeypatch.setitem(sys.modules, "trl.experimental", experimental)
        monkeypatch.setitem(sys.modules, "trl.experimental.ppo", ppo)
    else:
        trl.AutoModelForCausalLMWithValueHead = value_head_cls
        monkeypatch.setitem(sys.modules, "trl.experimental", None)
        monkeypatch.setitem(sys.modules, "trl.experimental.ppo", None)
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: True)
    return value_head_cls


def make_engine(monkeypatch, model, model_type="value_model", pad_to_length=False):
    engine = transformer_impl.FSDPEngineWithValueHead.__new__(transformer_impl.FSDPEngineWithValueHead)
    engine.model_config = SimpleNamespace(model_type=model_type)
    engine.engine_config = SimpleNamespace(forward_only=True)
    engine._is_lora = False
    engine._qat_enabled = False
    engine.rank = 1
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = 4
    engine.use_ulysses_sp = False
    engine._build_module = lambda: model
    engine._build_fsdp_module = lambda module: module
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    engine._build_model_optimizer()
    return engine


def make_micro_batch():
    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])], layout=torch.jagged)
    position_ids = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0, 1, 2, 3, 0, 1, 2]), offsets=input_ids.offsets()
    )
    micro_batch = TensorDict({"input_ids": input_ids, "position_ids": position_ids}, batch_size=[2])
    micro_batch.set_non_tensor("temperature", 1.0)
    return micro_batch


@pytest.mark.parametrize("pad_to_length", [False, True])
def test_trl_critic_receives_packed_boundaries(monkeypatch, trl_class, pad_to_length):
    model = trl_class(RecordingModel())
    engine = make_engine(monkeypatch, model, pad_to_length=pad_to_length)
    assert engine.pass_packed_cu_seqlens

    model_inputs, output_args = engine.prepare_model_inputs(make_micro_batch())
    expected = [0, 4, 7, 8] if pad_to_length else [0, 4, 7]
    assert model_inputs["cu_seqlens"].tolist() == expected
    assert model_inputs["cu_seqlens_cpu"].tolist() == expected
    assert output_args["pad_size"] == int(pad_to_length)
    model(**model_inputs)
    assert model.pretrained_model.received[0] is model_inputs["cu_seqlens"]
    assert model.pretrained_model.received[1] is model_inputs["cu_seqlens_cpu"]


def test_unpacked_inputs_do_not_receive_boundaries(monkeypatch, trl_class):
    model = trl_class(RecordingModel())
    engine = make_engine(monkeypatch, model)
    micro_batch = make_micro_batch()
    micro_batch.set_non_tensor("use_remove_padding", False)
    model_inputs, _ = engine.prepare_model_inputs(micro_batch)
    assert "cu_seqlens" not in model_inputs
    assert "cu_seqlens_cpu" not in model_inputs
    assert model_inputs["input_ids"].shape == (2, 4)


@pytest.mark.parametrize("model_type", ["language_model", "value_model"])
def test_explicit_capability_does_not_need_trl(monkeypatch, model_type):
    def unexpected_import():
        pytest.fail("An explicit capability must not require TRL")

    monkeypatch.setattr(import_utils, "is_trl_available", unexpected_import)
    assert make_engine(monkeypatch, RecordingModel(), model_type).pass_packed_cu_seqlens


def test_trl_requires_underlying_explicit_capability(monkeypatch, trl_class):
    assert not make_engine(monkeypatch, trl_class(UnsupportedModel())).pass_packed_cu_seqlens


def test_only_value_models_are_unwrapped(monkeypatch, trl_class):
    assert not make_engine(monkeypatch, trl_class(RecordingModel()), "language_model").pass_packed_cu_seqlens


@pytest.mark.parametrize("opaque_kind", ["unknown", "subclass", "namespace_lookalike", "instance_override"])
def test_opaque_wrappers_are_not_unwrapped(monkeypatch, trl_class, opaque_kind):
    class UnknownWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained_model = RecordingModel()

        def forward(self, **kwargs):
            return None

    class CustomValueHead(trl_class):
        pass

    if opaque_kind == "unknown":
        model = UnknownWrapper()
    elif opaque_kind == "subclass":
        model = CustomValueHead(RecordingModel())
    elif opaque_kind == "namespace_lookalike":
        lookalike = type(trl_class.__name__, (trl_class,), {"__module__": trl_class.__module__})
        model = lookalike(RecordingModel())
    else:
        model = trl_class(RecordingModel())
        model.forward = MethodType(lambda self, **kwargs: None, model)
    assert not make_engine(monkeypatch, model).pass_packed_cu_seqlens


def test_value_model_without_trl_remains_supported(monkeypatch):
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: False)
    assert not make_engine(monkeypatch, UnsupportedModel()).pass_packed_cu_seqlens


def test_known_wrapper_without_trl_is_not_unwrapped(monkeypatch, trl_class):
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: False)
    assert not make_engine(monkeypatch, trl_class(RecordingModel())).pass_packed_cu_seqlens


@pytest.mark.parametrize("error", [ImportError, RuntimeError])
@pytest.mark.parametrize("model_kind", ["native", "unknown_wrapper", "name_lookalike"])
def test_unrelated_value_models_do_not_import_trl(monkeypatch, error, model_kind):
    original_import = builtins.__import__

    def unavailable_trl(name, *args, **kwargs):
        if name == "trl" or name.startswith("trl."):
            raise error("Installed TRL is incompatible")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_trl)
    monkeypatch.setattr(import_utils, "is_trl_available", lambda: True)
    if model_kind == "native":
        model = UnsupportedModel()
    elif model_kind == "unknown_wrapper":
        model = TransparentValueHead(RecordingModel())
    else:
        lookalike = type("AutoModelForCausalLMWithValueHead", (TransparentValueHead,), {})
        model = lookalike(RecordingModel())
    assert not make_engine(monkeypatch, model).pass_packed_cu_seqlens


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_uninspectable_forward_keeps_previous_behavior(monkeypatch, error):
    def uninspectable(_):
        raise error("uninspectable forward")

    monkeypatch.setattr(transformer_impl, "signature", uninspectable)
    assert not make_engine(monkeypatch, UnsupportedModel()).pass_packed_cu_seqlens


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("pad_to_length", [False, True])
def test_real_trl_forward_preserves_boundaries(monkeypatch, grad_enabled, pad_to_length):
    pytest.importorskip("trl")
    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead
    except ImportError:
        from trl import AutoModelForCausalLMWithValueHead

    assert AutoModelForCausalLMWithValueHead.forward.__module__.startswith("trl.")
    # Use the installed, unmodified TRL forward with a tiny recording model.
    # No checkpoint download or accelerator is needed for this interface test.
    model = AutoModelForCausalLMWithValueHead.__new__(AutoModelForCausalLMWithValueHead)
    torch.nn.Module.__init__(model)
    model.pretrained_model = RecordingModel()
    model.v_head = torch.nn.Module()
    model.v_head.summary = torch.nn.Linear(4, 1, bias=False)
    model.v_head.forward = model.v_head.summary.forward
    model.is_peft_model = False
    engine = make_engine(monkeypatch, model, pad_to_length=pad_to_length)
    assert engine.pass_packed_cu_seqlens

    micro_batch = make_micro_batch()
    model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
    expected = [0, 4, 7, 8] if pad_to_length else [0, 4, 7]
    assert model_inputs["cu_seqlens"].tolist() == expected
    with torch.set_grad_enabled(grad_enabled):
        output = model(**model_inputs, use_cache=False)
        values = engine.prepare_model_outputs(output, output_args, micro_batch, None)["values"]
        assert values.offsets().tolist() == [0, 4, 7]
        assert values.values().shape == (7,)
        assert values.requires_grad is grad_enabled
        if grad_enabled:
            values.values().sum().backward()
            assert model.v_head.summary.weight.grad.abs().sum() > 0
            assert model.pretrained_model.embedding.weight.grad.abs().sum() > 0
    assert model.pretrained_model.received[0] is model_inputs["cu_seqlens"]
    assert model.pretrained_model.received[1] is model_inputs["cu_seqlens_cpu"]
