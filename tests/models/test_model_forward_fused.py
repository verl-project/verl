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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_NAME = "verl.models.mcore.model_forward_fused"
MODULE_PATH = Path(__file__).resolve().parents[2] / "verl/models/mcore/model_forward_fused.py"


class CausalLMOutputForPPO:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeGPTModelBase:
    pass


def _set_module(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_model_forward_fused(monkeypatch):
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith("megatron"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    _set_module(monkeypatch, "verl", _package("verl"))
    _set_module(monkeypatch, "verl.models", _package("verl.models"))
    _set_module(monkeypatch, "verl.models.mcore", _package("verl.models.mcore"))
    _set_module(monkeypatch, "verl.utils", _package("verl.utils"))
    _set_module(monkeypatch, "verl.utils.kernel", _package("verl.utils.kernel"))

    parallel_state = _module(
        "megatron.core.parallel_state",
        get_tensor_model_parallel_group=lambda: "tp-group",
    )
    mcore = _module("megatron.core", __version__="0.18.0", parallel_state=parallel_state)
    megatron = _module("megatron", core=mcore)
    _set_module(monkeypatch, "megatron", megatron)
    _set_module(monkeypatch, "megatron.core", mcore)
    _set_module(monkeypatch, "megatron.core.parallel_state", parallel_state)
    _set_module(
        monkeypatch,
        "megatron.core.config_logger",
        _module(
            "megatron.core.config_logger",
            has_config_logger_enabled=lambda _config: False,
            log_config_to_disk=lambda *_args, **_kwargs: None,
        ),
    )
    _set_module(
        monkeypatch,
        "megatron.core.inference.contexts",
        _module("megatron.core.inference.contexts", BaseInferenceContext=object),
    )
    _set_module(monkeypatch, "megatron.core.models", _package("megatron.core.models"))
    _set_module(monkeypatch, "megatron.core.models.gpt", _package("megatron.core.models.gpt"))
    _set_module(
        monkeypatch,
        "megatron.core.models.gpt.gpt_model",
        _module("megatron.core.models.gpt.gpt_model", GPTModel=FakeGPTModelBase),
    )
    _set_module(
        monkeypatch,
        "megatron.core.packed_seq_params",
        _module("megatron.core.packed_seq_params", PackedSeqParams=object),
    )
    _set_module(monkeypatch, "megatron.core.tensor_parallel", _package("megatron.core.tensor_parallel"))
    _set_module(
        monkeypatch,
        "megatron.core.tensor_parallel.mappings",
        _module(
            "megatron.core.tensor_parallel.mappings",
            gather_from_sequence_parallel_region=lambda hidden_states: hidden_states,
        ),
    )
    _set_module(
        monkeypatch,
        "megatron.core.utils",
        _module("megatron.core.utils", deprecate_inference_params=lambda context, _params: context),
    )
    _set_module(
        monkeypatch,
        "verl.models.mcore.util",
        _module(
            "verl.models.mcore.util",
            preprocess_packed_seqs=lambda *_args, **_kwargs: (_args[0], "packed"),
            preprocess_thd_engine=lambda value, **_kwargs: (value, "packed", None),
            postprocess_packed_seqs_for_dict_output=lambda *_args, **_kwargs: _args[1],
            postprocess_thd_engine=lambda value, *_args, **_kwargs: value,
        ),
    )
    _set_module(
        monkeypatch,
        "verl.utils.kernel.linear_cross_entropy",
        _module(
            "verl.utils.kernel.linear_cross_entropy",
            linear_cross_entropy=lambda *_args, **_kwargs: (torch.tensor([1.0]), torch.tensor([2.0])),
        ),
    )
    _set_module(
        monkeypatch,
        "verl.utils.megatron_utils",
        _module("verl.utils.megatron_utils", unwrap_model=lambda model: model),
    )
    _set_module(
        monkeypatch,
        "verl.utils.model",
        _module("verl.utils.model", CausalLMOutputForPPO=CausalLMOutputForPPO),
    )

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_auto_selects_hook_when_forward_supports_output_processor(monkeypatch):
    mff = load_model_forward_fused(monkeypatch)
    monkeypatch.delenv("VERL_FUSED_USE_OP_HOOK", raising=False)

    class Model(FakeGPTModelBase):
        def forward(self, input_ids=None, output_processor=None, output_processor_context=None):
            return input_ids, output_processor, output_processor_context

    model = Model()
    original_forward = model.forward.__func__

    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == "hook"
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "legacy")
    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == "hook"
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")


def test_auto_falls_back_to_legacy_and_patch_unpatch_are_idempotent(monkeypatch):
    mff = load_model_forward_fused(monkeypatch)
    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "auto")

    class Model(FakeGPTModelBase):
        config = SimpleNamespace(sequence_parallel=False)
        post_process = False

        def forward(self, input_ids=None):
            return input_ids

    model = Model()
    original_forward = model.forward.__func__

    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == "legacy"
    assert hasattr(model, "forward_backup")
    assert model.forward.__func__ is mff._fused_GPTModel_forward
    backup = model.forward_backup

    mff.patch_fused_forward(model)

    assert model.forward_backup is backup
    assert model.forward.__func__ is mff._fused_GPTModel_forward

    mff.unpatch_fused_forward(model)

    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")

    mff.unpatch_fused_forward(model)
    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == "legacy"
    assert model.forward.__func__ is mff._fused_GPTModel_forward

    mff.unpatch_fused_forward(model)
    assert model.forward.__func__ is original_forward


def test_forced_hook_fails_fast_when_forward_lacks_contract(monkeypatch):
    mff = load_model_forward_fused(monkeypatch)

    class Model(FakeGPTModelBase):
        def forward(self, input_ids=None):
            return input_ids

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "hook")
    with pytest.raises(RuntimeError, match="output_processor.*output_processor_context"):
        mff.patch_fused_forward(Model())

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "1")
    with pytest.raises(RuntimeError, match="auto.*legacy"):
        mff.patch_fused_forward(Model())

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "unsupported")
    with pytest.raises(ValueError, match="VERL_FUSED_USE_OP_HOOK"):
        mff.patch_fused_forward(Model())


def test_engine_caller_uses_build_time_mode_for_temperature(monkeypatch):
    mff = load_model_forward_fused(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])

    class Model:
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None)

        def __init__(self, mode):
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mode)
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(log_probs=torch.tensor([1.0]), entropy=torch.tensor([2.0]))

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "legacy")
    hook_model = Model("hook")
    hook_output = mff.fused_forward_model_engine()(hook_model, input_ids, labels, {}, 0.7, True, 0)

    hook_kwargs = hook_model.calls[-1]
    assert hook_output.log_probs.tolist() == [1.0]
    assert "temperature" not in hook_kwargs
    assert hook_kwargs["output_processor"] is mff.fused_output_processor
    assert hook_kwargs["output_processor_context"].temperature == pytest.approx(0.7)

    monkeypatch.setenv("VERL_FUSED_USE_OP_HOOK", "hook")
    legacy_model = Model("legacy")
    mff.fused_forward_model_engine()(legacy_model, input_ids, labels, {}, 0.5, True, 0)

    legacy_kwargs = legacy_model.calls[-1]
    assert legacy_kwargs["temperature"] == pytest.approx(0.5)
    assert "output_processor" not in legacy_kwargs
    assert "output_processor_context" not in legacy_kwargs


@pytest.mark.parametrize(
    ("sequence_parallel", "output_weight_name"),
    [(True, "shared"), (False, None)],
)
def test_output_processor_gathers_and_resolves_weight(monkeypatch, sequence_parallel, output_weight_name):
    mff = load_model_forward_fused(monkeypatch)
    hidden_states = torch.tensor([[1.0]])
    gathered_hidden_states = torch.tensor([[3.0]])
    shared_weight = torch.tensor([[5.0]])
    output_layer_weight = torch.tensor([[7.0]])
    labels = torch.tensor([0])
    log_probs = torch.tensor([11.0])
    entropy = torch.tensor([13.0])
    seen = {}

    def fake_gather(value):
        seen["gather_input"] = value
        return gathered_hidden_states

    def fake_linear_cross_entropy(hidden, weight, labels_arg, temperature, reduction, group):
        seen.update(
            hidden=hidden,
            weight=weight,
            labels=labels_arg,
            temperature=temperature,
            reduction=reduction,
            group=group,
        )
        return log_probs, entropy

    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: "tp-group")

    output_weight = shared_weight if output_weight_name == "shared" else None
    output = mff.fused_output_processor(
        hidden_states=hidden_states,
        output_layer=SimpleNamespace(weight=output_layer_weight),
        output_weight=output_weight,
        labels=labels,
        context=mff.FusedOutputProcessorContext(temperature=0.9),
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
    )

    expected_hidden = gathered_hidden_states if sequence_parallel else hidden_states
    expected_weight = shared_weight if output_weight is not None else output_layer_weight
    if sequence_parallel:
        assert seen["gather_input"] is hidden_states

    assert seen["hidden"] is expected_hidden
    assert seen["weight"] is expected_weight
    assert seen["labels"] is labels
    assert seen["temperature"] == pytest.approx(0.9)
    assert seen["reduction"] == "none"
    assert seen["group"] == "tp-group"
    assert output.log_probs is log_probs
    assert output.entropy is entropy
