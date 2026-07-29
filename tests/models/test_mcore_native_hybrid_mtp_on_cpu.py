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
"""CPU coverage for the native Megatron-Core HybridModel MTP adapter."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_forward_with_stubs(monkeypatch, *, native_hybrid=True):
    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model

    workers_config = types.ModuleType("verl.workers.config")
    workers_config.MtpConfig = object

    mtp_support = types.ModuleType("verl.models.mcore.mtp_support")
    mtp_support.is_native_hybrid_model = lambda model: native_hybrid

    util = types.ModuleType("verl.models.mcore.util")
    util_names = [
        "build_vlm_attn_mask_bshd",
        "build_vlm_attn_mask_thd",
        "postprocess_bshd",
        "postprocess_bshd_engine",
        "postprocess_packed_seqs",
        "postprocess_thd_engine",
        "preprocess_bshd",
        "preprocess_bshd_engine",
        "preprocess_packed_seqs",
        "preprocess_thd_engine",
    ]
    for name in util_names:
        setattr(util, name, lambda *args, **kwargs: None)

    mcore_package = types.ModuleType("verl.models.mcore")
    mcore_package.__path__ = []
    monkeypatch.setitem(sys.modules, "verl.models.mcore", mcore_package)
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", megatron_utils)
    monkeypatch.setitem(sys.modules, "verl.workers.config", workers_config)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.mtp_support", mtp_support)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.util", util)

    return _load_module(
        "verl.models.mcore.model_forward_native_mtp_test",
        "verl/models/mcore/model_forward.py",
    )


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_native_mtp_capability_detection(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    transformer = types.ModuleType("megatron.core.transformer")
    multi_token_prediction = types.ModuleType("megatron.core.transformer.multi_token_prediction")

    class TransformerConfig:
        mtp_detach_heads = False

    def process_mtp_loss(input_ids=None):
        return input_ids

    transformer.TransformerConfig = TransformerConfig
    multi_token_prediction.process_mtp_loss = process_mtp_loss
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.multi_token_prediction",
        multi_token_prediction,
    )
    mtp_support = _load_module(
        "mtp_support_capability_test",
        "verl/models/mcore/mtp_support.py",
    )

    assert mtp_support.has_native_mtp_support() is True


@pytest.mark.parametrize("detach_encoder", [False, True])
def test_configure_native_hybrid_mtp_maps_detach_encoder(monkeypatch, detach_encoder):
    mtp_support = _load_module(
        "mtp_support_config_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: True)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(
        enable=True,
        enable_train=True,
        detach_encoder=detach_encoder,
    )
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is True
    assert overrides["mtp_detach_heads"] is detach_encoder


def test_configure_native_hybrid_mtp_rejects_skip_compute(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_skip_compute_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: True)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(enable=True, enable_train=False, detach_encoder=True)

    with pytest.raises(ValueError, match="enable_train=False"):
        mtp_support.configure_native_hybrid_mtp(provider, mtp_config, {})


def test_configure_native_hybrid_mtp_requires_recent_mcore(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_recent_mcore_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: False)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=True)

    with pytest.raises(RuntimeError, match="mtp_detach_heads"):
        mtp_support.configure_native_hybrid_mtp(provider, mtp_config, {})


def test_gpt_provider_with_hybrid_pattern_is_not_misclassified(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_gpt_provider_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: False)
    provider = SimpleNamespace(
        is_hybrid_model=False,
        hybrid_layer_pattern="M*",
    )
    mtp_config = SimpleNamespace(enable=True, enable_train=False, detach_encoder=True)
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is False
    assert overrides == {}


def test_native_hybrid_mtp_uses_input_ids_targets_and_raw_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch)
    packed_seq_params = SimpleNamespace()
    preprocess_calls = []

    def preprocess_thd(value, *, need_roll=False, **kwargs):
        preprocess_calls.append((value, need_roll))
        values = value.values() if value.is_nested else value.reshape(-1)
        position_ids = torch.arange(values.shape[0], dtype=torch.long)
        return values.unsqueeze(0), packed_seq_params, position_ids.unsqueeze(0)

    model_forward.preprocess_thd_engine = preprocess_thd
    model_forward.postprocess_thd_engine = lambda output, *args, **kwargs: output

    input_ids = _nested(
        [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ]
    )
    response_loss_mask = _nested(
        [
            torch.tensor([True, False]),
            torch.tensor([True]),
        ]
    )
    temperature = _nested(
        [
            torch.ones(3, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        ]
    )

    class FakeHybridModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.zeros(1, kwargs["input_ids"].shape[1], 8)

    model = FakeHybridModel()
    processor_args = {}

    def logits_processor(logits, **kwargs):
        processor_args.update(kwargs)
        return {"log_probs": torch.zeros_like(kwargs["label"], dtype=torch.float32)}

    output = model_forward.gptmodel_forward_model_engine(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": temperature,
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="thd",
        mtp_enable_train=True,
    )

    assert model.forward_kwargs["labels"] is None
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[False, True, False, False, True]]),
    )
    assert any(value is response_loss_mask or value.is_nested for value, _ in preprocess_calls)
    assert processor_args["label"].shape == (1, 5)
    assert output["log_probs"].shape == (1, 5)


def test_legacy_gpt_mtp_keeps_shifted_labels_and_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch, native_hybrid=False)
    packed_seq_params = SimpleNamespace()
    preprocess_calls = []

    def preprocess_thd(value, *, need_roll=False, **kwargs):
        preprocess_calls.append(need_roll)
        values = value.values() if value.is_nested else value.reshape(-1)
        position_ids = torch.arange(values.shape[0], dtype=torch.long)
        return values.unsqueeze(0), packed_seq_params, position_ids.unsqueeze(0)

    model_forward.preprocess_thd_engine = preprocess_thd
    model_forward.postprocess_thd_engine = lambda output, *args, **kwargs: output

    input_ids = _nested(
        [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ]
    )
    response_loss_mask = _nested(
        [
            torch.tensor([True, False]),
            torch.tensor([True]),
        ]
    )

    class FakeGPTModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.zeros(1, kwargs["input_ids"].shape[1], 8)

    model = FakeGPTModel()
    model_forward.gptmodel_forward_model_engine(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=lambda logits, **kwargs: {"log_probs": torch.zeros_like(kwargs["label"]).float()},
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": _nested([torch.ones(3), torch.ones(2)]),
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="thd",
        mtp_enable_train=True,
    )

    assert model.forward_kwargs["labels"] is not None
    assert model.forward_kwargs["loss_mask"] is not None
    # Labels and the auxiliary loss mask are shifted before the legacy GPT
    # postprocess; the normal logits path independently shifts labels once.
    assert sum(preprocess_calls) >= 3
