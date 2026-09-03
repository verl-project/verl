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


def _load_megatron_utils_with_stubs(monkeypatch):
    def stub_module(name, **attributes):
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class DistributedDataParallel:
        pass

    class Float16Module:
        pass

    class ModelType:
        encoder_or_decoder = object()

    verl = stub_module("verl")
    verl.__path__ = []
    verl_utils = stub_module("verl.utils")
    verl_utils.__path__ = []
    verl_megatron = stub_module("verl.utils.megatron")
    verl_megatron.__path__ = []
    verl_workers = stub_module("verl.workers")
    verl_workers.__path__ = []
    verl_models = stub_module("verl.models")
    verl_models.__path__ = []
    verl_mcore = stub_module("verl.models.mcore")
    verl_mcore.__path__ = []
    verl.utils = verl_utils
    verl.workers = verl_workers
    verl.models = verl_models
    verl_utils.megatron = verl_megatron
    verl_models.mcore = verl_mcore

    core = stub_module(
        "megatron.core",
        ModelParallelConfig=object,
        mpu=SimpleNamespace(),
        parallel_state=SimpleNamespace(),
        tensor_parallel=SimpleNamespace(),
    )
    megatron = stub_module("megatron", core=core)
    megatron.__path__ = []
    core.__path__ = []
    stub_module(
        "megatron.core.distributed",
        DistributedDataParallel=DistributedDataParallel,
        DistributedDataParallelConfig=object,
    )
    stub_module("megatron.core.enums", ModelType=ModelType)
    stub_module("megatron.core.optimizer", ChainedOptimizer=object)
    stub_module("megatron.core.parallel_state", get_global_memory_buffer=lambda: None)
    transformer = stub_module(
        "megatron.core.transformer",
        MLATransformerConfig=object,
        TransformerConfig=object,
    )
    transformer.__path__ = []
    stub_module("megatron.core.transformer.module", Float16Module=Float16Module)
    stub_module("megatron.core.transformer.multi_token_prediction", MTPLossLoggingHelper=object)
    stub_module("megatron.core.utils", get_attr_wrapped_model=lambda model, *args, **kwargs: model)

    stub_module("tensordict", TensorDict=object)
    stub_module("transformers", PretrainedConfig=object)
    verl_megatron.tensor_parallel = stub_module("verl.utils.megatron.tensor_parallel")
    verl_utils.tensordict_utils = stub_module("verl.utils.tensordict_utils")
    stub_module(
        "verl.utils.device",
        get_device_id=lambda: 0,
        get_device_name=lambda: "cpu",
        get_torch_device=lambda: torch,
    )
    stub_module("verl.utils.fs", local_mkdir_safe=lambda path: path)
    stub_module("verl.utils.model", normalize_model_name=lambda name: name)
    stub_module("verl.utils.torch_dtypes", PrecisionType=object)
    verl_workers.config = stub_module("verl.workers.config", HFModelConfig=object, McoreEngineConfig=object)

    return _load_module(
        "verl.utils.megatron_utils_native_mtp_test",
        "verl/utils/megatron_utils.py",
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


@pytest.mark.parametrize(
    "provider",
    [
        type("HybridModelProvider", (), {})(),
        type("MarkedHybridProvider", (), {"is_hybrid_model": True})(),
    ],
    ids=["provider-class-name", "explicit-marker"],
)
def test_configure_native_hybrid_mtp_recognizes_supported_provider_forms(monkeypatch, provider):
    mtp_support = _load_module(
        "mtp_support_provider_detection_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: True)
    mtp_config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=False)
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is True
    assert overrides == {"mtp_detach_heads": False}


def test_configure_native_hybrid_mtp_recognizes_subclass_inheriting_marker(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_provider_subclass_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: True)

    class MarkedHybridProvider:
        is_hybrid_model = True

    class SpecializedHybridProvider(MarkedHybridProvider):
        pass

    overrides = {}
    configured = mtp_support.configure_native_hybrid_mtp(
        SpecializedHybridProvider(),
        SimpleNamespace(enable=True, enable_train=True, detach_encoder=True),
        overrides,
    )

    assert configured is True
    assert overrides == {"mtp_detach_heads": True}


def test_configure_native_hybrid_mtp_recognizes_provider_subclass_without_marker(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_provider_unmarked_subclass_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "has_native_mtp_support", lambda: True)

    class HybridModelProvider:
        pass

    class SpecializedHybridProvider(HybridModelProvider):
        pass

    overrides = {}
    configured = mtp_support.configure_native_hybrid_mtp(
        SpecializedHybridProvider(),
        SimpleNamespace(enable=True, enable_train=True, detach_encoder=True),
        overrides,
    )

    assert configured is True
    assert overrides == {"mtp_detach_heads": True}


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


def test_disabled_mtp_clears_generic_and_hybrid_provider_overrides(monkeypatch):
    megatron_utils = _load_megatron_utils_with_stubs(monkeypatch)
    hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        mtp=SimpleNamespace(enable=False, mtp_loss_scaling_factor=0.3),
    )
    engine_config = SimpleNamespace(
        override_transformer_config={
            "mtp_num_layers": 2,
            "mtp_hybrid_override_pattern": "*E",
            "mtp_use_repeated_layer": True,
            "keep_mtp_spec_in_bf16": True,
            "mtp_loss_scaling_factor": 0.3,
        }
    )

    megatron_utils.check_mtp_config(model_config, engine_config)

    assert hf_config.num_nextn_predict_layers == 0
    assert engine_config.override_transformer_config == {"mtp_num_layers": None}

    mtp_support = _load_module(
        "mtp_support_disabled_test",
        "verl/models/mcore/mtp_support.py",
    )

    class HybridModelProvider:
        pass

    overrides = dict(engine_config.override_transformer_config)
    configured = mtp_support.configure_native_hybrid_mtp(
        HybridModelProvider(),
        SimpleNamespace(enable=False),
        overrides,
    )

    assert configured is False
    assert overrides == {
        "mtp_num_layers": None,
        "mtp_hybrid_override_pattern": None,
        "mtp_use_repeated_layer": False,
        "keep_mtp_spec_in_bf16": False,
    }

    strict_gpt_overrides = {"mtp_num_layers": None}
    configured = mtp_support.configure_native_hybrid_mtp(
        object(),
        SimpleNamespace(enable=False),
        strict_gpt_overrides,
    )

    assert configured is False
    assert strict_gpt_overrides == {"mtp_num_layers": None}


def test_convert_to_nested_tensor_rejects_short_labels(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch)
    labels = torch.tensor([[10, 11, 12]])

    with pytest.raises(ValueError, match="label length 3 is shorter than input length 4"):
        model_forward._convert_to_nested_tensor(labels, [4])


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


def test_native_hybrid_mtp_bshd_uses_no_labels_and_unshifted_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch)
    preprocess_calls = []

    def preprocess_bshd(value, *, need_roll=False, **kwargs):
        dense = value.to_padded_tensor(0)
        if need_roll:
            dense = torch.roll(dense, shifts=-1, dims=1)
        attention_mask = torch.arange(dense.shape[1]).unsqueeze(0) < value.offsets().diff().unsqueeze(1)
        position_ids = torch.arange(dense.shape[1], dtype=torch.long).unsqueeze(0).expand_as(dense)
        preprocess_calls.append((value, need_roll, dense.clone()))
        return dense, attention_mask, position_ids

    model_forward.preprocess_bshd_engine = preprocess_bshd
    model_forward.postprocess_bshd_engine = lambda output, *args, **kwargs: output

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

    class FakeHybridModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            batch, sequence = kwargs["input_ids"].shape
            return torch.zeros(batch, sequence, 8)

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
            "temperature": _nested([torch.ones(3), torch.ones(2)]),
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="bshd",
        mtp_enable_train=True,
    )

    assert model.forward_kwargs["labels"] is None
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[False, True, False], [False, True, False]]),
    )
    # Input IDs and native MTP loss mask remain unshifted; only the normal
    # next-token label used by the logits processor follows the rolled path.
    assert [need_roll for _, need_roll, _ in preprocess_calls] == [False, False, True, False]
    torch.testing.assert_close(
        processor_args["label"],
        torch.tensor([[11, 12, 10], [21, 0, 20]]),
    )
    assert "loss_mask" not in processor_args
    assert output["log_probs"].shape == (2, 3)


def test_legacy_gpt_mtp_keeps_shifted_labels_and_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch, native_hybrid=False)
    packed_seq_params = SimpleNamespace()
    preprocess_calls = []

    def preprocess_thd(value, *, need_roll=False, **kwargs):
        preprocess_calls.append((need_roll, kwargs.get("pad_to_length_bucket")))
        values = value.values() if value.is_nested else value.reshape(-1)
        if need_roll:
            values = torch.roll(values, shifts=-1, dims=0)
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
    processor_args = {}

    def logits_processor(logits, **kwargs):
        processor_args.update(kwargs)
        return {"log_probs": torch.zeros_like(kwargs["label"]).float()}

    model_forward.gptmodel_forward_model_engine(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": _nested([torch.ones(3), torch.ones(2)]),
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="thd",
        mtp_enable_train=True,
        pad_to_length_bucket=8,
    )

    torch.testing.assert_close(
        model.forward_kwargs["labels"],
        torch.tensor([[11, 12, 20, 21, 10]]),
    )
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        processor_args["label"],
        torch.tensor([[11, 12, 20, 21, 10]]),
    )
    assert "loss_mask" not in processor_args
    # MTP labels, the auxiliary loss mask, and the normal logits label each
    # take the legacy shifted path. Input IDs and temperature remain unshifted.
    assert preprocess_calls == [(False, 8), (True, 8), (True, 8), (True, 8), (False, 8)]


def test_patch_engine_mtp_skips_all_legacy_patches_for_native_hybrid_model(monkeypatch):
    megatron_utils = _load_megatron_utils_with_stubs(monkeypatch)

    mtp_support = types.ModuleType("verl.models.mcore.mtp_support")
    mtp_support.is_native_hybrid_model = lambda model: True
    monkeypatch.setitem(sys.modules, "verl.models.mcore.mtp_support", mtp_support)

    def fail_if_called(*args, **kwargs):
        pytest.fail("legacy MTP patch was applied to a native HybridModel")

    mtp_patch = types.ModuleType("verl.models.mcore.mtp_patch")
    mtp_patch.patch_postprocess = fail_if_called
    mtp_patch.patch_mtp_layer_checkpointed_forward = fail_if_called
    mtp_patch.patch_mtp_layer_get_embeddings = fail_if_called
    monkeypatch.setitem(sys.modules, "verl.models.mcore.mtp_patch", mtp_patch)

    megatron_utils.patch_engine_mtp(
        object(),
        SimpleNamespace(mtp=SimpleNamespace(enable_train=True, detach_encoder=True)),
    )
