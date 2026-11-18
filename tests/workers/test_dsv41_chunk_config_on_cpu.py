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
"""Execute verl's chunk configuration and Core integration without GPU backends."""

import ast
import builtins
import collections
import dataclasses
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, Optional
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "verl/workers/engine/megatron/transformer_impl.py"


def _compile_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def config_class(monkeypatch):
    low_precision = ModuleType("verl.utils.real_nvfp4")
    low_precision.real_nvfp4_enabled = lambda overrides: False
    monkeypatch.setitem(sys.modules, low_precision.__name__, low_precision)
    namespace = dict(
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        fields=dataclasses.fields,
        FrozenInstanceError=dataclasses.FrozenInstanceError,
        collections=collections,
        Any=Any,
        Optional=Optional,
        warnings=warnings,
    )
    _compile_nodes(ROOT / "verl/base_config.py", ["BaseConfig"], namespace)
    _compile_nodes(
        ROOT / "verl/workers/config/engine.py",
        ["EngineRouterReplayConfig", "EngineConfig", "QATEngineConfig", "McoreEngineConfig"],
        namespace,
    )
    return namespace["McoreEngineConfig"]


def test_chunk_count_defaults_to_single_pass(config_class):
    assert config_class(tensor_model_parallel_size=2).cp_seq_chunks == 1


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
def test_chunk_count_must_be_a_positive_integer(config_class, value):
    with pytest.raises(ValueError, match="cp_seq_chunks") as error:
        config_class(tensor_model_parallel_size=2, cp_seq_chunks=value)
    assert repr(value) in str(error.value)


def _engine_namespace(monkeypatch):
    tree = ast.parse(ENGINE_PATH.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {
        "MegatronEngine": {"__init__", "_build_tf_config", "_build_megatron_module", "initialize"},
        "MegatronEngineWithLMHead": {"forward_step"},
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and "cp_seq_chunk" in node.name]
    for name, wanted in methods.items():
        original = classes[name]
        selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        nodes.append(ast.ClassDef(name=name, bases=original.bases, keywords=[], body=selected, decorator_list=[]))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(
        BaseEngine=type("BaseEngine", (), {}),
        HFModelConfig=object,
        McoreEngineConfig=object,
        McoreOptimizerConfig=object,
        CheckpointConfig=object,
        Iterator=Iterator,
        TensorDict=dict,
        os=os,
        partial=partial,
        logger=Mock(),
        _check_dcp_unsupported_features=Mock(),
        set_random_seed=Mock(),
        apply_router_replay_patch=Mock(),
        __version__="0.16.0",
        tu=SimpleNamespace(get_non_tensor_data=lambda data, key, default=None: data.get(key, default)),
        unwrap_model=lambda model: model,
        get_device_id=lambda: "cpu",
        extract_multi_modal_inputs=lambda values: {key: value for item in values for key, value in item.items()},
    )
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    low_precision = ModuleType("verl.utils.megatron.low_precision")
    low_precision.bf16_layers_at_start_end = lambda overrides: (0, 0)
    real_nvfp4 = ModuleType("verl.utils.real_nvfp4")
    real_nvfp4.real_nvfp4_enabled = lambda overrides: False
    device = ModuleType("verl.utils.device")
    device.is_cuda_available = device.is_npu_available = False
    optimizer = ModuleType("verl.utils.megatron.optimizer")
    optimizer.is_muon_layer_wise_config = lambda config: False
    megatron_utils = ModuleType("verl.utils.megatron_utils")
    megatron_utils.McoreModuleWrapperConfig = SimpleNamespace
    model_utils = ModuleType("verl.utils.model")
    model_utils.print_model_size = Mock()
    outer_chunk = ModuleType("megatron.core.pipeline_parallel.outer_chunk")
    outer_chunk.validate_outer_chunk_config = Mock()
    outer_chunk.validate_outer_chunk_models = Mock()
    namespace["torch"] = SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: 1))
    for module in (low_precision, real_nvfp4, device, optimizer, megatron_utils, model_utils, outer_chunk):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace["outer_chunk"] = outer_chunk
    namespace["megatron_utils"] = megatron_utils
    return namespace


def _engine_config(**overrides):
    config = dict(
        cp_seq_chunks=2,
        dynamic_context_parallel=False,
        pipeline_model_parallel_size=1,
        use_remove_padding=True,
        use_fused_kernels=False,
        context_parallel_size=2,
        override_ddp_config={},
        override_transformer_config={},
        use_megatron_fsdp=False,
        forward_only=False,
        router_replay=SimpleNamespace(mode="disabled"),
        use_mbridge=True,
        seed=42,
        param_offload=False,
        optimizer_offload=False,
        use_distributed_optimizer=False,
        override_mcore_model_config={},
        use_dist_checkpointing=False,
        pad_to_length=False,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _transformer_config(**overrides):
    config = dict(
        experimental_attention_variant="dsv4_hybrid",
        calculate_per_token_loss=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_z_loss_coeff=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.fixture
def constructor(monkeypatch):
    namespace = _engine_namespace(monkeypatch)
    cls = namespace["MegatronEngineWithLMHead"]
    cls._init_device_mesh = Mock()

    def build_provider(self):
        self.provider = self.model_config.provider
        self.tf_config = None

    cls._build_tf_config = build_provider
    return namespace, cls


def test_chunk_config_reaches_core_during_construction(constructor):
    namespace, cls = constructor
    config = _engine_config(
        context_parallel_size=4,
        pipeline_model_parallel_size=2,
        dynamic_context_parallel=True,
        use_remove_padding=False,
        use_fused_kernels=True,
        router_replay=SimpleNamespace(mode="R2"),
        forward_only=True,
    )
    provider = _transformer_config(experimental_attention_variant="mla")
    guard = namespace["outer_chunk"].validate_outer_chunk_config
    guard.side_effect = [None, NotImplementedError("unsupported chunk config")]

    with pytest.raises(NotImplementedError, match="unsupported chunk config"):
        cls(SimpleNamespace(provider=provider), config, None, None)

    expected = dict(
        context_parallel_size=4,
        pipeline_model_parallel_size=2,
        dynamic_context_parallel=True,
        use_remove_padding=False,
        use_fused_kernels=True,
        overlap_grad_reduce=False,
        router_replay_mode="R2",
        forward_only=True,
    )
    early_check, provider_check = guard.call_args_list
    assert early_check.kwargs == dict(expected, transformer_config=None)
    assert provider_check.kwargs == dict(expected, transformer_config=provider)


@pytest.mark.parametrize(
    "declared,legacy,expected", [(1, "4", 1), (2, "broken", 2), (None, "3", 3), (None, "", 1), (None, None, 1)]
)
def test_chunk_count_resolution(monkeypatch, declared, legacy, expected):
    namespace = _engine_namespace(monkeypatch)
    # Discover the compatibility key from its only production read.
    tree = ast.parse(ENGINE_PATH.read_text())
    resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_cp_seq_chunks")
    key = next(
        n.args[0].value
        for n in ast.walk(resolver)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
    )
    if legacy is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, legacy)
    config = SimpleNamespace() if declared is None else SimpleNamespace(cp_seq_chunks=declared)
    assert namespace["_resolve_cp_seq_chunks"](config) == expected


def _model_construction(namespace, engine):
    hooks = []
    engine.provider.register_pre_wrap_hook = hooks.append
    model = SimpleNamespace(config=engine.provider)
    engine.is_value_model = False
    engine.peft_cls = None
    engine.model_config.hf_config = SimpleNamespace()
    engine.model_config.local_path = "unused"
    engine.model_config.get = lambda key, default=None: default
    engine._resolve_override_ddp_config = lambda: engine.engine_config.override_ddp_config
    engine.bridge = SimpleNamespace(load_hf_weights=Mock())

    def construct(**kwargs):
        models = [model]
        for hook in hooks:
            models = hook(models)
        return models, engine.provider

    namespace["megatron_utils"].make_megatron_module = construct
    return model


@pytest.mark.parametrize("forward_only", [False, True])
def test_constructed_models_reach_core_guard_before_weight_loading(constructor, forward_only):
    namespace, cls = constructor
    config = _engine_config(forward_only=forward_only)
    engine = cls(SimpleNamespace(provider=_transformer_config()), config, None, None)
    model = _model_construction(namespace, engine)
    guard = namespace["outer_chunk"].validate_outer_chunk_models
    guard.side_effect = NotImplementedError("unsupported chunk router")

    with pytest.raises(NotImplementedError, match="unsupported chunk router"):
        engine._build_megatron_module()

    guard.assert_called_once_with([model], forward_only=forward_only)
    engine.bridge.load_hf_weights.assert_not_called()
    guard.side_effect = None
    guard.return_value = [model]
    assert engine._build_megatron_module() == [model]


class _Tensor:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


@pytest.mark.parametrize(
    "kind,key,observed",
    [
        ("replay", "enable_routing_replay", "enable_routing_replay=True"),
        ("video", "pixel_values_videos", "'pixel_values_videos': 2"),
        ("video", "video_grid_thw", "'video_grid_thw': 2"),
        ("image", "pixel_values", "supports_cp_chunk_images=False"),
        ("image", "image_grid_hws", "supports_cp_chunk_images=False"),
        ("image", "image_grid_thw", "supports_cp_chunk_images=False"),
    ],
)
def test_batch_preconditions_reject_before_chunk_driver(constructor, kind, key, observed):
    _, cls = constructor
    engine = cls(SimpleNamespace(provider=_transformer_config()), _engine_config(), None, None)
    engine._dsv41_chunked_forward_step = Mock()
    batch = {key: True} if kind == "replay" else {"multi_modal_inputs": [{key: _Tensor(2)}]}
    with pytest.raises(NotImplementedError) as error:
        engine.forward_step(iter([batch]), SimpleNamespace(), None, SimpleNamespace(keywords={"forward_only": False}))
    assert observed in str(error.value)
    engine._dsv41_chunked_forward_step.assert_not_called()


@pytest.mark.parametrize(
    "payload,support", [({}, False), ({"pixel_values": _Tensor(0)}, False), ({"pixel_values": _Tensor(2)}, True)]
)
def test_valid_payload_reaches_chunk_driver(constructor, payload, support):
    _, cls = constructor
    engine = cls(SimpleNamespace(provider=_transformer_config()), _engine_config(), None, None)
    engine._dsv41_chunked_forward_step = Mock(return_value="chunk-driver")
    result = engine.forward_step(
        iter([{"multi_modal_inputs": [payload]}]),
        SimpleNamespace(supports_cp_chunk_images=support),
        None,
        SimpleNamespace(keywords={"forward_only": False}),
    )
    assert result == "chunk-driver"
    engine._dsv41_chunked_forward_step.assert_called_once()


def test_single_pass_enters_unchanged_forward_without_chunk_imports(constructor, monkeypatch):
    namespace, cls = constructor
    engine = cls(SimpleNamespace(), _engine_config(cp_seq_chunks=1), None, None)

    def reject_chunks(name, *args, **kwargs):
        assert "outer_chunk" not in name and "chunk_tape" not in name
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", reject_chunks)

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    # Reach the existing per-batch consistency check, downstream of the K=1 gate.
    with pytest.raises(RuntimeError, match="Per-batch use_fused_kernels must match"):
        engine.forward_step(iter([Batch(use_fused_kernels=True)]), SimpleNamespace(), None, None)
    assert not hasattr(engine, "provider")
    namespace["outer_chunk"].validate_outer_chunk_config.assert_not_called()


def test_single_pass_returns_legacy_forward_output(constructor, monkeypatch):
    namespace, cls = constructor
    engine = cls(SimpleNamespace(), _engine_config(cp_seq_chunks=1), None, None)
    engine.model_config = SimpleNamespace(hf_config=object(), tokenizer=SimpleNamespace(pad_token_id=0))
    engine.tf_config = _transformer_config()
    engine._get_context_parallel_layout = lambda model: "zigzag"
    engine._lm_head_logits_processor = Mock()
    engine.model_config.mtp = SimpleNamespace(enable=False, enable_train=False)
    ids = SimpleNamespace(clone=lambda: "labels", shape=[1], device="cpu")
    temperature = SimpleNamespace(to=lambda dtype: temperature, shape=[1])
    engine.prepare_model_inputs = lambda data: dict(
        input_ids=ids, attention_mask=None, multi_modal_inputs={}, loss_mask=object()
    )
    namespace["torch"].Tensor = type(temperature)
    namespace["torch"].float32 = "float32"
    namespace["verl_F"] = SimpleNamespace(expand_as_nested=lambda value, inputs: value)
    namespace["DatasetPadMode"] = SimpleNamespace(NO_PADDING="none")
    namespace["_resident_logits_payload"] = lambda *args, **kwargs: None
    mcore = ModuleType("verl.models.mcore")
    legacy_forward = Mock(return_value="legacy-output")
    mcore.get_mcore_engine_forward_fn = lambda config: legacy_forward
    monkeypatch.setitem(sys.modules, mcore.__name__, mcore)
    original_import = builtins.__import__

    def reject_chunks(name, *args, **kwargs):
        assert "outer_chunk" not in name and "chunk_tape" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_chunks)

    class Batch(dict):
        def to(self, device):
            return self

    batch = Batch(temperature=temperature)
    post = Mock(return_value="loss")
    output, closure = engine.forward_step(iter([batch]), SimpleNamespace(), None, post)
    assert output == "legacy-output"
    assert closure(output) == "loss"
    legacy_forward.assert_called_once()
    assert not ({"chunk_tape", "cp_chunk_count", "cp_chunk_index"} & legacy_forward.call_args.kwargs.keys())
    assert not hasattr(engine, "provider")
    namespace["outer_chunk"].validate_outer_chunk_config.assert_not_called()


@pytest.mark.parametrize(
    "megatron_fsdp,ddp_override,forward_only,overlap",
    [
        (False, {}, False, False),
        (True, {}, False, True),
        (True, {"overlap_grad_reduce": False}, False, False),
        (False, {"overlap_grad_reduce": True}, True, False),
    ],
)
def test_overlap_uses_effective_wrapper_defaults(constructor, megatron_fsdp, ddp_override, forward_only, overlap):
    namespace, cls = constructor
    config = _engine_config(
        use_megatron_fsdp=megatron_fsdp, override_ddp_config=ddp_override, forward_only=forward_only
    )
    cls(SimpleNamespace(provider=_transformer_config()), config, None, None)
    guard = namespace["outer_chunk"].validate_outer_chunk_config
    assert guard.call_args.kwargs["overlap_grad_reduce"] is overlap
