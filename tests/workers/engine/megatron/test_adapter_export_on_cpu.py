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

"""Exercise the production adapter export method without loading GPU backends."""

import ast
import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_VLLM_UTILS = "verl.workers.rollout.vllm_rollout.utils"


@pytest.fixture
def adapter_export():
    """Compile the actual method, stubbing only GPU loading and PEFT config creation."""
    path = _ROOT / "verl/workers/engine/megatron/transformer_impl.py"
    tree = ast.parse(path.read_text())
    engine_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine")
    method = next(
        node for node in engine_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_per_tensor_param"
    )
    namespace = {"load_megatron_model_to_gpu": Mock(), "build_peft_config_for_vllm": Mock(return_value="peft")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_per_tensor_param"]


def _engine(**lora):
    return SimpleNamespace(
        peft_cls=object(),
        model_config=SimpleNamespace(lora=lora, hf_config=object()),
        module=object(),
        bridge=Mock(),
        _qat_enabled=False,
    )


@pytest.mark.parametrize("lora", [{}, {"experts_shared_outer_loras": False}])
def test_default_adapter_export_does_not_import_vllm(adapter_export, monkeypatch, lora):
    """Default adapter export must work when all vLLM-related imports are unavailable."""
    original_import = builtins.__import__

    def reject_vllm(name, *args, **kwargs):
        if "vllm" in name:
            pytest.fail(f"Default adapter export imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm)
    engine = _engine(**lora)
    weights, peft_config = adapter_export(engine, base_sync_done=True)
    engine.bridge.export_adapter_weights.assert_called_once_with(engine.module)
    assert weights is engine.bridge.export_adapter_weights.return_value
    assert peft_config == "peft"


@pytest.mark.parametrize("is_3d", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_adapter_layout_matches_vllm_wrapper(adapter_export, monkeypatch, is_3d, mixed, shared):
    """Mixed-format rollout must export 2D adapters even when model weights are 3D."""
    utils = ModuleType(_VLLM_UTILS)
    utils.is_3d_moe_vllm_model = Mock(return_value=is_3d)
    monkeypatch.setitem(sys.modules, _VLLM_UTILS, utils)
    engine = _engine(experts_shared_outer_loras=True)
    engine.vllm_enable_mixed_moe_lora_format = mixed
    engine.vllm_enable_moe_shared_loras = shared

    adapter_export(engine, base_sync_done=True)

    expected = "stack_3d_moe" if is_3d and not mixed else "moe_shared_loras" if shared else "expand_shared_outer"
    engine.bridge.export_adapter_weights.assert_called_once_with(engine.module, **{expected: True})


@pytest.mark.parametrize(
    "vllm_kwargs", [{}, {"enable_mixed_moe_lora_format": False}, {"enable_mixed_moe_lora_format": True}]
)
def test_rollout_mixed_format_setting_reaches_engine(vllm_kwargs):
    """Execute the production config forwarding assignments without starting workers."""
    path = _ROOT / "verl/workers/engine_workers.py"
    tree = ast.parse(path.read_text())
    worker_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker"
    )
    init_model = next(
        node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "init_model"
    )
    assignments = [
        node
        for node in ast.walk(init_model)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "vllm_engine_kwargs"
            or isinstance(target, ast.Attribute)
            and target.attr.startswith("vllm_enable_")
            for target in node.targets
        )
    ]
    engine = SimpleNamespace()
    worker = SimpleNamespace(
        config=SimpleNamespace(rollout={"engine_kwargs": {"vllm": vllm_kwargs}}),
        actor=SimpleNamespace(engine=engine),
    )
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), {"self": worker})
    assert engine.vllm_enable_mixed_moe_lora_format is vllm_kwargs.get("enable_mixed_moe_lora_format", False)
