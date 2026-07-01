# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from types import SimpleNamespace
from pathlib import Path

import pytest


def _load_patch_module():
    module_path = Path(__file__).parents[2] / "verl" / "models" / "transformers" / "qwen3_5_gdn_patch.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_gdn_patch", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qwen3_5_gdn_patch = _load_patch_module()


def _fake_triton_gdn():
    return "triton-gdn"


def _build_gdn_class():
    class DummyGatedDeltaNet:
        def __init__(self, config):
            self.use_triton_gdn_seen_by_original_init = getattr(config, "use_triton_gdn", None)
            self.chunk_gated_delta_rule = "eager-gdn"

    return DummyGatedDeltaNet


def test_patch_injects_triton_gdn_after_protected_original_init(monkeypatch):
    monkeypatch.setattr(qwen3_5_gdn_patch, "_load_triton_gdn", _fake_triton_gdn)
    gdn_cls = _build_gdn_class()

    qwen3_5_gdn_patch.patch_qwen3_5_gdn_class(gdn_cls)

    config = SimpleNamespace(use_triton_gdn=True)
    module = gdn_cls(config)

    assert module.use_triton_gdn_seen_by_original_init is False
    assert module.chunk_gated_delta_rule == "triton-gdn"
    assert module.use_triton_gdn is True
    assert config.use_triton_gdn is True


def test_patch_treats_auto_compute_mode_as_triton(monkeypatch):
    monkeypatch.setattr(qwen3_5_gdn_patch, "_load_triton_gdn", _fake_triton_gdn)
    gdn_cls = _build_gdn_class()

    qwen3_5_gdn_patch.patch_qwen3_5_gdn_class(gdn_cls)

    module = gdn_cls(SimpleNamespace(gdn_compute_mode="auto", use_triton_gdn=False))

    assert module.chunk_gated_delta_rule == "triton-gdn"


def test_patch_rejects_unwired_ascendc_backend():
    gdn_cls = _build_gdn_class()
    qwen3_5_gdn_patch.patch_qwen3_5_gdn_class(gdn_cls)

    with pytest.raises(ValueError, match="ascendc"):
        gdn_cls(SimpleNamespace(gdn_compute_mode="ascendc", use_triton_gdn=False))
