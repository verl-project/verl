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
import sys
import types
from pathlib import Path

import pytest
import torch

PATCH_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "vllm" / "patch.py"


def _load_patch_module():
    # Load patch.py directly to bypass ``verl.utils.vllm.__init__`` which
    # imports vllm at module load time.
    spec = importlib.util.spec_from_file_location("verl_vllm_patch_under_test", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patch_mod():
    return _load_patch_module()


@pytest.fixture
def stub_thinker_module(monkeypatch, patch_mod):
    # WeightsMapper lookalike with the same prefix-rewrite semantics as vLLM's.
    class WeightsMapper:
        def __init__(self, orig_to_new_prefix):
            self.orig_to_new_prefix = dict(orig_to_new_prefix)

        def map_name(self, key: str) -> str:
            for prefix, new_prefix in self.orig_to_new_prefix.items():
                if key.startswith(prefix):
                    key = new_prefix + key[len(prefix) :]
            return key

    class Qwen3OmniMoeThinkerForConditionalGeneration:
        hf_to_vllm_mapper = WeightsMapper(
            {
                "thinker.lm_head.": "language_model.lm_head.",
                "thinker.model.": "language_model.model.",
                "thinker.": "",
            }
        )

        def load_weights(self, weights):
            return {self.hf_to_vllm_mapper.map_name(name): tuple(weight.shape) for name, weight in weights}

    module = types.ModuleType("vllm.model_executor.models.qwen3_omni_moe_thinker")
    module.Qwen3OmniMoeThinkerForConditionalGeneration = Qwen3OmniMoeThinkerForConditionalGeneration
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_omni_moe_thinker", module)

    patch_mod._QWEN3_OMNI_THINKER_MAPPER_PATCHED = False
    return Qwen3OmniMoeThinkerForConditionalGeneration


def test_patch_rewrites_bare_keys(patch_mod, stub_thinker_module):
    patch_mod.apply_qwen3_omni_thinker_patches()

    mapper = stub_thinker_module.hf_to_vllm_mapper
    assert mapper.map_name("model.layers.0.self_attn.q_proj.weight") == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert mapper.map_name("lm_head.weight") == "language_model.lm_head.weight"
    # audio_tower has no rewrite rule, stays as-is.
    assert mapper.map_name("audio_tower.layers.0.conv.weight") == "audio_tower.layers.0.conv.weight"


def test_patch_preserves_existing_thinker_rules(patch_mod, stub_thinker_module):
    patch_mod.apply_qwen3_omni_thinker_patches()

    mapper = stub_thinker_module.hf_to_vllm_mapper
    # HF full-omni keys should still be rewritten through the original rules.
    assert mapper.map_name("thinker.model.layers.0.self_attn.q_proj.weight") == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert mapper.map_name("thinker.lm_head.weight") == "language_model.lm_head.weight"
    assert mapper.map_name("thinker.audio_tower.layers.0.conv.weight") == "audio_tower.layers.0.conv.weight"


def test_patch_is_idempotent(patch_mod, stub_thinker_module):
    patch_mod.apply_qwen3_omni_thinker_patches()
    first = dict(stub_thinker_module.hf_to_vllm_mapper.orig_to_new_prefix)
    patch_mod.apply_qwen3_omni_thinker_patches()
    second = dict(stub_thinker_module.hf_to_vllm_mapper.orig_to_new_prefix)
    assert first == second


def test_patch_expands_packed_expert_weights_before_loading(patch_mod, stub_thinker_module):
    patch_mod.apply_qwen3_omni_thinker_patches()

    model = stub_thinker_module()
    loaded = model.load_weights(
        [
            (
                "model.layers.0.mlp.experts.gate_up_proj",
                torch.empty(2, 6, 4),
            ),
            (
                "thinker.model.layers.0.mlp.experts.down_proj",
                torch.empty(2, 4, 3),
            ),
        ]
    )

    assert loaded == {
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": (3, 4),
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight": (3, 4),
        "language_model.model.layers.0.mlp.experts.1.gate_proj.weight": (3, 4),
        "language_model.model.layers.0.mlp.experts.1.up_proj.weight": (3, 4),
        "language_model.model.layers.0.mlp.experts.0.down_proj.weight": (4, 3),
        "language_model.model.layers.0.mlp.experts.1.down_proj.weight": (4, 3),
    }


def test_patch_noop_without_vllm_module(monkeypatch, patch_mod):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_omni_moe_thinker", None)
    patch_mod._QWEN3_OMNI_THINKER_MAPPER_PATCHED = False
    # Should not raise even if the module is unavailable.
    patch_mod.apply_qwen3_omni_thinker_patches()


def test_training_monkey_patch_matches_actual_qwen3_omni_model_type(monkeypatch):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    fake_module = types.ModuleType("fake_qwen3_omni_model")
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    class Qwen3OmniMoeThinkerForConditionalGeneration:
        __module__ = fake_module.__name__

        def __init__(self):
            self.set_experts_implementation_calls = []
            self.config = types.SimpleNamespace(
                model_type="qwen3_omni_moe",
                num_attention_heads=8,
                num_key_value_heads=8,
                _experts_implementation=None,
                text_config=types.SimpleNamespace(_experts_implementation=None),
            )

        def set_experts_implementation(self, implementation):
            self.set_experts_implementation_calls.append(implementation)
            self.config._experts_implementation = implementation
            self.config.text_config._experts_implementation = implementation

    model = Qwen3OmniMoeThinkerForConditionalGeneration()

    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)

    assert model.set_experts_implementation_calls == []
    assert model.config._experts_implementation == "batched_mm"


def test_training_monkey_patch_does_not_override_existing_experts_implementation(monkeypatch):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    fake_module = types.ModuleType("fake_qwen3_omni_model_existing_impl")
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    class Qwen3OmniMoeThinkerForConditionalGeneration:
        __module__ = fake_module.__name__

        def __init__(self):
            self.set_experts_implementation_calls = []
            self.config = types.SimpleNamespace(
                model_type="qwen3_omni_moe",
                num_attention_heads=8,
                num_key_value_heads=8,
                _experts_implementation="eager",
                text_config=types.SimpleNamespace(_experts_implementation="eager"),
            )

        def set_experts_implementation(self, implementation):
            self.set_experts_implementation_calls.append(implementation)
            self.config._experts_implementation = implementation
            self.config.text_config._experts_implementation = implementation

    model = Qwen3OmniMoeThinkerForConditionalGeneration()

    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)

    assert model.set_experts_implementation_calls == []
    assert model.config._experts_implementation == "eager"
    assert model.config.text_config._experts_implementation == "eager"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
