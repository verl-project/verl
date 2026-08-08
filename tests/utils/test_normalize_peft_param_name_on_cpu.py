# Copyright 2026 Amazon.com Inc and/or its affiliates
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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLVisionConfig,
    Qwen3Config,
)

import verl.workers.engine.fsdp.transformer_impl as fsdp_transformer_impl
from verl.utils.fsdp_utils import normalize_peft_param_name
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def create_base_model():
    """Create a simple base model for testing."""
    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model


def create_peft_model():
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    model = create_base_model()
    model = get_peft_model(model, lora_config)
    return model


@pytest.fixture
def base_model():
    """Create a simple base model for testing."""
    return create_base_model()


@pytest.fixture
def peft_model():
    """Create a PEFT model with LoRA adapters."""
    return create_peft_model()


def test_normalize_peft_param_name_keys_match_base_model():
    """Test that normalized PEFT model keys match base model keys."""
    # Get state dicts
    base_model = create_base_model()
    peft_model = create_peft_model()
    base_state_dict = base_model.state_dict()
    peft_state_dict = peft_model.state_dict()

    # Normalize PEFT model keys
    normalized_peft_state_dict = normalize_peft_param_name(peft_state_dict)

    # Get key sets
    base_keys = set(base_state_dict.keys())
    normalized_peft_keys = set(normalized_peft_state_dict.keys())
    print(f"{base_keys=}")
    print(f"{normalized_peft_keys=}")

    # Verify that all base model keys are in the normalized PEFT keys
    missing_keys = base_keys - normalized_peft_keys
    assert len(missing_keys) == 0, f"Missing keys from base model: {missing_keys}"

    # Verify that all normalized PEFT keys are in the base model
    extra_keys = normalized_peft_keys - base_keys
    assert len(extra_keys) == 0, f"Extra keys not in base model: {extra_keys}"

    # Verify exact match
    assert base_keys == normalized_peft_keys, "Normalized PEFT keys should exactly match base model keys"


def test_normalize_peft_param_name_removes_lora_keys(peft_model):
    """Test that LoRA-specific parameters are removed after normalization."""
    peft_state_dict = peft_model.state_dict()

    # Before normalization, should have lora_A and lora_B keys
    lora_keys_before = [k for k in peft_state_dict.keys() if "lora_" in k]
    assert len(lora_keys_before) > 0, "PEFT model should have LoRA parameters"

    # After normalization, should not have any lora keys
    normalized_state_dict = normalize_peft_param_name(peft_state_dict)
    lora_keys_after = [k for k in normalized_state_dict.keys() if "lora_" in k]
    assert len(lora_keys_after) == 0, (
        f"Normalized state dict should not contain LoRA keys, but found: {lora_keys_after}"
    )


def test_normalize_peft_param_name_removes_base_model_prefix(peft_model):
    """Test that base_model prefix is removed from parameter names."""
    peft_state_dict = peft_model.state_dict()

    # Before normalization, should have base_model prefix
    base_model_keys = [k for k in peft_state_dict.keys() if "base_model" in k]
    assert len(base_model_keys) > 0, "PEFT model should have base_model prefix"

    # After normalization, should not have base_model prefix
    normalized_state_dict = normalize_peft_param_name(peft_state_dict)
    base_model_keys_after = [k for k in normalized_state_dict.keys() if "base_model" in k]
    assert len(base_model_keys_after) == 0, (
        f"Normalized keys should not contain base_model prefix, but found: {base_model_keys_after}"
    )


def test_normalize_peft_param_name_removes_base_layer_suffix(peft_model):
    """Test that .base_layer suffix is removed from parameter names."""
    peft_state_dict = peft_model.state_dict()

    # Before normalization, should have .base_layer suffix
    base_layer_keys = [k for k in peft_state_dict.keys() if ".base_layer" in k]
    assert len(base_layer_keys) > 0, "PEFT model should have .base_layer suffix"

    # After normalization, should not have .base_layer suffix
    normalized_state_dict = normalize_peft_param_name(peft_state_dict)
    base_layer_keys_after = [k for k in normalized_state_dict.keys() if ".base_layer" in k]
    assert len(base_layer_keys_after) == 0, (
        f"Normalized keys should not contain .base_layer suffix, but found: {base_layer_keys_after}"
    )


def test_normalize_peft_param_name_tensor_shapes_match(base_model, peft_model):
    """Test that tensor shapes match between base model and normalized PEFT model."""
    base_state_dict = base_model.state_dict()
    peft_state_dict = peft_model.state_dict()

    # Normalize PEFT model keys
    normalized_peft_state_dict = normalize_peft_param_name(peft_state_dict)

    # Check that shapes match for all common keys
    for key in base_state_dict.keys():
        assert key in normalized_peft_state_dict, f"Key {key} not found in normalized PEFT state dict"
        base_shape = base_state_dict[key].shape
        peft_shape = normalized_peft_state_dict[key].shape
        assert base_shape == peft_shape, f"Shape mismatch for {key}: base={base_shape}, peft={peft_shape}"


def test_normalize_peft_param_name_uses_active_modules_to_save():
    """Test that full-weight sync uses the trainable modules_to_save copy."""
    base_model = create_base_model()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        modules_to_save=["lm_head"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    lm_head = peft_model.base_model.model.lm_head

    with torch.no_grad():
        lm_head.original_module.weight.zero_()
        lm_head.modules_to_save["default"].weight.fill_(1.0)

    normalized = normalize_peft_param_name(peft_model.state_dict())

    assert set(normalized) == set(create_base_model().state_dict())
    assert torch.all(normalized["lm_head.weight"] == 1.0)
    assert not any("modules_to_save" in key or "original_module" in key for key in normalized)


def test_fsdp_exporter_streams_active_modules_to_save(monkeypatch):
    """Test that the merged FSDP export returns full weights without PEFT metadata."""
    peft_model = get_peft_model(
        create_base_model(),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            modules_to_save=["lm_head"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    lm_head = peft_model.base_model.model.lm_head
    with torch.no_grad():
        lm_head.original_module.weight.zero_()
        lm_head.modules_to_save["default"].weight.fill_(1.0)

    engine = object.__new__(FSDPEngine)
    engine.module = peft_model
    engine.model_config = SimpleNamespace(should_merge_lora=True)
    engine._uses_fsdp2_cpu_offload_policy = True
    engine._is_offload_param = False

    monkeypatch.setattr(fsdp_transformer_impl, "merged_lora_context", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(fsdp_transformer_impl, "get_device_id", lambda: 0)
    monkeypatch.setattr(fsdp_transformer_impl, "log_gpu_memory_usage", lambda *_args, **_kwargs: None)

    per_tensor_param, peft_config = engine.get_per_tensor_param()
    exported = dict(per_tensor_param)

    assert peft_config is None
    assert torch.all(exported["lm_head.weight"] == 1.0)
    assert not any("modules_to_save" in key or "original_module" in key for key in exported)


def test_vlm_projector_modules_to_save_semantics():
    """Test that a VLM projector is fully trainable, saved, and excluded from LoRA."""
    vision_config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    config = Qwen2_5_VLConfig(
        vision_config=vision_config,
        text_config={
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 128,
        },
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    base_model = Qwen2_5_VLForConditionalGeneration(config)
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            r=2,
            target_modules="all-linear",
            exclude_modules=".*visual.*",
            modules_to_save=["visual.merger"],
            task_type="CAUSAL_LM",
        ),
    )

    trainable_names = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    trainable_projector = [name for name in trainable_names if "visual.merger" in name]
    adapter_state = get_peft_model_state_dict(peft_model, save_embedding_layers=False)

    assert trainable_projector
    assert all("modules_to_save.default" in name for name in trainable_projector)
    assert not any("visual" in name and "lora_" in name for name in trainable_names)
    assert any("visual.merger" in key for key in adapter_state)


def test_normalize_peft_param_name_empty_dict():
    """Test that normalize_peft_param_name handles empty dict."""
    result = normalize_peft_param_name({})
    assert result == {}, "Empty dict should return empty dict"


@pytest.mark.parametrize(
    "lora_key_pattern",
    [
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "model.layers.0.adapter_layer.weight",
        "base_model.model.layers.0.lora_embedding_A",
    ],
)
def test_normalize_peft_param_name_filters_lora_patterns(lora_key_pattern):
    """Test that various LoRA key patterns are filtered out."""
    test_dict = {
        lora_key_pattern: torch.randn(10, 10),
        "model.layers.0.weight": torch.randn(10, 10),
    }

    normalized = normalize_peft_param_name(test_dict)

    # LoRA key should be filtered out
    assert lora_key_pattern not in normalized, f"LoRA key {lora_key_pattern} should be filtered out"

    # Regular key should remain
    assert len(normalized) == 1, "Should have exactly one key remaining"
    assert "model.layers.0.weight" in normalized, "Regular weight should remain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
