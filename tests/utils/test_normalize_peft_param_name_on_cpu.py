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

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.utils.fsdp_utils import normalize_peft_param_name


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
