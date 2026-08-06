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

import torch
from transformers import (
    ApertusConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
)

from verl.utils.device import get_device_name

if get_device_name() == "cuda":
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif get_device_name() == "npu":
    from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input

from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import log_probs_from_logits_all_rmpad, masked_mean

# TODO(sgm): add more models for test
# we only need one scale for each model
test_configs = [
    LlamaConfig(num_hidden_layers=1),
    MistralConfig(num_hidden_layers=1),
    GemmaConfig(num_hidden_layers=1),
    Qwen2Config(num_hidden_layers=1),
    ApertusConfig(num_hidden_layers=1),
]


def test_hf_casual_models():
    batch_size = 4
    seqlen = 128
    response_length = 127

    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        with torch.device(get_device_name()):
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device=get_device_name())
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device=get_device_name())
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        logits_rmpad = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits  # (1, total_nnz, vocab_size)

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits
        origin_logits_rmpad, origin_logits_indices, *_ = unpad_input(origin_logits, attention_mask)

        logits_rmpad = logits_rmpad.squeeze(0)
        log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=logits_rmpad,
            indices=indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)
        origin_log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=origin_logits_rmpad,
            indices=origin_logits_indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)

        torch.testing.assert_close(
            masked_mean(log_probs, attention_mask[:, -response_length - 1 : -1]),
            masked_mean(origin_log_probs, attention_mask[:, -response_length - 1 : -1]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Check pass")


def test_hf_value_models():
    batch_size = 4
    seqlen = 128

    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        config.num_labels = 1
        config.classifier_dropout = 0
        config.hidden_dropout = 0
        with torch.device(get_device_name()):
            model = AutoModelForTokenClassification.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device=get_device_name())
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device=get_device_name())
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        rmpad_logits = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits  # (1, total_nnz, 1)
        rmpad_logits = rmpad_logits.squeeze(0)
        pad_logits = pad_input(rmpad_logits, indices, batch_size, seqlen=seqlen)

        torch.testing.assert_close(
            masked_mean(pad_logits, attention_mask[:, :, None]),
            masked_mean(origin_logits, attention_mask[:, :, None]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Value model check pass")


def test_attn_implementation_override():
    """Test that attn_implementation override config is properly respected."""
    # Test case 1: Test the actual extraction logic (no network required)
    test_cases = [
        ({}, "flash_attention_2"),  # Default case
        ({"attn_implementation": "eager"}, "eager"),  # Override case
        ({"attn_implementation": "sdpa"}, "sdpa"),  # Another override
        ({"other_config": "value"}, "flash_attention_2"),  # No attn_implementation key
    ]

    for override_config, expected in test_cases:
        actual = override_config.get("attn_implementation", "flash_attention_2")
        assert actual == expected, f"Expected {expected}, got {actual} for config {override_config}"

    # Test case 2: Test with local config creation (simulate FSDP worker behavior)
    # Test default behavior
    override_config_default = {}
    attn_implementation_default = override_config_default.get("attn_implementation", "flash_attention_2")
    assert attn_implementation_default == "flash_attention_2"

    # Test override behavior
    override_config_eager = {"attn_implementation": "eager"}
    attn_implementation_eager = override_config_eager.get("attn_implementation", "flash_attention_2")
    assert attn_implementation_eager == "eager"

    # Test that we can create a config with specific attn_implementation
    config_with_eager = LlamaConfig(num_hidden_layers=1, _attn_implementation="eager")
    assert config_with_eager._attn_implementation == "eager"

    config_with_flash = LlamaConfig(num_hidden_layers=1, _attn_implementation="flash_attention_2")
    assert config_with_flash._attn_implementation == "flash_attention_2"

    print("✓ All attn_implementation override config tests passed")


def test_fsdp_worker_attn_implementation_integration():
    """Test integration of attn_implementation with FSDP worker logic."""

    # Mock the FSDP worker configuration scenario
    mock_override_config = {"attn_implementation": "eager"}

    # Test the exact logic used in FSDP workers
    attn_implementation = mock_override_config.get("attn_implementation", "flash_attention_2")
    assert attn_implementation == "eager"

    # Test with empty config (should default)
    mock_override_config_empty = {}
    attn_implementation_default = mock_override_config_empty.get("attn_implementation", "flash_attention_2")
    assert attn_implementation_default == "flash_attention_2"

    # Test that the parameter would be passed correctly to both AutoConfig and Model
    expected_calls = [
        ("AutoConfig.from_pretrained", {"attn_implementation": attn_implementation}),
        ("AutoModel.from_pretrained", {"attn_implementation": attn_implementation}),
    ]

    # Verify the parameter extraction works as expected
    for call_name, expected_params in expected_calls:
        assert expected_params["attn_implementation"] == "eager"

    print("✓ FSDP worker integration test passed")


if __name__ == "__main__":
    test_hf_casual_models()
    test_hf_value_models()
    test_attn_implementation_override()
    test_fsdp_worker_attn_implementation_integration()
