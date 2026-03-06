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

import torch
from tensordict import TensorDict

from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def test_padding_conversion_with_log_probs():
    """Test that log probability tensors remain in padded format after conversion

    This test verifies the fix for the bug where ratio values were ~451,728 instead of ~1.0.
    The key insight is that old_log_probs should STAY in padded format and be sliced
    in the loss computation to match log_prob from model output, rather than being
    converted to nested format.
    """
    batch_size = 4
    max_seq_len = 128
    max_response_len = 64

    # Create test data with varying sequence lengths
    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))

    # Create attention masks with different valid lengths per sample
    attention_mask = torch.zeros(batch_size, max_seq_len)
    valid_lens = [100, 120, 90, 128]  # Different lengths for each batch item
    for i, vlen in enumerate(valid_lens):
        attention_mask[i, :vlen] = 1

    # Create response masks aligned with the end of each sequence
    response_mask = torch.zeros(batch_size, max_response_len)
    response_lens = [50, 60, 45, 64]  # Different response lengths
    for i, rlen in enumerate(response_lens):
        response_mask[i, :rlen] = 1

    # Create position IDs
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)

    # Add log probability tensors in padded format
    old_log_probs = torch.randn(batch_size, max_seq_len)
    ref_log_prob = torch.randn(batch_size, max_seq_len)
    advantages = torch.randn(batch_size, max_response_len)
    rollout_log_probs = torch.randn(batch_size, max_seq_len)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "old_log_probs": old_log_probs,
            "ref_log_prob": ref_log_prob,
            "advantages": advantages,
            "rollout_log_probs": rollout_log_probs,
        }
    )

    # Convert to no-padding format
    data_converted = left_right_2_no_padding(data)

    # Verify input_ids and position_ids are nested tensors
    assert isinstance(data_converted["input_ids"], torch.Tensor)
    assert data_converted["input_ids"].is_nested
    assert data_converted["position_ids"].is_nested

    # Verify log probs REMAIN in padded format (NOT converted to nested)
    # They will be sliced in the loss computation to match log_prob format
    assert isinstance(data_converted["old_log_probs"], torch.Tensor)
    assert not data_converted["old_log_probs"].is_nested, "old_log_probs should remain in padded format"
    assert not data_converted["ref_log_prob"].is_nested, "ref_log_prob should remain in padded format"
    assert not data_converted["advantages"].is_nested, "advantages should remain in padded format"
    assert not data_converted["rollout_log_probs"].is_nested, "rollout_log_probs should remain in padded format"

    # Verify they maintain their original shapes
    assert data_converted["old_log_probs"].shape == (batch_size, max_seq_len)
    assert data_converted["ref_log_prob"].shape == (batch_size, max_seq_len)
    assert data_converted["advantages"].shape == (batch_size, max_response_len)
    assert data_converted["rollout_log_probs"].shape == (batch_size, max_seq_len)

    # Verify that nested tensors (input_ids, position_ids) have correct number of elements per batch item
    for i, vlen in enumerate(valid_lens):
        assert data_converted["input_ids"][i].numel() == vlen, (
            f"Batch {i}: input_ids should have {vlen} elements, got {data_converted['input_ids'][i].numel()}"
        )


def test_padding_conversion_without_log_probs():
    """Test that padding conversion works correctly when log prob tensors are not present"""
    batch_size = 4
    max_seq_len = 128
    max_response_len = 64

    # Create minimal test data
    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))
    attention_mask = torch.ones(batch_size, max_seq_len)
    response_mask = torch.ones(batch_size, max_response_len)
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
    )

    # Convert to no-padding format
    data_converted = left_right_2_no_padding(data)

    # Verify basic conversion works
    assert data_converted["input_ids"].is_nested
    assert data_converted["position_ids"].is_nested
    assert "old_log_probs" not in data_converted
    assert "ref_log_prob" not in data_converted


def test_padding_roundtrip():
    """Test that converting from padding to nested and back preserves values in the response region"""
    batch_size = 2
    max_seq_len = 64
    max_response_len = 32
    prompt_len = max_seq_len - max_response_len  # 32

    # Create simple test data with known values
    input_ids = torch.arange(1, max_seq_len + 1).unsqueeze(0).expand(batch_size, -1).clone()
    attention_mask = torch.ones(batch_size, max_seq_len)
    response_mask = torch.ones(batch_size, max_response_len)
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)

    # Create nested prompts and responses (required by no_padding_2_padding)
    prompt_list = [input_ids[i, :prompt_len] for i in range(batch_size)]
    response_list = [input_ids[i, prompt_len:] for i in range(batch_size)]
    prompts_nested = torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged)
    responses_nested = torch.nested.as_nested_tensor(response_list, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts_nested,
            "responses": responses_nested,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
    )

    # Convert to nested format
    data_nested = left_right_2_no_padding(data)

    # Verify input_ids is nested
    assert data_nested["input_ids"].is_nested

    # Convert back to padding format
    recovered = no_padding_2_padding(data_nested["input_ids"], data_nested)

    # Verify the shape is correct (response region only)
    assert recovered.shape == (batch_size, max_response_len)

    # Verify values are correct (left-shifted by 1 for log_probs alignment)
    # Response tokens are 33,34,...,64 -> left-shifted: 32,33,...,63
    expected = torch.arange(prompt_len, max_seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    torch.testing.assert_close(recovered, expected)


def test_no_padding_2_padding_varying_lengths():
    """Test no_padding_2_padding with varied prompt/response lengths."""
    batch_size = 4
    max_seq_len = 100
    max_response_len = 50

    prompt_lens = [10, 30, 5, 40]
    response_lens = [40, 20, 45, 10]

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i in range(batch_size):
        total_len = prompt_lens[i] + response_lens[i]
        input_ids[i, :total_len] = torch.arange(1, total_len + 1)

    attention_mask = torch.zeros(batch_size, max_seq_len)
    for i in range(batch_size):
        attention_mask[i, : prompt_lens[i] + response_lens[i]] = 1

    response_mask = torch.zeros(batch_size, max_response_len)
    for i in range(batch_size):
        response_mask[i, : response_lens[i]] = 1

    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).clone()

    prompt_list = [input_ids[i, : prompt_lens[i]] for i in range(batch_size)]
    response_list = [input_ids[i, prompt_lens[i] : prompt_lens[i] + response_lens[i]] for i in range(batch_size)]

    prompts_nested = torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged)
    responses_nested = torch.nested.as_nested_tensor(response_list, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "prompts": prompts_nested,
            "responses": responses_nested,
        }
    )

    data_nested = left_right_2_no_padding(data)
    input_ids_nested = data_nested["input_ids"]
    log_probs_values = input_ids_nested.values().float()
    log_probs_nested = torch.nested.nested_tensor_from_jagged(log_probs_values, offsets=input_ids_nested.offsets())

    result_slice_response = no_padding_2_padding(log_probs_nested, data_nested)

    # Verify no_padding_2_padding produces correct values (left-shifted by 1)
    for i in range(batch_size):
        resp_len = response_lens[i]
        expected_start = prompt_lens[i]
        expected_values = torch.arange(expected_start, expected_start + resp_len, dtype=torch.float)
        torch.testing.assert_close(
            result_slice_response[i, :resp_len],
            expected_values,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Batch {i} (prompt_len={prompt_lens[i]}, resp_len={resp_len}): values incorrect",
        )
    print("All varied length tests passed")


if __name__ == "__main__":
    test_padding_conversion_with_log_probs()
    test_padding_conversion_without_log_probs()
    test_padding_roundtrip()
    test_no_padding_2_padding_varying_lengths()
    print("All padding conversion tests passed!")
