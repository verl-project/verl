# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test that MultiTurnSFTDataset preserves the reasoning channel (`<think>` blocks built from
`reasoning_content`) of intermediate assistant turns when tokenizing multi-turn conversations.

Regression test for https://github.com/verl-project/verl/issues/7807: rendering each message
in isolation with `apply_chat_template([message])` drops the `<think>` block for Qwen3, because
the template only renders it when the assistant message is rendered with preceding context.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers.utils import get_json_schema

from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template

custom_model_prefix = Path("~/models").expanduser().resolve()


def find_parking(location: str):
    """Find the nearest parking lot.

    Args:
        location: The place to search parking lots around.

    Returns:
        The coordinates of the nearest parking lot.
    """
    ...


TOOLS = [get_json_schema(find_parking)]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Where is the nearest parking lot?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "User wants the nearest parking lot.",
        "tool_calls": [{"type": "function", "function": {"name": "find_parking", "arguments": '{"location": "here"}'}}],
    },
    {"role": "tool", "content": "[34.0, -117.0]"},
    {"role": "assistant", "content": "Parking is 100m ahead."},
]


@pytest.fixture
def reasoning_data_file(tmp_path):
    test_file = tmp_path / "test_qwen3_reasoning.parquet"
    pd.DataFrame({"messages": [MESSAGES], "tools": [TOOLS]}).to_parquet(test_file)
    return str(test_file)


def build_dataset(model_path: str, data_file: str, **config_overrides):
    tokenizer = hf_tokenizer(model_path)
    config = {
        "max_length": 512,
        "truncation": "error",
        "pad_mode": "no_padding",
        "messages_key": "messages",
        "enable_thinking_default": True,
        "ignore_input_ids_mismatch": False,
    }
    config.update(config_overrides)
    dataset = MultiTurnSFTDataset(parquet_files=data_file, tokenizer=tokenizer, processor=None, config=config)
    return tokenizer, dataset


@pytest.mark.parametrize("model_path", [f"{custom_model_prefix}/Qwen/Qwen3-0.6B"])
def test_multiturn_sft_qwen3_reasoning_preserved(model_path, reasoning_data_file):
    """With Qwen3, the concatenated sequence must keep the `<think>` block of every assistant
    turn and equal a one-shot `apply_chat_template` over the whole conversation, without
    resorting to `ignore_input_ids_mismatch=True`."""
    tokenizer, dataset = build_dataset(model_path, reasoning_data_file)

    # (a) default configuration: sanity_check must not raise
    item = dataset[0]

    input_ids = item["input_ids"]
    loss_mask = item["loss_mask"]
    supervised_text = tokenizer.decode(input_ids[loss_mask == 1])
    non_supervised_text = tokenizer.decode(input_ids[loss_mask == 0])
    print(f"Supervised text (loss_mask==1):\n{supervised_text}")
    print(f"Non-supervised text (loss_mask==0):\n{non_supervised_text}")

    # (b) both assistant turns are trained on, including their `<think>` blocks
    assert supervised_text.count("<think>") == 2, f"Expected 2 <think> blocks, got: {supervised_text}"
    assert "User wants the nearest parking lot." in supervised_text, (
        "reasoning_content of the tool-call turn must be kept in the supervised tokens"
    )
    assert "<tool_call>" in supervised_text and "find_parking" in supervised_text
    assert "Parking is 100m ahead." in supervised_text, "final assistant answer must be supervised"
    assert "User wants the nearest parking lot." not in non_supervised_text, (
        "reasoning_content must not leak into the non-supervised tokens"
    )
    assert "[34.0, -117.0]" in non_supervised_text, "tool response must stay non-supervised"

    # (c) the concatenated sequence equals a one-shot render of the whole conversation
    truth_ids = tokenizer.apply_chat_template(
        MESSAGES,
        tools=TOOLS,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    )["input_ids"].squeeze(0)
    assert torch.equal(input_ids, truth_ids), (
        "MultiTurnSFTDataset sequence must equal apply_chat_template on the whole conversation"
    )


@pytest.mark.parametrize("model_path", [f"{custom_model_prefix}/Qwen/Qwen2.5-0.5B"])
def test_multiturn_sft_non_thinking_model_unchanged(model_path, reasoning_data_file):
    """For templates without position-dependent rendering (Qwen2.5), the fixed prefix-based
    tokenization must produce exactly the same tokens (and loss mask) as the previous
    per-message implementation."""
    tokenizer, dataset = build_dataset(model_path, reasoning_data_file)
    item = dataset[0]

    # Replicate the previous implementation: render each message in isolation, strip the
    # system prompt tokens from every turn after the first, and pass tools on turn 0 only.
    legacy_segments = []
    for index, message in enumerate(MESSAGES):
        inputs = apply_chat_template(
            tokenizer,
            messages=[message],
            tools=TOOLS if index == 0 else None,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        segment_ids = inputs["input_ids"][0]
        if index != 0 and message["role"] != "system":
            segment_ids = segment_ids[len(dataset.system_prompt) :]
        if message["role"] == "assistant":
            loss_mask = torch.ones_like(segment_ids)
            loss_mask[: len(dataset.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(segment_ids)
        legacy_segments.append((segment_ids, loss_mask))

    legacy_input_ids = torch.cat([seg[0] for seg in legacy_segments])
    legacy_loss_mask = torch.cat([seg[1] for seg in legacy_segments])

    assert torch.equal(item["input_ids"], legacy_input_ids), (
        "input_ids must be unchanged for templates without position-dependent rendering"
    )
    assert torch.equal(item["loss_mask"], legacy_loss_mask), "loss_mask must be unchanged"
