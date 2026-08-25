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
Test the MultiTurnSFTDataset implementation
"""

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from tensordict import TensorDict
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.utils import get_json_schema

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.model import extract_multi_modal_inputs

custom_model_prefix = Path("~/models").expanduser().resolve()


def test_multiturn_sft_message_group_boundaries():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "calling tools"},
        {"role": "tool", "content": "first"},
        {"role": "tool", "content": "second"},
        {"role": "assistant", "content": "answer"},
    ]

    assert MultiTurnSFTDataset._message_group_end(messages, 0) == 2
    assert MultiTurnSFTDataset._message_group_end(messages, 2) == 3
    assert MultiTurnSFTDataset._message_group_end(messages, 3) == 5
    assert MultiTurnSFTDataset._message_group_end(messages, 5) == 6
    assert MultiTurnSFTDataset._is_user_query(messages[1])
    assert not MultiTurnSFTDataset._is_user_query({"role": "user", "content": "<tool_response>result</tool_response>"})
    assert MultiTurnSFTDataset._common_prefix_length(torch.tensor([1, 2, 4]), [1, 2, 3]) == 2


@pytest.mark.parametrize("model_path", [f"{custom_model_prefix}/Qwen/Qwen3.5-0.8B"])
def test_multiturn_sft_qwen35_following_user_context(model_path: str, tmp_path: Path):
    conversations = [
        [
            {"role": "user", "content": "Run the task."},
            {"role": "assistant", "content": "<think>plan A</think>doing step one"},
            {"role": "user", "content": "Format error: retry."},
            {"role": "assistant", "content": "<think>plan B</think>doing step two"},
        ],
        [
            {"role": "system", "content": "You are a powerful assistant."},
            {"role": "assistant", "content": "premature response"},
            {"role": "user", "content": "actual query"},
            {"role": "assistant", "content": "answer"},
        ],
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>I need to output the tag</think><think>"},
            {"role": "user", "content": "Format error: retry."},
            {"role": "assistant", "content": "done"},
        ],
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think></think><think>"},
            {"role": "user", "content": "Format error: retry."},
            {"role": "assistant", "content": "done"},
        ],
    ]
    enable_thinking = [True, True, True, False]
    test_file = tmp_path / "qwen35_following_user_context.parquet"
    pd.DataFrame({"messages": conversations, "enable_thinking": enable_thinking}).to_parquet(test_file)

    tokenizer = hf_tokenizer(model_path)
    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        config={"max_length": 512, "truncation": "error", "pad_mode": "no_padding"},
    )

    multi_user_item = dataset[0]
    system_then_assistant_item = dataset[1]
    thinking_literal_item = dataset[2]
    non_thinking_literal_item = dataset[3]
    for conversation, item, item_enable_thinking in zip(
        conversations,
        (multi_user_item, system_then_assistant_item, thinking_literal_item, non_thinking_literal_item),
        enable_thinking,
        strict=True,
    ):
        expected_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=item_enable_thinking,
        )["input_ids"][0]
        assert torch.equal(item["input_ids"], expected_ids)

    full_text = tokenizer.decode(multi_user_item["input_ids"])
    assistant_text = tokenizer.decode(multi_user_item["input_ids"][multi_user_item["loss_mask"] == 1])
    assert "plan A" not in full_text
    assert "doing step one" in assistant_text
    assert "plan B" in assistant_text
    assert "doing step two" in assistant_text

    assistant_text = tokenizer.decode(
        system_then_assistant_item["input_ids"][system_then_assistant_item["loss_mask"] == 1]
    )
    assert "premature response" in assistant_text
    assert "answer" in assistant_text

    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    for item in (thinking_literal_item, non_thinking_literal_item):
        think_loss_mask = item["loss_mask"][item["input_ids"] == think_token_id]
        # The historical assistant's final literal <think> is model output and
        # supervised; the final assistant's template-generated <think> is not.
        assert think_loss_mask.tolist() == [1, 0]

    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    thinking_end_loss_mask = thinking_literal_item["loss_mask"][
        thinking_literal_item["input_ids"] == think_end_token_id
    ]
    non_thinking_end_loss_mask = non_thinking_literal_item["loss_mask"][
        non_thinking_literal_item["input_ids"] == think_end_token_id
    ]
    # Thinking mode only supplies the opening tag, so the model-generated closing
    # tag is supervised. Non-thinking mode supplies the complete empty think block.
    assert thinking_end_loss_mask.tolist() == [1]
    assert non_thinking_end_loss_mask.tolist() == [0]

    assert "I need to output" not in tokenizer.decode(thinking_literal_item["input_ids"])


@pytest.mark.parametrize("model_path", [f"{custom_model_prefix}/Qwen/Qwen3.5-0.8B"])
def test_multiturn_sft_qwen36_preserve_thinking(model_path: str, tmp_path: Path):
    tokenizer = hf_tokenizer(model_path)
    qwen35_condition = "{%- if loop.index0 > ns.last_query_index %}"
    qwen36_condition = (
        "{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}"
    )
    assert tokenizer.chat_template.count(qwen35_condition) == 1
    # Qwen3.6 uses the same relevant assistant rendering as Qwen3.5 plus
    # this preserve_thinking branch, so exercise it without another model fixture.
    tokenizer.chat_template = tokenizer.chat_template.replace(qwen35_condition, qwen36_condition)

    conversations = [
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>reason A</think>answer A<think>"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "<think>reason B</think>answer B"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think></think>answer A"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "answer B"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think></think>answer A<think>"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "answer B"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "reason A", "content": "answer A"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "reasoning_content": "reason B", "content": "answer B"},
        ],
    ]
    enable_thinking = [True, True, False, True]
    test_file = tmp_path / "qwen36_preserve_thinking.parquet"
    pd.DataFrame({"messages": conversations, "enable_thinking": enable_thinking}).to_parquet(test_file)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        config={
            "max_length": 512,
            "truncation": "error",
            "pad_mode": "no_padding",
            "apply_chat_template_kwargs": {"preserve_thinking": True},
        },
    )
    items = [dataset[index] for index in range(len(conversations))]
    for conversation, item, item_enable_thinking in zip(conversations, items, enable_thinking, strict=True):
        expected_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=item_enable_thinking,
            preserve_thinking=True,
        )["input_ids"][0]
        assert torch.equal(item["input_ids"], expected_ids)
        supervised_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1])
        assert "answer A" in supervised_text
        assert "answer B" in supervised_text

    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    for item in (items[1], items[3]):
        assert item["loss_mask"][item["input_ids"] == think_token_id].tolist() == [0, 0]
        assert item["loss_mask"][item["input_ids"] == think_end_token_id].tolist() == [1, 1]
    assert items[0]["loss_mask"][items[0]["input_ids"] == think_token_id].tolist() == [0, 1, 0]
    assert items[0]["loss_mask"][items[0]["input_ids"] == think_end_token_id].tolist() == [1, 1]
    assert "reason A" in tokenizer.decode(items[0]["input_ids"][items[0]["loss_mask"] == 1])
    assert "reason A" in tokenizer.decode(items[3]["input_ids"][items[3]["loss_mask"] == 1])

    non_thinking_item = items[2]
    assert non_thinking_item["loss_mask"][non_thinking_item["input_ids"] == think_token_id].tolist() == [0, 1, 0]
    assert non_thinking_item["loss_mask"][non_thinking_item["input_ids"] == think_end_token_id].tolist() == [0, 0]

    # Empty model-generated reasoning merges the prompt newline with the response's
    # leading newline; mask that merged token while keeping </think> supervised.
    empty_thinking_item = items[1]
    think_end_positions = torch.nonzero(empty_thinking_item["input_ids"] == think_end_token_id).flatten()
    assert empty_thinking_item["loss_mask"][think_end_positions - 1].tolist() == [0, 0]

    drop_file = tmp_path / "qwen36_drop_thinking.parquet"
    pd.DataFrame({"messages": [conversations[0]], "enable_thinking": [True]}).to_parquet(drop_file)
    drop_dataset = MultiTurnSFTDataset(
        parquet_files=str(drop_file),
        tokenizer=tokenizer,
        config={
            "max_length": 512,
            "truncation": "error",
            "pad_mode": "no_padding",
            "apply_chat_template_kwargs": {"preserve_thinking": False},
        },
    )
    drop_item = drop_dataset[0]
    drop_expected_ids = tokenizer.apply_chat_template(
        conversations[0],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
        preserve_thinking=False,
    )["input_ids"][0]
    assert torch.equal(drop_item["input_ids"], drop_expected_ids)
    assert "reason A" not in tokenizer.decode(drop_item["input_ids"])
    assert "answer A" in tokenizer.decode(drop_item["input_ids"][drop_item["loss_mask"] == 1])
    assert drop_item["loss_mask"][drop_item["input_ids"] == think_token_id].tolist() == [1, 0]
    assert drop_item["loss_mask"][drop_item["input_ids"] == think_end_token_id].tolist() == [1]


@pytest.mark.parametrize(
    "model_path, ignore_input_ids_mismatch",
    [
        (f"{custom_model_prefix}/Qwen/Qwen2.5-0.5B", False),
        (f"{custom_model_prefix}/Qwen/Qwen3-0.6B", True),
        (f"{custom_model_prefix}/Qwen/Qwen3.5-0.8B", False),
    ],
)
def test_multiturn_sft_dataset(model_path: str, ignore_input_ids_mismatch: bool):
    print(f"Starting test... model_path={model_path}, ignore_input_ids_mismatch={ignore_input_ids_mismatch}")
    # Create a temporary parquet file with test data
    test_data = {
        "messages": [
            [
                {"role": "system", "content": "You are a powerful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "tool", "content": "And what is 4+4?"},
                {"role": "tool", "content": "Please calculate it exactly."},
                {"role": "assistant", "content": "4+4 equals 8."},
            ],
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "tool", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"},
            ],
        ]
    }

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    # Initialize tokenizer and dataset
    tokenizer = hf_tokenizer(model_path)
    # processor = hf_processor(model_path)
    processor = None
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=config)

    # Test 1: Dataset Length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    # Get items for testing
    item0 = dataset[0]  # Math conversation
    item1 = dataset[1]  # Joke conversation

    # Test 2: Required Keys and Types
    required_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key}"
        assert item0[key].dtype == torch.long, f"Expected torch.long for {key}, got {item0[key].dtype}"

    # Test 3: Shape Consistency
    assert item0["loss_mask"].shape == item0["input_ids"].shape, "Loss mask shape doesn't match input_ids shape"
    assert item0["attention_mask"].shape == item0["input_ids"].shape, (
        "Attention mask shape doesn't match input_ids shape"
    )
    assert item0["position_ids"].shape == item0["input_ids"].shape, "Position IDs shape doesn't match input_ids shape"

    # Test 4: Loss Mask Pattern - Math Conversation
    loss_mask0 = item0["loss_mask"]
    input_ids0 = item0["input_ids"]

    # Find assistant response positions
    assistant_positions0 = torch.where(loss_mask0 == 1)[0]
    assert len(assistant_positions0) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text0 = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"Math conversation assistant text: {assistant_text0}")
    assert "2+2 equals 4" in assistant_text0, "First assistant response not found"
    assert "4+4 equals 8" in assistant_text0, "Second assistant response not found"

    # Test 5: Loss Mask Pattern - Joke Conversation
    loss_mask1 = item1["loss_mask"]
    input_ids1 = item1["input_ids"]

    # Find assistant response positions
    assistant_positions1 = torch.where(loss_mask1 == 1)[0]
    assert len(assistant_positions1) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text1 = tokenizer.decode(input_ids1[loss_mask1 == 1])
    print(f"Joke conversation assistant text: {assistant_text1}")
    assert "chicken cross the road" in assistant_text1, "First assistant response not found"
    assert "other side" in assistant_text1, "Second assistant response not found"

    # Test 6: Attention Mask Pattern
    attention_mask0 = item0["attention_mask"]
    sequence_length = torch.sum(attention_mask0)
    assert sequence_length > 0, "No tokens marked as attended in attention mask"
    assert torch.all(attention_mask0[:sequence_length] == 1), "Incorrect attention mask pattern"
    if sequence_length < len(attention_mask0):
        assert torch.all(attention_mask0[sequence_length:] == 0), "Padding not properly masked"

    # Test 7: Position IDs Pattern
    position_ids0 = item0["position_ids"]
    assert torch.equal(position_ids0[:sequence_length], torch.arange(sequence_length)), (
        "Position IDs not sequential for non-padded tokens"
    )
    if sequence_length < len(position_ids0):
        assert torch.all(position_ids0[sequence_length:] == 0), "Padding position IDs not zero"

    # Test 8: Verify loss mask for assistant responses
    # Get the full conversation text
    full_text = tokenizer.decode(input_ids0)
    print(f"\nFull conversation text:\n{full_text}")

    # Get the assistant responses
    assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"\nAssistant responses (from loss mask):\n{assistant_text}")

    # Verify that loss mask is set for all assistant responses
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] == "assistant":
            # The content should appear in the masked text
            assert msg["content"] in assistant_text, f"Assistant message '{msg['content']}' not found in masked text"

            # The content should NOT appear in the non-masked text
            non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
            assert msg["content"] not in non_assistant_text, (
                f"Assistant message '{msg['content']}' found in non-assistant text"
            )

    # Test 9: Verify non-assistant parts have loss_mask=0
    # Get non-assistant text
    non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
    print(f"\nNon-assistant text (from loss mask):\n{non_assistant_text}")

    # Verify that system and user messages are in the non-assistant text
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] in ["system", "user", "tool"]:
            assert msg["content"] in non_assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"
            )

            # And verify they're NOT in the assistant text
            assert msg["content"] not in assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' found in assistant text"
            )

    # Test 10: Verify padding behavior
    padding_config = {
        "max_length": 1024,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    small_dataset = MultiTurnSFTDataset(
        parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=padding_config
    )
    padded_item = small_dataset[0]

    # Get actual sequence length (before padding)
    actual_length = torch.sum(padded_item["attention_mask"])

    # Verify padding tokens
    assert torch.all(padded_item["input_ids"][actual_length:] == tokenizer.pad_token_id), (
        "Padding tokens not set correctly"
    )
    assert torch.all(padded_item["attention_mask"][actual_length:] == 0), "Attention mask not set correctly for padding"
    assert torch.all(padded_item["loss_mask"][actual_length:] == 0), "Loss mask not set correctly for padding"

    # test no-padding
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "pad_mode": "no_padding",
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=config)

    item0 = dataset[0]

    # Verify that the output contains expected keys for no-padding mode
    required_keys = ["input_ids", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in no-padding mode dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key} in no-padding mode"

    # make sure assistant_text matches with expected
    assistant_text = tokenizer.decode(item0["input_ids"][item0["loss_mask"] == 1])
    assert assistant_text == "2+2 equals 4.<|im_end|>\n4+4 equals 8.<|im_end|>\n"

    print("All tests passed!")
    print("Starting test...")


@pytest.mark.parametrize(
    "model_path, apply_chat_template_kwargs",
    [
        (f"{custom_model_prefix}/openai/gpt-oss-20b", {"model_identity": "You are a helpful assistant."}),
    ],
)
def test_multiturn_sft_dataset_with_chat_template_kwargs(model_path: str, apply_chat_template_kwargs: dict):
    """Test that custom apply_chat_template_kwargs are forwarded to system prompt
    measurement so the loss mask is not shifted when kwargs change tokenization.

    Some chat templates embed configurable fields (e.g. model_identity) in the
    system prompt. If these kwargs are not forwarded to system prompt length
    measurement, the per-turn strip length is wrong, causing role markers to be
    removed and the loss mask to shift.
    """
    test_data = {
        "messages": [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ],
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ],
        ]
    }

    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_kwargs.parquet"
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    tokenizer = hf_tokenizer(model_path)

    config = {
        "max_length": 1024,
        "truncation": "error",
        "pad_mode": "no_padding",
        "apply_chat_template_kwargs": apply_chat_template_kwargs,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=None, config=config)

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]

        assistant_text = tokenizer.decode(input_ids[loss_mask == 1])
        non_assistant_text = tokenizer.decode(input_ids[loss_mask == 0])

        for msg in test_data["messages"][idx]:
            if msg["role"] == "assistant":
                assert msg["content"] in assistant_text, (
                    f"Assistant message '{msg['content']}' not found in masked text. "
                    f"This may indicate system prompt length mismatch when using "
                    f"custom chat_template_kwargs."
                )
                assert msg["content"] not in non_assistant_text, (
                    f"Assistant message '{msg['content']}' found in non-assistant text"
                )
            elif msg["role"] in ["system", "user"]:
                assert msg["content"] in non_assistant_text, (
                    f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"
                )
                assert msg["content"] not in assistant_text, (
                    f"{msg['role'].title()} message '{msg['content']}' found in assistant text"
                )

    print("All chat_template_kwargs tests passed!")


def generate_image(description: str, size: str = "256x256"):
    """Generate a simple image based on description.

    Args:
        description: The description of the image to generate.
        size: The size of the image. Defaults to "256x256". (choices: ["256x256", "512x512"])

    Returns:
        A generated image
    """
    ...


@pytest.fixture
def vlm_data_file():
    test_data = [
        # sample 0: single turn with image input
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Describe this image.",
                },
                {
                    "role": "assistant",
                    "content": "The image is a red square.",
                },
            ],
            "images": [Image.new("RGB", (300, 300), color="red")],
            "tools": [],
        },
        # sample 1: single turn with multiple images input
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image><image>Compare these images.",
                },
                {
                    "role": "assistant",
                    "content": "The first image is a red square and the second image is a green square.",
                },
            ],
            "images": [Image.new("RGB", (100, 100), color="red"), Image.new("RGB", (100, 300), color="green")],
            "tools": [],
        },
        # sample 2: multi turn with image input and tool generated image
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Describe this image.",
                },
                {
                    "role": "assistant",
                    "content": "Let's generate a zoom-in image.",
                    "tool_calls": [
                        {
                            "function": {"arguments": {"bbox_2d": "[0, 1, 2, 4]"}, "name": "image_zoom_in_tool"},
                            "type": "function",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "<image>Generated image.",
                },
                {"role": "assistant", "content": "The zoom-in image is a red square."},
            ],
            "images": [Image.new("RGB", (300, 500), color="red"), Image.new("RGB", (100, 100), color="red")],
            "tools": [get_json_schema(generate_image)],
        },
        # sample 3: single turn without image input
        {
            "messages": [
                {"role": "user", "content": "How is the weather today?"},
                {"role": "assistant", "content": "The weather is sunny."},
            ],
            "images": [],
            "tools": [],
        },
    ]

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_vlm.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)

    def serialize_image(img):
        if isinstance(img, Image.Image):
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            return {"bytes": img_byte_arr.getvalue()}
        return img

    df["images"] = df["images"].apply(lambda x: [serialize_image(img) for img in x])

    df.to_parquet(test_file)
    return test_file


@pytest.mark.parametrize(
    "model_path",
    [
        f"{custom_model_prefix}/Qwen/Qwen3-VL-2B-Instruct",
    ],
)
def test_multiturn_sft_vlm_dataset_on_cpu(model_path, vlm_data_file):
    df = pd.read_parquet(vlm_data_file)
    tokenizer = hf_tokenizer(model_path)
    processor = hf_processor(model_path)
    config = {"max_length": 1024, "pad_mode": "no_padding", "truncation": "error", "messages_key": "messages"}
    dataset = MultiTurnSFTDataset(parquet_files=vlm_data_file, tokenizer=tokenizer, processor=processor, config=config)
    assert dataset.pad_mode == DatasetPadMode.NO_PADDING

    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        position_ids = item["position_ids"]
        pixel_values = item.get("multi_modal_inputs", {}).get("pixel_values")
        image_grid_thw = item.get("multi_modal_inputs", {}).get("image_grid_thw")

        assert input_ids.shape == loss_mask.shape, "Shapes of input_ids and loss_mask must be equal"
        assert position_ids.dim() == 2, "position_ids must be 2-dimensional"
        assert position_ids.shape[0] == 4, f"position_ids[0] should be 4: {position_ids[0]}"
        assert position_ids.shape[1] == input_ids.shape[0]

        # 1. verify input_ids without assistant text
        text = tokenizer.decode(input_ids[loss_mask == 0], skip_special_tokens=True)
        print(f"Text without assistant: {repr(text)}")
        for message in df["messages"][i]:
            if message["role"] != "assistant":
                content = message["content"].replace("<image>", "")
                assert content in text, f"user/tool text should be in the input_ids: {text}"

        # 2. verify input_ids with assistant text
        text = tokenizer.decode(input_ids[loss_mask == 1], skip_special_tokens=True)
        print(f"Text with assistant: {repr(text)}")
        for message in df["messages"][i]:
            if message["role"] == "assistant":
                assert message["content"] in text, f"Assistant text should be in the input_ids: {text}"
                assert "assistant" not in text, f"Assistant token should not be in the input_ids: {text}"

        # 3. verify image token match with image_grid_thw
        if len(df["images"][i]) > 0:
            patch_size = processor.image_processor.patch_size
            temporal_patch_size = processor.image_processor.temporal_patch_size
            merge_size = processor.image_processor.merge_size
            num_patches = image_grid_thw.prod(dim=1).sum()
            assert image_grid_thw.shape == (len(df["images"][i]), 3), (
                f"image_grid_thw: {image_grid_thw.shape} should have shape ({len(df['images'][i])}, 3)"
            )
            assert pixel_values.shape == (num_patches, 3 * temporal_patch_size * patch_size * patch_size), (
                f"pixel_values: {pixel_values.shape} should have shape ({num_patches}, {3 * patch_size * patch_size})"
            )
            assert (input_ids == processor.image_token_id).sum() == num_patches // (merge_size**2)
        else:
            assert pixel_values is None, "pixel_values should be None when no image is provided"
            assert image_grid_thw is None, "image_grid_thw should be None when no image is provided"


@pytest.mark.parametrize(
    "model_path",
    [
        f"{custom_model_prefix}/Qwen/Qwen3-VL-2B-Instruct",
    ],
)
def test_multiturn_sft_vlm_dataloader_on_cpu(model_path, vlm_data_file):
    df = pd.read_parquet(vlm_data_file)
    tokenizer = hf_tokenizer(model_path)
    processor = hf_processor(model_path)
    config = {"max_length": 1024, "pad_mode": "no_padding", "truncation": "error", "messages_key": "messages"}
    dataset = MultiTurnSFTDataset(parquet_files=vlm_data_file, tokenizer=tokenizer, processor=processor, config=config)
    assert dataset.pad_mode == DatasetPadMode.NO_PADDING

    collate_fn = SFTTensorCollator(DatasetPadMode.NO_PADDING)
    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=1, rank=0, drop_last=True)
    batch_size = 2
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for i, batch in enumerate(dataloader):
        # 1. verify input_ids, loss_mask
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]
        assert input_ids.is_nested, "input_ids should be a nested tensor"
        assert loss_mask.is_nested, "loss_mask should be a nested tensor"
        assert input_ids.shape[0] == loss_mask.shape[0] == batch_size, "Shapes of input_ids, loss_mask must be equal"

        # 2. verify position_ids: (bs, 4, seq_len)
        position_ids = batch["position_ids"]
        assert position_ids.is_nested, "position_ids should be a nested tensor"
        assert position_ids.dim() == 3, "position_ids must be 3-dimensional"
        assert position_ids.shape[0] == batch_size
        values = position_ids.values()
        assert values.shape == (4, len(input_ids.values()))

        # 3. verify multi-modal data
        td = TensorDict(**batch, batch_size=batch_size)
        multi_modal_inputs = extract_multi_modal_inputs(td["multi_modal_inputs"])
        pixel_values = multi_modal_inputs["pixel_values"]
        image_grid_thw = multi_modal_inputs["image_grid_thw"]

        num_images = sum([len(images) for images in df["images"][i * batch_size : (i + 1) * batch_size]])
        assert image_grid_thw.shape == (num_images, 3), (
            f"image_grid_thw: {image_grid_thw.shape} should have shape ({num_images}, 3)"
        )
        patch_size = processor.image_processor.patch_size
        temporal_patch_size = processor.image_processor.temporal_patch_size
        num_patches = image_grid_thw.prod(dim=1).sum()
        assert pixel_values.shape[0] == num_patches, (
            f"pixel_values: {pixel_values.shape} should have shape "
            f"({num_patches}, 3 * {temporal_patch_size} * {patch_size} * {patch_size})"
        )
