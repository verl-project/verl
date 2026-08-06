# Copyright 2025 Amazon.com, Inc. or its affiliates
# Copyright 2023-2024 SGLang Team
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


import os

import pytest

from verl.tools.schemas import ToolResponse
from verl.utils.dataset.vision_utils import process_image
from verl.utils.tokenizer import hf_processor
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    TokenizationSanityCheckModeEnum,
)


def _test_add_tool_response_messages_image_delta(processor, image_list, description_list, resize_image=False):
    assert len(image_list) == len(description_list)
    # Get the smallest dimensions across all images
    processed_images = []
    for img_url in image_list:
        img = process_image(img_url)
        processed_images.append(img)

    min_width = min(img.size[0] for img in processed_images)
    min_height = min(img.size[1] for img in processed_images)
    min_size = (min_width, min_height)

    if resize_image:
        processed_images_resized = []
        for img in processed_images:
            img = img.resize(min_size)
            processed_images_resized.append(img)
        processed_images = processed_images_resized

    # Initial message history
    system_prompt = (
        "You will be provided with an image. Describe this image and then generate a new image for the next round"
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the first image provided: "},
                {"type": "image", "image": [processed_images[0]]},
            ],
        },
    ]

    # Initial multi_modal_data with one image
    multi_modal_data = {"image": [processed_images[0]], "video": []}
    # Minimal required fields for AsyncRolloutRequest

    req = AsyncRolloutRequest(
        batch_data_id=0,
        request_id="test-req-1",
        state=AsyncRolloutRequestStateEnum.PENDING,
        messages=messages,
        multi_modal_keys=["image", "video"],
        multi_modal_data=multi_modal_data.copy(),
        tool_schemas=[],
        tools_kwargs={},
        input_ids=None,
        prompt_ids=None,
        response_ids=None,
        attention_mask=None,
        prompt_attention_mask=None,
        response_attention_mask=None,
        position_ids=None,
        prompt_position_ids=None,
        response_position_ids=None,
        loss_mask=None,
        prompt_loss_mask=None,
        response_loss_mask=None,
        reward_scores={},
        max_prompt_len=8192,
        max_response_len=8192,
        max_model_len=16384,
        metrics={},
        use_inference_chat_template=True,
        tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.STRICT,
        generation_prompt_ids=None,
        base_conv_wo_gen_prompt_end_pos=0,
        base_conv_with_gen_prompt_end_pos=0,
        processing_class=processor,
    )

    prev_generated_len = 0
    # Add First Assistant Message and first tool response message(image)
    for idx, img in enumerate(processed_images):
        if idx == 0:
            continue
        _ = req.get_generation_prompt_ids(processor)
        req.add_assistant_message(processor, content=description_list[idx - 1])
        before_tool_call_len = req.input_ids.shape[-1]
        req.add_tool_response_messages(
            processor, [ToolResponse(image=[img], text="Here is the new image you requested: ")]
        )
        after_tool_call_len = req.input_ids.shape[-1]
        if prev_generated_len == 0:
            prev_generated_len = after_tool_call_len - before_tool_call_len
        else:
            if resize_image:
                assert after_tool_call_len - before_tool_call_len == prev_generated_len
        assert req.multi_modal_data["image"] == processed_images[: idx + 1]

    _ = req.get_generation_prompt_ids(processor)
    req.add_assistant_message(processor, content=description_list[-1])

    messages = [msg.model_dump() for msg in req.messages]
    tools = [tool.model_dump() for tool in req.tool_schemas] if req.tool_schemas else None
    full_prompt_info = req._handle_apply_chat_template(
        processor,
        messages,
        multi_modal_data=req.multi_modal_data,
        tools=tools,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
    )
    full_prompt_ids = full_prompt_info["input_ids"]
    assert full_prompt_ids.eq(req.input_ids).all()

    # We must use dict(full_prompt_info) to convert BatchFeature values to a new dict
    # because np.array() only keeps the keys for BatchFeature.
    full_prompt_multi_modal_inputs = full_prompt_info.copy()
    full_prompt_multi_modal_inputs.pop("input_ids", None)
    full_prompt_multi_modal_inputs.pop("attention_mask", None)

    for key in full_prompt_multi_modal_inputs:
        assert full_prompt_multi_modal_inputs[key].eq(req.multi_modal_inputs[key]).all()


@pytest.mark.skipif(
    hf_processor(os.path.expanduser("~/models/Qwen/Qwen2.5-VL-3B-Instruct")) is None,
    reason="Processor not available for Qwen/Qwen2.5-VL-B-Instruct",
)
def test_add_tool_response_messages_image_delta():
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2.5-VL-3B-Instruct"))

    # From Qwen2.5-VL-3B-Instruct HF example
    img_1_url = {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    img_1_description = "A woman sits on the beach at sunset, smiling as she shares a high five with her large dog."
    # GitHub Logo
    img_2_url = {"image": "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"}
    img_2_description = "A GitHub Logo image"
    # Octocat
    img_3_url = {"image": "https://octodex.github.com/images/orderedlistocat.png"}
    img_3_description = "An Octocat image"

    image_list = [img_1_url, img_2_url, img_3_url]
    description_list = [img_1_description, img_2_description, img_3_description]
    _test_add_tool_response_messages_image_delta(processor, image_list, description_list, resize_image=False)


@pytest.mark.skipif(
    hf_processor(os.path.expanduser("~/models/Qwen/Qwen2.5-VL-3B-Instruct")) is None,
    reason="Processor not available for Qwen/Qwen2.5-VL-B-Instruct",
)
def test_add_tool_response_messages_image_delta_resize_image():
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2.5-VL-3B-Instruct"))

    # From Qwen2.5-VL-3B-Instruct HF example
    img_1_url = {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    img_1_description = "A woman sits on the beach at sunset, smiling as she shares a high five with her large dog."
    # GitHub Logo
    img_2_url = {"image": "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"}
    img_2_description = "A GitHub Logo image"
    # Octocat
    img_3_url = {"image": "https://octodex.github.com/images/orderedlistocat.png"}
    img_3_description = "An Octocat image"

    image_list = [img_1_url, img_2_url, img_3_url]
    description_list = [img_1_description, img_2_description, img_3_description]
    _test_add_tool_response_messages_image_delta(processor, image_list, description_list, resize_image=True)
