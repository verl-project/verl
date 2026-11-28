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

from types import SimpleNamespace

from verl.workers.rollout.utils import dedup_image_tokens

START, END, IMAGE_PAD, VIDEO_PAD = 151652, 151653, 151655, 151656
BOI, IMAGE, EOI = 255999, 258880, 258882


class Qwen2_5_VLProcessor(SimpleNamespace):
    """Stands in for transformers' Qwen2_5_VLProcessor, which the dedup matches by class name."""


class Glm5NextProcessor(SimpleNamespace):
    """Stands in for the GLM-5.3-Flash processor."""


class Gemma4Processor(SimpleNamespace):
    """Stands in for transformers' Gemma4Processor, which the dedup matches by class name."""


def make_gemma4_processor(cls=Gemma4Processor):
    tokenizer = SimpleNamespace(convert_tokens_to_ids={"<|image>": BOI, "<image|>": EOI}.get)
    return cls(tokenizer=tokenizer, image_token_id=IMAGE, boi_token="<|image>", eoi_token="<image|>")


def image_block(num_soft_tokens: int) -> list[int]:
    return [BOI] + [IMAGE] * num_soft_tokens + [EOI]


def test_qwen_image_and_video_pads_collapse_to_one_pad():
    processor = Qwen2_5_VLProcessor(image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD)
    prompt_ids = [1, START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, END, START, VIDEO_PAD, VIDEO_PAD, END, 2]

    assert dedup_image_tokens(prompt_ids, processor) == [1, START, IMAGE_PAD, END, START, VIDEO_PAD, END, 2]


def test_glm_collapses_repeated_pads_and_keeps_distinct_adjacent_pads():
    processor = Glm5NextProcessor(image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD)
    prompt_ids = [1, START, IMAGE_PAD, IMAGE_PAD, VIDEO_PAD, VIDEO_PAD, END, 2]

    assert dedup_image_tokens(prompt_ids, processor) == [1, START, IMAGE_PAD, VIDEO_PAD, END, 2]


def test_each_gemma4_image_block_collapses_to_one_image_token():
    prompt_ids = [1, 2] + image_block(256) + image_block(256) + [3]

    assert dedup_image_tokens(prompt_ids, make_gemma4_processor()) == [1, 2, IMAGE, IMAGE, 3]


def test_other_processors_are_untouched():
    prompt_ids = [1] + image_block(3) + [2]

    assert dedup_image_tokens(prompt_ids, make_gemma4_processor(SimpleNamespace)) == prompt_ids
    assert dedup_image_tokens(prompt_ids, None) == prompt_ids
