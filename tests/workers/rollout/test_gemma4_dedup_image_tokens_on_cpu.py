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

from verl.workers.rollout.utils import gemma4_dedup_image_tokens

BOI, IMAGE, EOI = 255999, 258880, 258882


class Gemma4Processor(SimpleNamespace):
    """Stands in for transformers' Gemma4Processor, which the dedup matches by class name."""


def make_processor(cls=Gemma4Processor):
    tokenizer = SimpleNamespace(convert_tokens_to_ids={"<|image>": BOI, "<image|>": EOI}.get)
    return cls(tokenizer=tokenizer, image_token_id=IMAGE, boi_token="<|image>", eoi_token="<image|>")


def image_block(num_soft_tokens: int) -> list[int]:
    return [BOI] + [IMAGE] * num_soft_tokens + [EOI]


def test_each_image_block_collapses_to_one_image_token():
    prompt_ids = [1, 2] + image_block(256) + image_block(256) + [3]

    assert gemma4_dedup_image_tokens(prompt_ids, make_processor()) == [1, 2, IMAGE, IMAGE, 3]


def test_other_processors_are_untouched():
    prompt_ids = [1] + image_block(3) + [2]

    assert gemma4_dedup_image_tokens(prompt_ids, make_processor(SimpleNamespace)) == prompt_ids
