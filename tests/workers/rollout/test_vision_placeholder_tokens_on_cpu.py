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
"""The vision placeholder tokens handed to vLLM as bad_words, so the policy cannot sample them."""

from __future__ import annotations

from types import SimpleNamespace

from verl.workers.rollout.utils import add_vision_bad_words, get_vision_placeholder_tokens

TOKENS = {151655: "<|image_pad|>", 151656: "<|video_pad|>"}


def make_processor(**attrs):
    tokenizer = SimpleNamespace(convert_ids_to_tokens=TOKENS.get)
    return SimpleNamespace(tokenizer=tokenizer, **attrs)


def test_resolves_both_placeholders_from_token_ids():
    processor = make_processor(image_token_id=151655, video_token_id=151656)

    assert get_vision_placeholder_tokens(processor) == ["<|image_pad|>", "<|video_pad|>"]


def test_resolves_placeholders_from_token_strings():
    """Newer processors expose image_token/video_token strings instead of the ids."""
    processor = make_processor(image_token="<|image_pad|>", video_token="<|video_pad|>")
    processor.tokenizer.convert_tokens_to_ids = {v: k for k, v in TOKENS.items()}.get

    assert get_vision_placeholder_tokens(processor) == ["<|image_pad|>", "<|video_pad|>"]


def test_image_only_processor_bans_only_the_image_placeholder():
    processor = make_processor(image_token_id=151655)

    assert get_vision_placeholder_tokens(processor) == ["<|image_pad|>"]


def test_text_only_model_leaves_sampling_untouched():
    assert get_vision_placeholder_tokens(None) == []


def test_bad_words_are_added_when_the_caller_passed_none():
    sampling_params = {"temperature": 1.0}

    add_vision_bad_words(sampling_params, make_processor(image_token_id=151655, video_token_id=151656))

    assert sampling_params["bad_words"] == ["<|image_pad|>", "<|video_pad|>"]


def test_placeholders_are_merged_into_the_callers_bad_words():
    """Asking to ban some other word is not a request to allow the placeholders through."""
    sampling_params = {"bad_words": ["foo"]}

    add_vision_bad_words(sampling_params, make_processor(image_token_id=151655, video_token_id=151656))

    assert sampling_params["bad_words"] == ["foo", "<|image_pad|>", "<|video_pad|>"]


def test_merging_does_not_duplicate_placeholders():
    sampling_params = {"bad_words": ["<|image_pad|>"]}

    add_vision_bad_words(sampling_params, make_processor(image_token_id=151655, video_token_id=151656))

    assert sampling_params["bad_words"] == ["<|image_pad|>", "<|video_pad|>"]


def test_text_only_model_gets_no_bad_words_key():
    sampling_params = {"temperature": 1.0}

    add_vision_bad_words(sampling_params, None)

    assert sampling_params == {"temperature": 1.0}
