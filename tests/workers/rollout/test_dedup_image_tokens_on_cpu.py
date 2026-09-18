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

import pytest

from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens


@pytest.mark.parametrize(
    "processor_name", ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast", "Glm5NextImageProcessor"]
)
@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([], []),
        ([1, 1, 2], [1, 1, 2]),
        ([1, 10, 10, 10, 2, 10, 10, 3], [1, 10, 2, 10, 3]),
        ([10, 10, 11, 11], [10, 11]),
    ],
)
def test_dedup_preserves_text_and_modality_boundaries(processor_name, tokens, expected):
    processor = SimpleNamespace(image_processor=type(processor_name, (), {})(), image_token_id=10, video_token_id=11)
    original = tokens.copy()
    assert qwen2_5_vl_dedup_image_tokens(tokens, processor) == expected
    assert tokens == original


@pytest.mark.parametrize("processor", [None, SimpleNamespace(), SimpleNamespace(image_processor=object())])
def test_unrelated_processors_are_unchanged(processor):
    tokens = [1, 10, 10, 2]
    assert qwen2_5_vl_dedup_image_tokens(tokens, processor) is tokens
