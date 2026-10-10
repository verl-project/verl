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

import pytest
from transformers import (
    LlavaProcessor,
    Qwen2_5_VLProcessor,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
    Qwen3VLProcessor,
)

from verl.utils.transformers_compat import normalize_mm_processor_kwargs


def _uninitialized(processor_type):
    """Create a real Transformers processor instance without loading a checkpoint."""
    return object.__new__(processor_type)


@pytest.mark.parametrize("processor_type", [Qwen2VLProcessor, Qwen2_5_VLProcessor, Qwen3VLProcessor])
def test_normalize_mm_processor_kwargs_disables_qwen_vl_resize(processor_type):
    assert normalize_mm_processor_kwargs(_uninitialized(processor_type)) == {"do_resize": False}


@pytest.mark.parametrize("do_resize", [True, False])
def test_normalize_mm_processor_kwargs_preserves_explicit_resize(do_resize):
    kwargs = {"do_resize": do_resize, "max_pixels": 1234}

    normalized = normalize_mm_processor_kwargs(_uninitialized(Qwen3VLProcessor), kwargs)

    assert normalized == kwargs
    assert normalized is not kwargs


def test_normalize_mm_processor_kwargs_does_not_mutate_input():
    kwargs = {"max_pixels": 1234}

    normalized = normalize_mm_processor_kwargs(_uninitialized(Qwen2VLProcessor), kwargs)

    assert kwargs == {"max_pixels": 1234}
    assert normalized == {"max_pixels": 1234, "do_resize": False}


@pytest.mark.parametrize("processor_type", [Qwen2Tokenizer, LlavaProcessor])
def test_normalize_mm_processor_kwargs_is_noop_for_non_qwen_vl(processor_type):
    processor = _uninitialized(processor_type)
    kwargs = {"max_pixels": 1234}

    normalized = normalize_mm_processor_kwargs(processor, kwargs)

    assert normalized == kwargs
    assert normalized is not kwargs


def test_normalize_mm_processor_kwargs_is_noop_without_processor():
    assert normalize_mm_processor_kwargs(None) == {}
