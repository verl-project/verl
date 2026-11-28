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

"""Unit tests for default EOS stop token collection in the SGLang rollout."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from verl.workers.rollout.sglang_rollout.utils import default_stop_token_ids


def _model_config(*, hf_config: Any = None, generation_config: Any = None, tokenizer: Any = None) -> SimpleNamespace:
    return SimpleNamespace(hf_config=hf_config, generation_config=generation_config, tokenizer=tokenizer)


def test_collects_and_deduplicates_across_sources():
    cfg = _model_config(
        hf_config=SimpleNamespace(eos_token_id=151645),
        generation_config=SimpleNamespace(eos_token_id=[151645, 151643]),
        tokenizer=SimpleNamespace(eos_token_id=151645),
    )
    assert default_stop_token_ids(cfg) == (151645, 151643)


def test_vlm_text_config_and_missing_generation_config():
    cfg = _model_config(
        hf_config=SimpleNamespace(eos_token_id=None, text_config=SimpleNamespace(eos_token_id=248044)),
        generation_config=None,
        tokenizer=SimpleNamespace(eos_token_id=248046),
    )
    assert default_stop_token_ids(cfg) == (248044, 248046)


def test_hf_config_list_and_distinct_tokenizer_eos():
    cfg = _model_config(
        hf_config=SimpleNamespace(eos_token_id=[128001, 128008]),
        generation_config=None,
        tokenizer=SimpleNamespace(eos_token_id=128009),
    )
    assert default_stop_token_ids(cfg) == (128001, 128008, 128009)


def test_filters_none_bool_and_normalizes_numpy_ints():
    cfg = _model_config(
        hf_config=SimpleNamespace(eos_token_id=None),
        generation_config=SimpleNamespace(eos_token_id=[None, True, False, np.int64(2)]),
    )
    result = default_stop_token_ids(cfg)
    assert result == (2,)
    assert type(result[0]) is int
    assert default_stop_token_ids(_model_config()) == ()
