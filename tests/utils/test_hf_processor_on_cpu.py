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

import types
import warnings

import pytest
import transformers
from transformers import PreTrainedTokenizerBase

from verl.utils.tokenizer import hf_processor


def _patch_loaders(monkeypatch, processor, config=None):
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)


def _instance(class_name):
    return type(class_name, (), {})()


def test_tokenizer_fallback_returns_none(monkeypatch):
    _patch_loaders(monkeypatch, PreTrainedTokenizerBase.__new__(PreTrainedTokenizerBase))
    assert hf_processor("model") is None


def test_supported_processor_gets_rope_index(monkeypatch):
    processor, config = _instance("Qwen2VLProcessor"), types.SimpleNamespace()
    _patch_loaders(monkeypatch, processor, config)

    assert hf_processor("model") is processor
    assert processor.config is config
    assert processor.get_rope_index.__func__ is transformers.models.qwen2_vl.Qwen2VLModel.get_rope_index


def test_unsupported_processor_returns_none_with_warning(monkeypatch):
    _patch_loaders(monkeypatch, _instance("Gemma3Processor"), types.SimpleNamespace())
    with pytest.warns(UserWarning, match="Gemma3Processor is not supported by verl"):
        assert hf_processor("model") is None


def test_non_processor_returns_none_without_warning(monkeypatch):
    _patch_loaders(monkeypatch, _instance("WhisperFeatureExtractor"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hf_processor("model") is None


@pytest.mark.parametrize("loader", ["AutoProcessor", "AutoConfig"])
def test_load_errors_propagate(monkeypatch, loader):
    _patch_loaders(monkeypatch, _instance("Qwen2VLProcessor"), types.SimpleNamespace())

    def _fail(*args, **kwargs):
        raise OSError("cannot load")

    monkeypatch.setattr(getattr(transformers, loader), "from_pretrained", _fail)
    with pytest.raises(OSError, match="cannot load"):
        hf_processor("model")
