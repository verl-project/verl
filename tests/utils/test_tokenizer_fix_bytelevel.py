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

from unittest.mock import patch
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import ByteLevel

from verl.utils.tokenizer import hf_tokenizer


class _FakeTokenizer:
    def __init__(self, pre_tokenizer, decoder):
        self.backend_tokenizer = type("B", (), {"pre_tokenizer": pre_tokenizer, "decoder": decoder})()

    def encode(self, text, add_special_tokens=False):
        return [p for p, _ in self.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]

    def decode(self, tokens):
        return self.backend_tokenizer.decoder.decode(tokens)


def test_hf_tokenizer_fixes_bytelevel_mismatch():
    # Simulate transformers>=5 broken state: Metaspace pre_tokenizer, ByteLevel vocabulary.
    # hf_tokenizer must patch pre_tokenizer to ByteLevel so encode→decode preserves spaces.
    tok = _FakeTokenizer(Metaspace(), ByteLevel())
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
        result = hf_tokenizer("dummy-model", correct_pad_token=False)
    assert " " in result.decode(result.encode("a b"))
