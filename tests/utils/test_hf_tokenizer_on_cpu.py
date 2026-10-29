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
import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from verl.utils.tokenizer.tokenizer import _PROBE, hf_tokenizer


class TestHonorTokenizerJson:
    @pytest.mark.parametrize(
        "tokenizer_class",
        ["PreTrainedTokenizerFast", "LlamaTokenizerFast"],
        ids=["generic-class", "llama-class-on-byte-level-vocab"],
    )
    def test_whitespace_survives(self, tmp_path, tokenizer_class):
        raw = Tokenizer(models.BPE())
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<s>", "</s>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        raw.train_from_iterator([_PROBE] * 4, trainer)
        raw.save(str(tmp_path / "tokenizer.json"))
        config = {"tokenizer_class": tokenizer_class, "bos_token": "<s>", "eos_token": "</s>", "pad_token": "</s>"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))

        tokenizer = hf_tokenizer(str(tmp_path))
        ids = tokenizer(_PROBE, add_special_tokens=False)["input_ids"]

        assert ids == raw.encode(_PROBE, add_special_tokens=False).ids
        assert tokenizer.decode(ids) == _PROBE
