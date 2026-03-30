# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from collections import UserDict

from verl.utils.chat_template import extract_system_prompt_and_generation, initialize_system_prompt


class FakeBatchEncoding(UserDict):
    pass


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=False,
        return_dict=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
                "kwargs": kwargs,
            }
        )
        assert tokenize is True
        if return_dict is False:
            user_turns = len(messages)
            base = [100, 101]
            per_turn = []
            for idx in range(user_turns):
                per_turn.extend([10 + idx, 20 + idx, 30 + idx])
            if add_generation_prompt:
                return base + per_turn + [999]
            return base + per_turn
        return FakeBatchEncoding({"input_ids": [1, 2], "attention_mask": [1, 1]})


def test_extract_system_prompt_and_generation_forces_return_dict_false():
    tokenizer = FakeTokenizer()

    system_prompt, generation_prompt = extract_system_prompt_and_generation(tokenizer)

    assert system_prompt == [100, 101]
    assert generation_prompt == [999]
    assert len(tokenizer.calls) == 3
    assert all(call["return_dict"] is False for call in tokenizer.calls)


def test_initialize_system_prompt_passes_apply_chat_template_kwargs():
    tokenizer = FakeTokenizer()

    system_prompt = initialize_system_prompt(tokenizer, chat_template="custom-template")

    assert system_prompt == [100, 101]
    assert len(tokenizer.calls) == 2
    assert all(call["return_dict"] is False for call in tokenizer.calls)
    assert all(call["kwargs"]["chat_template"] == "custom-template" for call in tokenizer.calls)
