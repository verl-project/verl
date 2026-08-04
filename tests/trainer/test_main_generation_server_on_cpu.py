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

import numpy as np

from verl.trainer.main_generation_server import _chat_list_to_object_array


def test_chat_list_to_object_array_preserves_varying_turn_counts():
    chats = [
        [{"role": "user", "content": "one"}],
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    ]

    chat_array = _chat_list_to_object_array(chats)

    assert chat_array.shape == (2,)
    assert chat_array.dtype == object
    assert chat_array.tolist() == chats


def test_chat_list_to_object_array_does_not_add_a_turn_axis():
    chats = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]

    chat_array = _chat_list_to_object_array(chats)

    assert chat_array.shape == (2,)
    assert chat_array.tolist() == chats


def test_chat_object_array_splits_without_changing_order():
    chats = [[{"role": "user", "content": str(i)}] for i in range(5)]

    chat_array = _chat_list_to_object_array(chats)
    splits = [split.tolist() for split in np.array_split(chat_array, 3)]

    assert splits == [chats[:2], chats[2:4], chats[4:]]
