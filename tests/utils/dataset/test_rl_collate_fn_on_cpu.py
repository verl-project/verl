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
import torch


def test_rl_collate_fn():
    from verl.utils.dataset.rl_dataset import collate_fn

    max_prompt_length = 5

    test_data = [
        {
            # test tensor
            "input_ids": torch.randint(0, 10, (max_prompt_length,)),
            # test fixed length (1) list within a batch
            "messages": [{"role": "user", "content": "Hi."}],
            # test variable length list within a batch
            "raw_prompt_ids": [1, 2, 3, 4],
            # test string
            "ability": "math",
            # test dict
            "reward_model": {"ground_truth": 5, "style": "rule"},
            # test empty dict
            "tools_kwargs": {},
        },
        {
            "input_ids": torch.randint(0, 10, (max_prompt_length,)),
            "messages": [{"role": "user", "content": "Hello."}],
            "raw_prompt_ids": [1, 2, 3],
            "ability": "toolcall",
            "reward_model": {
                "ground_truth": '[{"name": "rgb_to_cmyk", "arguments": {"r": 0, "g": 0, "b": 255}}]',
                "style": "rule",
            },
            "tools_kwargs": {},
        },
    ]

    batch_size = len(test_data)
    batch = collate_fn(test_data)

    # Tensor part
    assert batch["input_ids"].shape == (batch_size, max_prompt_length)
    assert isinstance(batch["input_ids"], torch.Tensor)

    # Non-tensor parts
    expected_types = {
        "messages": list,
        "raw_prompt_ids": list,
        "ability": str,
        "reward_model": dict,
        "tools_kwargs": dict,
    }

    for key, dtype in expected_types.items():
        assert batch[key].shape == (batch_size,), (
            f"Expected shape {(batch_size,)} for '{key}', but got {batch[key].shape}"
        )
        assert isinstance(batch[key][0], dtype), (
            f"'{key}' should contain elements of type {dtype}, but got {type(batch[key][0])}"
        )
