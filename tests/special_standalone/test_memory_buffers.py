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
"""
Test memory buffers
- We start with two models with the same weights
- We use Memory buffer to make one of the models and then compare the parameters
"""

import gc

import torch
from transformers import LlamaConfig, LlamaModel


def test_memory_buffers():
    llama_config = LlamaConfig(
        vocab_size=256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
    )

    model = LlamaModel(config=llama_config).cuda()
    model_copy = LlamaModel(config=llama_config).cuda()
    model_copy.load_state_dict(model.state_dict())

    norm_factor = 1024**3

    t_before = torch.cuda.get_device_properties(0).total_memory / norm_factor
    r_before = torch.cuda.memory_reserved(0) / norm_factor
    a_before = torch.cuda.memory_allocated(0) / norm_factor

    print(f"Before Total memory: {t_before} GB, reserved: {r_before} GB, allocated: {a_before} GB")

    t = torch.cuda.get_device_properties(0).total_memory / norm_factor
    r = torch.cuda.memory_reserved(0) / norm_factor
    a = torch.cuda.memory_allocated(0) / norm_factor

    gc.collect()
    torch.cuda.empty_cache()

    print(f"After Total memory: {t} GB, reserved: {r} GB, allocated: {a} GB")

    change_ratio = (a - a_before) / a_before
    assert change_ratio < 0.01, f"make sure the allocated change is less than 1%, Got {change_ratio}"

    for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_copy.named_parameters(), strict=True):
        assert name1 == name2
        assert torch.eq(param1.data, param2.data).all(), f"{param1.data}, {param2.data}, {name1}"


if __name__ == "__main__":
    test_memory_buffers()
