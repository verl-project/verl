# Copyright 2025 Bytedance Ltd. and/or its affiliates

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
Reverse process of moe_merge.py - splits merged MoE expert weights back to individual experts.

This script takes a HF checkpoint that has been processed by moe_merge.py (where expert weights
are stacked into single tensors) and splits them back to the original format with individual
expert weights.

The process reverses the merging by:
1. Loading stacked tensors like model.layers.{i}.mlp.experts.gate_proj
2. Unstacking them back to individual experts model.layers.{i}.mlp.experts.{j}.gate_proj.weight
3. Handling all three projection types: gate_proj, up_proj, down_proj

Usage: python moe_split.py --merge_hf_path <merged_checkpoint> --split_hf_path <output_dir>
"""

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator

import torch
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoConfig
from veomni.models import build_tokenizer, save_model_weights


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def main(merge_hf_path, split_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(split_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(merge_hf_path)
    tokenizer = build_tokenizer(merge_hf_path)

    safetensor_files = list(glob(os.path.join(merge_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    num_experts = config.num_experts
    num_hidden_layers = config.num_hidden_layers
    for i in range(num_hidden_layers):
        print(f"Converting layer {i}")
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            stacked_key = f"model.layers.{i}.mlp.experts.{proj_name}"
            if stacked_key in new_state_dict:
                stacked_tensor = new_state_dict.pop(stacked_key)
                for j in range(num_experts):
                    expert_key = f"model.layers.{i}.mlp.experts.{j}.{proj_name}.weight"
                    new_state_dict[expert_key] = stacked_tensor[j]

    model_assets = [config, tokenizer]

    print("Saving to safetensors")
    save_model_weights(split_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--merge_hf_path", type=str, required=True)
    parser.add_argument("--split_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.merge_hf_path, args.split_hf_path)
