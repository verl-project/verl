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
Merge individual MoE expert weights into stacked tensors for efficient loading.

This script takes a HuggingFace checkpoint with individual expert weights
(e.g., model.layers.{i}.mlp.experts.{j}.gate_proj.weight) and merges them
into stacked tensors (e.g., model.layers.{i}.mlp.experts.gate_proj) for
faster loading and better memory efficiency in VeOmni.

The merging process:
1. Loads individual expert weights from the HF checkpoint
2. Stacks them into single tensors for each projection type
3. Handles all three projection types: gate_proj, up_proj, down_proj
4. Supports both Qwen3-MoE (num_experts) and DeepSeek (n_routed_experts) formats
5. Handles models with initial dense layers (first_k_dense_replace)

Usage: python moe_merge.py --raw_hf_path <input_checkpoint> --merge_hf_path <output_dir>
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


def main(raw_hf_path, merge_hf_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(merge_hf_path, exist_ok=True)

    config = AutoConfig.from_pretrained(raw_hf_path)
    tokenizer = build_tokenizer(raw_hf_path)

    safetensor_files = list(glob(os.path.join(raw_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [StateDictIterator(shard_file) for shard_file in safetensor_files]
    new_state_dict = {}
    for state_dict_iterator in tqdm(state_dict_iterators, desc="Loading checkpoint shards"):
        for name, tensor in state_dict_iterator:
            new_state_dict[name] = tensor.cpu()

    print(new_state_dict.keys())

    if hasattr(config, "num_experts"):
        # qwen3moe
        num_experts = config.num_experts
    elif hasattr(config, "n_routed_experts"):
        # deepseek
        num_experts = config.n_routed_experts
    else:
        raise RuntimeError("could not find how many experts to assign")
    num_hidden_layers = config.num_hidden_layers

    if hasattr(config, "first_k_dense_replace"):
        # deepseek first k dense layer
        moe_layer_start_idx = config.first_k_dense_replace
    else:
        # moe layer only in the model
        moe_layer_start_idx = 0

    for i in range(moe_layer_start_idx, num_hidden_layers):
        gate_proj = []
        for j in range(num_experts):
            gate_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.gate_proj"] = torch.stack(gate_proj)
        up_proj = []
        for j in range(num_experts):
            up_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.up_proj"] = torch.stack(up_proj)
        down_proj = []
        for j in range(num_experts):
            down_proj.append(new_state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"))

        new_state_dict[f"model.layers.{i}.mlp.experts.down_proj"] = torch.stack(down_proj)

    model_assets = [config, tokenizer]
    save_model_weights(merge_hf_path, new_state_dict, model_assets=model_assets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_hf_path", type=str, required=True)
    parser.add_argument("--merge_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.raw_hf_path, args.merge_hf_path)
