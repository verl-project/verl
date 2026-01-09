# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset import QwenDataset
from verl.utils.dataset.rl_dataset import collate_fn


def get_ocr_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/ocr/")
    local_path = os.path.join(local_folder, "train.txt")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_qwen_dataset():
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen-Image"), trust_remote_code=True)
    local_path = get_ocr_data()
    config = OmegaConf.create(
        {
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "data_source": "ocr",
        }
    )
    dataset = QwenDataset(data_files=local_path, tokenizer=tokenizer, config=config)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert len(data_proto) == 16
    assert "input_ids" in data_proto.batch
    assert "attention_mask" in data_proto.batch


def test_qwen_dataset_with_max_samples():
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen-Image"), trust_remote_code=True)
    local_path = get_ocr_data()
    config = OmegaConf.create(
        {
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "data_source": "ocr",
        }
    )
    dataset = QwenDataset(data_files=local_path, tokenizer=tokenizer, config=config, max_samples=5)
    assert len(dataset) == 5

    # test split
    dataset_split = dataset.split(5)
    assert len(dataset_split) == 5
