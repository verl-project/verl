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
import os

from verl.utils import hf_tokenizer
from verl.utils.dataset.sft_dataset import SFTDataset


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    return local_path


def test_sft_cot_dataset():
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"))
    local_path = get_gsm8k_data()
    from omegaconf import OmegaConf

    dataset = SFTDataset(
        parquet_files=local_path,
        tokenizer=tokenizer,
        config=OmegaConf.create(
            {
                "prompt_key": "prompt",
                "prompt_dict_keys": ["content"],
                "response_key": "extra_info",
                "response_dict_keys": ["answer"],
                "max_length": 512,
            }
        ),
    )

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    assert len(output) > 1
    assert isinstance(output, str)


def test_sft_dataset():
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"))
    local_path = get_gsm8k_data()
    from omegaconf import OmegaConf

    dataset = SFTDataset(
        parquet_files=local_path,
        tokenizer=tokenizer,
        config=OmegaConf.create(
            {
                "prompt_key": "extra_info",
                "prompt_dict_keys": ["question"],
                "response_key": "extra_info",
                "response_dict_keys": ["answer"],
                "max_length": 512,
            }
        ),
    )

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    assert len(output) > 1
    assert isinstance(output, str)


def test_sft_dataset_with_max_samples():
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"))
    local_path = get_gsm8k_data()
    from omegaconf import OmegaConf

    dataset = SFTDataset(
        parquet_files=local_path,
        tokenizer=tokenizer,
        config=OmegaConf.create(
            {
                "prompt_key": "extra_info",
                "prompt_dict_keys": ["question"],
                "response_key": "extra_info",
                "response_dict_keys": ["answer"],
                "max_length": 512,
            }
        ),
        max_samples=5,
    )

    assert len(dataset) == 5
