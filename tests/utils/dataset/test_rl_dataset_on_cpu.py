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
import json
import os

import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl import DataProto
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/deepseek-coder-1.3b-instruct"))
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)

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
    assert "raw_prompt" in data_proto.non_tensor_batch


def test_rl_dataset_with_max_samples():
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/deepseek-coder-1.3b-instruct"))
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
            "max_samples": 5,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config, max_samples=5)
    assert len(dataset) == 5


def test_image_rl_data():
    from verl import DataProto
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,  # num_workers=1 hang in ci
        }
    )
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/geo3k/train.parquet"),
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

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
    assert "images" not in data_proto.non_tensor_batch

    for prompt in data_proto.non_tensor_batch["raw_prompt"]:
        assert len(prompt) == 1
        prompt = prompt[0]
        role, content = prompt["role"], prompt["content"]
        assert role == "user"
        assert len(content) == 2
        assert content[0]["type"] == "image" and isinstance(content[0]["image"], Image.Image)
        assert content[1]["type"] == "text" and isinstance(content[1]["text"], str)

    print("raw_prompt", data_proto.non_tensor_batch["raw_prompt"][0])


@pytest.fixture
def video_data_file():
    data = [
        {
            "problem_id": 17,
            "problem": "How does the crowd's excitement change as the match progresses?",
            "data_type": "video",
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "LLaVA-Video-178K/academic_source/activitynet/v_2g9GrshWQrU.mp4"},
                        {
                            "type": "text",
                            "text": "How does the crowd's excitement change as the match progresses? "
                            "A. It fluctuates; B. It decreases; C. It builds up; D. It remains the same. "
                            "Put your answer in <answer></answer>",
                        },
                    ],
                }
            ],
            "problem_type": "multiple choice",
            "solution": "C",
            "data_source": "LLaVA-Video-178K/2_3_m_academic_v0_1",
        }
    ] * 30

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_video.json"
    with open(test_file, "w") as f:
        json.dump(data, f, indent=2)

    return test_file


def test_video_rl_data(video_data_file):
    from verl import DataProto
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
        }
    )
    dataset = RLHFDataset(
        data_files=video_data_file,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    tensors = {}
    non_tensors = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert len(data_proto) == 16
    assert "images" not in data_proto.non_tensor_batch

    print("raw_prompt", data_proto.non_tensor_batch["raw_prompt"][0])


def test_maybe_filter_out_long_prompts_keeps_video_payload(monkeypatch):
    class FakeProcessor:
        def __init__(self):
            self.tokenizer = self

        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            assert messages[0]["content"][0]["type"] == "video"
            return "prompt"

        def __call__(self, text, images=None, videos=None, videos_kwargs=None):
            assert images is None
            assert videos is not None
            assert len(videos) == 1
            assert videos[0].shape == (2, 3, 4, 4)
            return {"input_ids": [[1, 2, 3]]}

    def fake_process_video(video, image_patch_size, return_video_metadata):
        assert video["video"] == "demo.mp4"
        assert return_video_metadata is True
        return torch.ones(2, 3, 4, 4), {"fps": 2}

    monkeypatch.setattr("verl.utils.dataset.vision_utils.process_video", fake_process_video)

    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.image_patch_size = 14
    dataset.max_prompt_length = 16
    dataset.filter_overlong_prompts = True
    dataset.num_workers = None
    dataset.processor = FakeProcessor()
    dataset.tokenizer = None
    dataset.tool_schemas = None
    dataset.apply_chat_template_kwargs = {}

    row = {
        "prompt": [{"role": "user", "content": "<video>Describe the clip."}],
        "videos": [{"video": "demo.mp4"}],
        "data_source": "unit-test",
    }
    dataframe = Dataset.from_list([row])

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
    kept_row = filtered[0]
    assert kept_row["videos"] == [{"video": "demo.mp4"}]

    raw_prompt = dataset._build_messages(kept_row)
    assert raw_prompt[0]["content"][0]["type"] == "video"
    assert kept_row["videos"] == [{"video": "demo.mp4"}]
