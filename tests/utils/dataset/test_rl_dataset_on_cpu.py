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
import asyncio
import json
import os
import pickle
import ssl
import threading
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

import verl.utils.dataset.rl_dataset as rl_dataset_module
from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _mock_rlhf_dataset():
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = object()
    return dataset


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


class _FakeTokenizer:
    """Count whitespace-separated words without requiring model files."""

    name_or_path = "fake-processor"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        """Return one token per word while checking the filter's template options."""
        assert add_generation_prompt
        assert tokenize
        return list(range(len(messages[0]["content"].split())))

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        """Tokenize a rendered prompt without adding padding or special tokens."""
        assert not add_special_tokens
        assert not return_attention_mask
        return {"input_ids": list(range(len(text.split())))}


class _FakeProcessor:
    """Provide the text-only path of a multimodal processor for isolated filter tests."""

    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        """Render the user content while checking that processor tokenization is disabled."""
        assert add_generation_prompt
        assert not tokenize
        return messages[0]["content"]


class _UnpickleableProcessor(_FakeProcessor):
    """Model a processor holding a live network context that workers cannot deserialize."""

    def __init__(self):
        """Attach the unpicklable state that originally broke multiprocessing."""
        self.context = ssl.create_default_context()


def _mock_filter_dataset(processor=None, num_workers=2):
    """Construct only the dataset state needed to exercise prompt filtering."""
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.processor = processor
    dataset.tokenizer = None if processor is not None else _FakeTokenizer()
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.image_patch_size = 14
    dataset.config = OmegaConf.create({"trust_remote_code": False})
    dataset.max_prompt_length = 2
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.mm_processor_kwargs = {}
    dataset.tool_schemas = None
    dataset.num_workers = num_workers
    return dataset


def test_maybe_filter_out_long_prompts_uses_picklable_multiprocess_processor_filter(monkeypatch):
    """Multiprocessing receives a reload source and serializable state, never the live processor."""
    calls = []

    def load_fake_processor(source, trust_remote_code=False, use_fast=True):
        """Validate the worker's reload options without accessing external model files."""
        assert source == "fake-processor"
        assert not trust_remote_code
        assert use_fast
        return _FakeProcessor()

    monkeypatch.setattr(rl_dataset_module, "_load_filter_processor", load_fake_processor, raising=False)
    monkeypatch.setattr(rl_dataset_module, "_FILTER_PROCESSOR_CACHE", {})

    class FakeDataFrame(list):
        def filter(self, function, num_proc=None, desc=None, fn_kwargs=None):
            """Check worker serialization before applying the predicate locally."""
            calls.append(num_proc)
            if num_proc is not None:
                pickle.dumps((function, fn_kwargs))
                assert fn_kwargs["processor_path"] == "fake-processor"
                assert "processor" not in fn_kwargs
            return FakeDataFrame([doc for doc in self if function(doc, **(fn_kwargs or {}))])

    dataset = _mock_filter_dataset(processor=_UnpickleableProcessor(), num_workers=2)
    dataframe = FakeDataFrame(
        [
            {"prompt": [{"role": "user", "content": "short"}]},
            {"prompt": [{"role": "user", "content": "one two three"}]},
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert calls == [2]
    assert len(filtered) == 1
    assert filtered[0]["prompt"][0]["content"] == "short"


@pytest.mark.parametrize("reload_path", [None, ""])
def test_maybe_filter_out_long_prompts_disables_multiprocessing_without_processor_path(reload_path):
    """A processor without a reload source must remain in the serial filtering path."""
    calls = []

    class PathlessProcessor(_UnpickleableProcessor):
        tokenizer = type("PathlessTokenizer", (_FakeTokenizer,), {"name_or_path": reload_path})()

    class FakeDataFrame(list):
        def filter(self, function, num_proc=None, desc=None, fn_kwargs=None):
            """Check serial execution and keep live objects out of fingerprint kwargs."""
            calls.append(num_proc)
            assert num_proc is None
            assert fn_kwargs is None
            return FakeDataFrame([doc for doc in self if function(doc, **(fn_kwargs or {}))])

    dataset = _mock_filter_dataset(processor=PathlessProcessor(), num_workers=2)
    dataframe = FakeDataFrame(
        [
            {"prompt": [{"role": "user", "content": "short"}]},
            {"prompt": [{"role": "user", "content": "one two three"}]},
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert calls == [None]
    assert len(filtered) == 1
    assert filtered[0]["prompt"][0]["content"] == "short"


@pytest.mark.parametrize("reload_path", [None, ""])
def test_maybe_filter_out_long_prompts_disables_multiprocessing_without_tokenizer_path(reload_path):
    """Missing and empty tokenizer sources must preserve the live tokenizer in serial mode."""
    calls = []

    class PathlessTokenizer(_FakeTokenizer):
        name_or_path = reload_path

        def __init__(self):
            """Ensure the fallback also handles genuinely unpicklable tokenizer state."""
            self.context = ssl.create_default_context()

    class FakeDataFrame(list):
        def filter(self, function, num_proc=None, desc=None, fn_kwargs=None):
            """Verify that filtering uses the serial callable without object kwargs."""
            calls.append(num_proc)
            assert num_proc is None
            assert fn_kwargs is None
            return FakeDataFrame([doc for doc in self if function(doc, **(fn_kwargs or {}))])

    dataset = _mock_filter_dataset(num_workers=2)
    dataset.tokenizer = PathlessTokenizer()
    dataframe = FakeDataFrame(
        [
            {"prompt": [{"role": "user", "content": "short"}]},
            {"prompt": [{"role": "user", "content": "one two three"}]},
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert calls == [None]
    assert len(filtered) == 1
    assert filtered[0]["prompt"][0]["content"] == "short"


def _make_local_filter_models(tmp_path, use_processor):
    """Build reloadable tokenizer/processor fixtures without model downloads."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLConfig,
        Qwen2VLImageProcessor,
        Qwen2VLProcessor,
        Qwen2VLVideoProcessor,
    )

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "short": 2, "extra": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    )
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer_path = tmp_path / "tokenizer"
    processor = None
    if use_processor:
        processor = Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(),
            video_processor=Qwen2VLVideoProcessor(),
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )
        processor.save_pretrained(tokenizer_path)
        Qwen2VLConfig().save_pretrained(tokenizer_path)
        processor = hf_processor(str(tokenizer_path))
        assert processor is not None
    else:
        tokenizer.save_pretrained(tokenizer_path)
    tokenizer = hf_tokenizer(str(tokenizer_path))
    return tokenizer, processor


@pytest.mark.parametrize("num_workers", [None, 2])
@pytest.mark.parametrize("use_processor", [False, True])
def test_prompt_filter_preserves_runtime_chat_template(tmp_path, num_workers, use_processor):
    """Real filtering must retain runtime templates without pickling live SSL contexts."""
    from datasets import Dataset

    tokenizer, processor = _make_local_filter_models(tmp_path, use_processor)
    formatter = processor if use_processor else tokenizer
    formatter.chat_template += " extra extra"
    # Reproduce the original failure with a live object that cannot be pickled.
    formatter.context = ssl.create_default_context()
    with pytest.raises(TypeError, match="SSLContext"):
        pickle.dumps(formatter)

    data_path = tmp_path / "prompts.parquet"
    Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": "short"}]},
            {"prompt": [{"role": "user", "content": "one two three"}]},
        ]
    ).to_parquet(data_path)
    dataset = RLHFDataset(
        data_files=str(data_path),
        tokenizer=tokenizer,
        processor=processor,
        config=OmegaConf.create(
            {"max_prompt_length": 3, "filter_overlong_prompts_workers": num_workers, "cache_dir": str(tmp_path)}
        ),
    )

    assert len(dataset) == 1
    assert dataset[0]["raw_prompt"][0]["content"] == "short"


@pytest.mark.parametrize("use_processor", [False, True])
def test_prompt_filter_propagates_worker_load_failure(tmp_path, use_processor):
    """A real worker loading an invalid checkpoint must fail instead of emptying the dataset."""
    from datasets import Dataset

    dataset = _mock_filter_dataset(processor=_FakeProcessor() if use_processor else None)
    formatter = dataset.processor if use_processor else dataset.tokenizer
    formatter.name_or_path = str(tmp_path)
    dataframe = Dataset.from_list([{"prompt": [{"role": "user", "content": "short"}]}] * 2)

    with pytest.raises(ValueError):
        dataset.maybe_filter_out_long_prompts(dataframe)


@pytest.mark.parametrize("use_processor", [False, True])
def test_prompt_filter_still_skips_malformed_samples(use_processor):
    """Only a malformed row is dropped when a valid live formatter is already available."""
    from datasets import Dataset

    dataset = _mock_filter_dataset(processor=_FakeProcessor() if use_processor else None, num_workers=None)
    dataframe = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": text}]} for text in ["short", None, "one two three"]]
    )
    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert filtered["prompt"] == [[{"role": "user", "content": "short"}]]


@pytest.mark.parametrize("num_workers", [None, 2])
@pytest.mark.parametrize("use_processor", [False, True])
@pytest.mark.parametrize("selection", ["default", "tool_use", "custom"])
def test_prompt_filter_preserves_named_templates(tmp_path, num_workers, use_processor, selection):
    """Native default/tool/named selection must see runtime templates in either execution mode."""
    from datasets import Dataset

    tokenizer, processor = _make_local_filter_models(tmp_path, use_processor)
    formatter = processor if use_processor else tokenizer
    formatter.chat_template = {
        "default": "{{ messages[0]['content'] }} extra extra",
        "tool_use": "{{ messages[0]['content'] }} extra extra extra extra",
        "custom": "{{ messages[0]['content'] }} extra",
    }
    dataset = _mock_filter_dataset(processor=processor, num_workers=num_workers)
    dataset.tokenizer = tokenizer
    dataset.max_prompt_length = 3
    if selection == "custom":
        dataset.apply_chat_template_kwargs = {"chat_template": "custom"}
        dataset.max_prompt_length = 2
    elif selection == "tool_use":
        dataset.tool_schemas = [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
        ]
        # Processors select their default template; tokenizers auto-select tool_use.
        dataset.max_prompt_length = 3 if use_processor else 5
    dataframe = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": text}]} for text in ["short", "one two three"]]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert filtered["prompt"] == [[{"role": "user", "content": "short"}]]


class _CustomMultimodalDataset(RLHFDataset):
    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        """Reject rows according to the custom dataset's media configuration."""
        if messages[0]["content"] != config.keep_prompt:
            raise ValueError("Unsupported custom media sample")
        return None, None, None


class _CustomAudioDataset(RLHFDataset):
    @classmethod
    def _extract_audio_info(cls, messages):
        """Reject rows that violate a custom audio extraction rule."""
        if messages[0]["content"] != "short":
            raise ValueError("Unsupported custom audio sample")
        return None


@pytest.mark.parametrize("num_workers", [None, 2])
@pytest.mark.parametrize("dataset_class", [_CustomMultimodalDataset, _CustomAudioDataset])
def test_prompt_filter_preserves_custom_media_hooks(tmp_path, num_workers, dataset_class):
    """Real multiprocessing must honor class hooks without capturing a live dataset instance."""
    from datasets import Dataset

    tokenizer, processor = _make_local_filter_models(tmp_path, use_processor=True)
    dataset = _mock_filter_dataset(processor=processor, num_workers=num_workers)
    dataset.__class__ = dataset_class
    dataset.tokenizer = tokenizer
    dataset.config.keep_prompt = "short"
    dataset.max_prompt_length = 10
    dataset.context = ssl.create_default_context()
    dataframe = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": text}]} for text in ["short", "one two three"]]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert filtered["prompt"] == [[{"role": "user", "content": "short"}]]


def test_rl_dataset():
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


def test_build_messages_accepts_image_path():
    dataset = _mock_rlhf_dataset()
    image_path = "file:///tmp/image.jpg"
    example = {
        "prompt": [{"role": "user", "content": "Describe <image>"}],
        "images": [image_path],
    }

    messages = dataset._build_messages(example, key=dataset.prompt_key)

    assert messages[0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "image", "image": image_path},
    ]


@pytest.mark.parametrize(
    ("videos", "expected_video"),
    [
        (["file:///tmp/video.mp4"], "file:///tmp/video.mp4"),
        ([["file:///tmp/frame1.jpg", "file:///tmp/frame2.jpg"]], ["file:///tmp/frame1.jpg", "file:///tmp/frame2.jpg"]),
        ([[Path("/tmp/frame1.jpg"), Path("/tmp/frame2.jpg")]], ["/tmp/frame1.jpg", "/tmp/frame2.jpg"]),
    ],
)
def test_build_messages_accepts_video_path_or_frame_list(videos, expected_video):
    dataset = _mock_rlhf_dataset()
    example = {
        "prompt": [{"role": "user", "content": "Describe <video>"}],
        "videos": videos,
    }

    messages = dataset._build_messages(example, key=dataset.prompt_key)

    assert messages[0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "video", "video": expected_video},
    ]


def test_maybe_filter_out_long_prompts_accepts_image_path(monkeypatch):
    """Image paths must reach the dataset's media hook before prompt length is measured."""

    class FakeDataFrame(list):
        def filter(self, fn, num_proc=None, desc=None, fn_kwargs=None):
            """Apply the predicate with the same kwargs contract as Dataset.filter."""
            return FakeDataFrame([doc for doc in self if fn(doc, **(fn_kwargs or {}))])

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
            return "formatted prompt"

        def __call__(self, **kwargs):
            assert kwargs["images"] == ["processed-image"]
            return {"input_ids": [[1, 2, 3]]}

    captured = {}

    def fake_process_multi_modal_info(cls, messages, image_patch_size, config):
        captured["messages"] = deepcopy(messages)
        return ["processed-image"], None, None

    monkeypatch.setattr(
        RLHFDataset,
        "_process_multi_modal_info",
        classmethod(fake_process_multi_modal_info),
    )

    dataset = _mock_rlhf_dataset()
    dataset.processor = FakeProcessor()
    dataset.tokenizer = None
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.image_patch_size = 14
    dataset.config = OmegaConf.create({})
    dataset.max_prompt_length = 10
    dataset.mm_processor_kwargs = {}
    dataset.num_workers = None

    image_path = "file:///tmp/image.jpg"
    dataframe = FakeDataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Describe <image>"}],
                "images": [image_path],
            }
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "image", "image": image_path},
    ]


def test_process_multi_modal_info_keeps_event_loop_responsive(monkeypatch):
    processing_started = threading.Event()
    observer_ran = threading.Event()

    def blocking_process_multi_modal_info(cls, messages, image_patch_size, config):
        processing_started.set()
        observer_ran.wait(timeout=0.2)
        return None, None, None

    monkeypatch.setattr(
        RLHFDataset,
        "_process_multi_modal_info",
        classmethod(blocking_process_multi_modal_info),
    )

    async def run():
        async def observe_event_loop():
            await asyncio.sleep(0)
            observer_ran.set()

        observer = asyncio.create_task(observe_event_loop())
        await RLHFDataset.process_multi_modal_info([], image_patch_size=14, config=OmegaConf.create({}))
        observer_ran_while_processing = observer_ran.is_set()
        await observer
        return observer_ran_while_processing

    assert asyncio.run(run())
    assert processing_started.is_set()


def test_image_rl_data():
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
