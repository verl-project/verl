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
import sys
import types
from dataclasses import dataclass

import pytest
import torch

from verl.experimental.agent_loop.preprocessed_multimodal import (
    _qwen3_video_replacement,
    build_vllm_preprocessed_multimodal_input,
    refresh_vllm_preprocessed_multimodal_prompt_ids,
)


class DummyTokenizer:
    token_ids = {
        "<|image_pad|>": 101,
        "<|video_pad|>": 102,
        "<|vision_start|>": 103,
        "<|vision_end|>": 104,
    }

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "<0.0 seconds>":
            return [201]
        if text == "<0.1 seconds>":
            return [202]
        raise AssertionError(f"Unexpected timestamp text: {text}")


class DummyQwen3VLVideoProcessor:
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 1
    max_frames = 128


class DummyQwen3VLImageProcessor:
    merge_size = 2


class DummyQwen2VLVideoProcessor:
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 1
    max_frames = 128


class DummyQwen2VLImageProcessor:
    merge_size = 2


class DummyQwen3VLProcessor:
    model_type = "qwen3_vl"
    image_processor = DummyQwen3VLImageProcessor()
    video_processor = DummyQwen3VLVideoProcessor()
    tokenizer = DummyTokenizer()
    video_token_id = DummyTokenizer.token_ids["<|video_pad|>"]
    vision_start_token_id = DummyTokenizer.token_ids["<|vision_start|>"]
    vision_end_token_id = DummyTokenizer.token_ids["<|vision_end|>"]


class DummyQwen3_5Processor(DummyQwen3VLProcessor):
    model_type = "qwen3_5"


class DummyQwen3_5MoeConfig:
    model_type = "qwen3_5_moe"


class DummyQwenGenericProcessor(DummyQwen3VLProcessor):
    config = DummyQwen3_5MoeConfig()


class DummyQwen2VLProcessor:
    image_processor = DummyQwen2VLImageProcessor()
    video_processor = DummyQwen2VLVideoProcessor()
    tokenizer = DummyTokenizer()
    image_token_id = DummyTokenizer.token_ids["<|image_pad|>"]
    video_token_id = DummyTokenizer.token_ids["<|video_pad|>"]


@dataclass
class FakePlaceholderRange:
    offset: int
    length: int
    is_embed: torch.Tensor | None = None


@dataclass
class FakeFieldElem:
    data: object
    field: object


@dataclass
class FakeFieldConfig:
    modality: str
    kind: str
    sizes: torch.Tensor | None = None

    @staticmethod
    def flat_from_sizes(modality, sizes, *args, **kwargs):
        return FakeFieldConfig(modality=modality, kind="flat", sizes=sizes)

    @staticmethod
    def batched(modality, *args, **kwargs):
        return FakeFieldConfig(modality=modality, kind="batched")


class FakeKwargsItem(dict):
    def get_data(self):
        return {key: elem.data for key, elem in self.items()}


class FakeKwargsItems(dict):
    @staticmethod
    def from_hf_inputs(inputs, config_by_key):
        elems_by_key = {}
        keys_by_modality = {}
        for key, config in config_by_key.items():
            data = inputs[key]
            if config.kind == "flat":
                sizes = [int(size) for size in config.sizes.tolist()]
                offsets = [0]
                for size in sizes:
                    offsets.append(offsets[-1] + size)
                elems = [
                    FakeFieldElem(data=data[offsets[idx] : offsets[idx + 1]], field=config) for idx in range(len(sizes))
                ]
            else:
                elems = [FakeFieldElem(data=data[idx], field=config) for idx in range(len(data))]

            elems_by_key[key] = elems
            keys_by_modality.setdefault(config.modality, set()).add(key)

        items_by_modality = {}
        for modality, keys in keys_by_modality.items():
            batch_size = len(elems_by_key[next(iter(keys))])
            items_by_modality[modality] = [
                FakeKwargsItem({key: elems_by_key[key][idx] for key in keys}) for idx in range(batch_size)
            ]
        return FakeKwargsItems(items_by_modality)


@pytest.fixture()
def fake_vllm_modules(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    inputs_module = types.ModuleType("vllm.inputs")
    multimodal_module = types.ModuleType("vllm.multimodal")
    mm_inputs_module = types.ModuleType("vllm.multimodal.inputs")

    def mm_input(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders):
        return {
            "type": "multimodal",
            "prompt_token_ids": prompt_token_ids,
            "mm_kwargs": mm_kwargs,
            "mm_hashes": mm_hashes,
            "mm_placeholders": mm_placeholders,
        }

    inputs_module.mm_input = mm_input
    mm_inputs_module.MultiModalFieldConfig = FakeFieldConfig
    mm_inputs_module.MultiModalKwargsItems = FakeKwargsItems
    mm_inputs_module.PlaceholderRange = FakePlaceholderRange

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.inputs", mm_inputs_module)


@pytest.mark.parametrize(
    "processor",
    [
        DummyQwen3VLProcessor(),
        DummyQwen3_5Processor(),
        DummyQwenGenericProcessor(),
    ],
)
def test_build_qwen3_family_direct_input_adds_timestamps_and_embed_mask(fake_vllm_modules, processor):
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
    }
    videos = [
        (
            torch.zeros(4, 3, 16, 16),
            {
                "fps": 30.0,
                "frames_indices": [0, 1, 2, 3],
                "total_num_frames": 4,
                "do_sample_frames": False,
            },
        )
    ]
    replacement = _qwen3_video_replacement(
        tokenizer=processor.tokenizer,
        timestamps=[1 / 60, 5 / 60],
        tokens_per_frame=1,
        vision_start_token_id=processor.vision_start_token_id,
        vision_end_token_id=processor.vision_end_token_id,
        video_token_id=processor.video_token_id,
    )
    prompt_ids = [11, *replacement, 12]

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        videos=videos,
    )

    assert direct_input is not None
    assert direct_input["type"] == "multimodal"
    assert direct_input["prompt_token_ids"] == prompt_ids
    assert set(direct_input["mm_kwargs"]["video"][0]) == {
        "pixel_values_videos",
        "video_grid_thw",
        "timestamps",
    }
    assert direct_input["mm_kwargs"]["video"][0]["timestamps"].data == pytest.approx([1 / 60, 5 / 60])
    assert len(direct_input["mm_hashes"]["video"]) == 1
    assert isinstance(direct_input["mm_hashes"]["video"][0], str)

    placeholder = direct_input["mm_placeholders"]["video"][0]
    assert placeholder.offset == 1
    assert placeholder.length == len(replacement)
    assert placeholder.is_embed.tolist() == [token_id == processor.video_token_id for token_id in replacement]


def test_build_qwen2_video_direct_input_keeps_second_per_grid_ts(fake_vllm_modules):
    processor = DummyQwen2VLProcessor()
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
        "second_per_grid_ts": torch.tensor([0.5]),
    }
    prompt_ids = [11, processor.video_token_id, processor.video_token_id, 12]

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        videos=[object()],
    )

    assert direct_input is not None
    assert set(direct_input["mm_kwargs"]["video"][0]) == {
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    assert direct_input["mm_kwargs"]["video"][0]["second_per_grid_ts"].data.item() == 0.5

    placeholder = direct_input["mm_placeholders"]["video"][0]
    assert placeholder.offset == 1
    assert placeholder.length == 2
    assert placeholder.is_embed is None


def test_qwen3_timestamp_count_must_match_video_count(fake_vllm_modules):
    processor = DummyQwen3VLProcessor()
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
        "timestamps": [],
    }

    with pytest.raises(ValueError, match="timestamps count"):
        build_vllm_preprocessed_multimodal_input(
            prompt_ids=[11, 12],
            processor=processor,
            model_inputs=model_inputs,
            videos=[object()],
        )


def test_build_video_embedding_direct_input(fake_vllm_modules):
    processor = DummyQwen2VLProcessor()
    model_inputs = {
        "video_embeds": torch.ones(2, 5),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
    }
    prompt_ids = [11, processor.video_token_id, processor.video_token_id, 12]

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        videos=[object()],
    )

    assert direct_input is not None
    assert set(direct_input["mm_kwargs"]["video"][0]) == {"video_embeds", "video_grid_thw"}
    assert direct_input["mm_kwargs"]["video"][0]["video_embeds"].data.shape == (2, 5)

    placeholder = direct_input["mm_placeholders"]["video"][0]
    assert placeholder.offset == 1
    assert placeholder.length == 2


def test_build_image_direct_input(fake_vllm_modules):
    processor = DummyQwen2VLProcessor()
    model_inputs = {
        "pixel_values": torch.ones(4, 3),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    prompt_ids = [
        11,
        processor.image_token_id,
        processor.image_token_id,
        processor.image_token_id,
        processor.image_token_id,
    ]

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        images=[object()],
    )

    assert direct_input is not None
    assert set(direct_input["mm_kwargs"]) == {"image"}
    assert set(direct_input["mm_kwargs"]["image"][0]) == {"pixel_values", "image_grid_thw"}
    assert direct_input["mm_placeholders"]["image"][0].offset == 1
    assert direct_input["mm_placeholders"]["image"][0].length == 4


def test_refresh_vllm_preprocessed_multimodal_prompt_ids_mutates_in_place():
    mm_kwargs_obj = object()
    payload = {
        "type": "multimodal",
        "prompt_token_ids": [1],
        "mm_kwargs": mm_kwargs_obj,
        "mm_hashes": object(),
        "mm_placeholders": object(),
    }

    refreshed = refresh_vllm_preprocessed_multimodal_prompt_ids(payload, prompt_ids=[1, 2, 3])

    assert refreshed is payload
    assert payload["prompt_token_ids"] == [1, 2, 3]
    assert payload["mm_kwargs"] is mm_kwargs_obj

    assert refresh_vllm_preprocessed_multimodal_prompt_ids(None, prompt_ids=[1]) is None


def test_real_vllm_direct_multimodal_contract_when_available():
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.inputs")
    pytest.importorskip("vllm.multimodal.inputs")

    processor = DummyQwen2VLProcessor()
    model_inputs = {
        "pixel_values": torch.ones(4, 3),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    prompt_ids = [
        11,
        processor.image_token_id,
        processor.image_token_id,
        processor.image_token_id,
        processor.image_token_id,
    ]

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=prompt_ids,
        processor=processor,
        model_inputs=model_inputs,
        images=[object()],
    )

    assert direct_input is not None
    assert direct_input["type"] == "multimodal"
    assert direct_input["prompt_token_ids"] == prompt_ids
    assert list(direct_input["mm_kwargs"].keys()) == ["image"]
    assert set(direct_input["mm_kwargs"]["image"][0]) == {"pixel_values", "image_grid_thw"}
    assert direct_input["mm_placeholders"]["image"][0].offset == 1
    assert direct_input["mm_placeholders"]["image"][0].length == 4
    assert isinstance(direct_input["mm_hashes"]["image"][0], str)


def test_real_vllm_qwen3_video_direct_multimodal_contract_when_available():
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.inputs")
    pytest.importorskip("vllm.multimodal.inputs")

    processor = DummyQwen3VLProcessor()
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
    }
    videos = [
        (
            torch.zeros(4, 3, 16, 16),
            {
                "fps": 30.0,
                "frames_indices": [0, 1, 2, 3],
                "total_num_frames": 4,
                "do_sample_frames": False,
            },
        )
    ]
    replacement = _qwen3_video_replacement(
        tokenizer=processor.tokenizer,
        timestamps=[1 / 60, 5 / 60],
        tokens_per_frame=1,
        vision_start_token_id=processor.vision_start_token_id,
        vision_end_token_id=processor.vision_end_token_id,
        video_token_id=processor.video_token_id,
    )

    direct_input = build_vllm_preprocessed_multimodal_input(
        prompt_ids=[11, *replacement, 12],
        processor=processor,
        model_inputs=model_inputs,
        videos=videos,
    )

    assert direct_input is not None
    placeholder = direct_input["mm_placeholders"]["video"][0]
    assert placeholder.offset == 1
    assert placeholder.length == len(replacement)
    assert placeholder.is_embed.tolist() == [token_id == processor.video_token_id for token_id in replacement]
