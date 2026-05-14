# Copyright 2026 Tencent Ltd. and/or its affiliates
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
import torch
from PIL import Image
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.fully_async_policy.image_refs import (
    IMAGE_BANK_REF_KEY,
    MULTI_MODAL_DATA_KEY,
    MULTI_MODAL_INPUTS_KEY,
    MULTI_MODAL_REFS_KEY,
    attach_image_bank_ref,
    attach_image_refs_to_dataproto,
)
from verl.experimental.fully_async_policy.intermediate_trajectory_utils import _compute_position_ids
from verl.utils.model import compute_vlm_position_ids, resolve_multi_modal_refs


class DummyImageProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors="pt"):
        del images, return_tensors
        self.calls += 1
        return {
            "pixel_values": torch.full((1, 3, 2, 2), float(self.calls)),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class DummyProcessor:
    image_token_id = 42
    video_token_id = 43

    def __init__(self):
        self.image_processor = DummyImageProcessor()

    def get_rope_index(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len)
        return position_ids, None


class StrictDummyProcessor(DummyProcessor):
    def get_rope_index(self, input_ids, attention_mask, **kwargs):
        if kwargs.get("image_grid_thw") is None and torch.any(input_ids == self.image_token_id):
            raise TypeError("image_grid_thw is required when image tokens are present")
        return super().get_rope_index(input_ids, attention_mask, **kwargs)


class TextAxisDummyProcessor(StrictDummyProcessor):
    position_ids_need_text_axis = True


def _object_array(values):
    arr = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        arr[idx] = value
    return arr


def test_attach_image_refs_builds_processed_bank_and_dedupes_images():
    image = Image.new("RGB", (16, 16), color="red")
    processor = DummyProcessor()
    batch = TensorDict(
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
        },
        batch_size=2,
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            MULTI_MODAL_DATA_KEY: _object_array([{"images": [image]}, {"images": [image.copy()]}]),
            "intermediate_trajectories": _object_array(
                [
                    [
                        {
                            "prompt_ids": [1],
                            "response_ids": [2],
                            "response_mask": [1],
                            MULTI_MODAL_DATA_KEY: {"images": [image.copy()]},
                        }
                    ],
                    [],
                ]
            ),
            MULTI_MODAL_INPUTS_KEY: _object_array([{"pixel_values": torch.zeros(1)}, {}]),
            "raw_prompt": _object_array([[{"role": "user", "content": image}], [{"role": "user", "content": "text"}]]),
        },
    )

    output, image_bank, stats = attach_image_refs_to_dataproto(data, processor=processor, sample_id="sample-1")

    assert processor.image_processor.calls == 1
    assert len(image_bank) == 1
    assert stats["unique_images"] == 1
    assert stats["row_image_refs"] == 3
    assert stats["processed_bytes"] > 0
    assert MULTI_MODAL_DATA_KEY not in output.non_tensor_batch
    assert MULTI_MODAL_INPUTS_KEY not in output.non_tensor_batch
    assert output.non_tensor_batch[MULTI_MODAL_REFS_KEY].shape == (2,)
    assert output.non_tensor_batch[MULTI_MODAL_REFS_KEY].dtype == object
    assert (
        output.non_tensor_batch[MULTI_MODAL_REFS_KEY][0]["image_ids"]
        == output.non_tensor_batch[MULTI_MODAL_REFS_KEY][1]["image_ids"]
    )
    assert (
        output.non_tensor_batch["intermediate_trajectories"][0][0][MULTI_MODAL_REFS_KEY]["image_ids"]
        == output.non_tensor_batch[MULTI_MODAL_REFS_KEY][0]["image_ids"]
    )
    assert "image_ref_omitted" in output.non_tensor_batch["raw_prompt"][0][0]["content"]

    payload = next(iter(image_bank.values()))
    assert set(payload["inputs"]) >= {"pixel_values", "image_grid_thw", "images_seqlens"}
    assert payload["inputs"]["pixel_values"].shape == (1, 3, 2, 2)

    output = attach_image_bank_ref(output, "bank-ref")
    assert output.non_tensor_batch[IMAGE_BANK_REF_KEY].shape == (2,)
    assert output.non_tensor_batch[IMAGE_BANK_REF_KEY].tolist() == ["bank-ref", "bank-ref"]


def test_attach_image_bank_ref_adds_text_only_empty_refs():
    batch = TensorDict({"input_ids": torch.ones(1, 2, dtype=torch.long)}, batch_size=1)
    data = DataProto(batch=batch, non_tensor_batch={})

    output = attach_image_bank_ref(data, None)

    assert output.non_tensor_batch[IMAGE_BANK_REF_KEY].tolist() == [None]
    assert output.non_tensor_batch[MULTI_MODAL_REFS_KEY][0] == {"image_ids": [], "video_ids": []}


def test_resolve_multi_modal_refs_uses_processed_bank_without_ray_get():
    image_bank = {
        "sha1:image": {
            "inputs": {
                "pixel_values": torch.ones(1, 3, 2, 2),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
                "images_seqlens": torch.tensor([4], dtype=torch.long),
            }
        }
    }
    micro_batch = {
        "input_ids": torch.tensor([[1, 42, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "position_ids": torch.arange(3).unsqueeze(0),
        MULTI_MODAL_REFS_KEY: _object_array([{"image_ids": ["sha1:image"], "video_ids": []}]),
        IMAGE_BANK_REF_KEY: _object_array(["bank-ref"]),
    }

    resolved = resolve_multi_modal_refs(
        micro_batch,
        tokenizer=None,
        processor=DummyProcessor(),
        bank_cache={"bank-ref": image_bank},
    )

    assert resolved["pixel_values"].shape == (1, 3, 2, 2)
    assert resolved["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert micro_batch["position_ids"].dim() == 3
    assert micro_batch["position_ids"][0].shape == (3, 3)
    assert micro_batch["position_ids"].values().shape[0] == 3


def test_resolve_multi_modal_refs_uses_placeholder_position_ids_without_image_grid():
    micro_batch = {
        "input_ids": torch.tensor([[1, 42, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "position_ids": torch.arange(3).unsqueeze(0),
        MULTI_MODAL_REFS_KEY: _object_array([{"image_ids": [], "video_ids": []}]),
        IMAGE_BANK_REF_KEY: _object_array([None]),
    }

    resolved = resolve_multi_modal_refs(
        micro_batch,
        tokenizer=None,
        processor=StrictDummyProcessor(),
        bank_cache={},
    )

    assert resolved == {}
    assert micro_batch["position_ids"].dim() == 3
    assert micro_batch["position_ids"][0].shape == (3, 3)


def test_resolve_multi_modal_refs_handles_unpadded_input_with_padded_attention_mask():
    image_bank = {
        "sha1:image": {
            "inputs": {
                "pixel_values": torch.ones(1, 3, 2, 2),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
                "images_seqlens": torch.tensor([4], dtype=torch.long),
            }
        }
    }
    micro_batch = {
        "input_ids": torch.nested.as_nested_tensor([torch.tensor([1, 42, 2])], layout=torch.jagged),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        "position_ids": torch.arange(3).unsqueeze(0),
        MULTI_MODAL_REFS_KEY: _object_array([{"image_ids": ["sha1:image"], "video_ids": []}]),
        IMAGE_BANK_REF_KEY: _object_array(["bank-ref"]),
    }

    resolved = resolve_multi_modal_refs(
        micro_batch,
        tokenizer=None,
        processor=StrictDummyProcessor(),
        bank_cache={"bank-ref": image_bank},
    )

    assert resolved["pixel_values"].shape == (1, 3, 2, 2)
    assert micro_batch["position_ids"][0].shape == (3, 3)
    assert micro_batch["position_ids"].values().shape[0] == 3


def test_compute_vlm_position_ids_adds_text_axis_for_legacy_vl_models():
    input_ids = torch.tensor([[1, 42, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = compute_vlm_position_ids(TextAxisDummyProcessor(), input_ids, attention_mask, {})

    assert position_ids.shape == (1, 4, 3)


def test_agent_loop_position_ids_skip_rope_without_image_grid():
    worker = object.__new__(AgentLoopWorker)
    worker.processor = StrictDummyProcessor()
    input_ids = torch.tensor([[1, 42, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    position_ids = worker._compute_position_ids(input_ids, attention_mask, {})

    assert position_ids.shape == (1, 3, 3)


def test_intermediate_position_ids_skip_rope_without_image_grid():
    input_ids = torch.tensor([[1, 42, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    position_ids = _compute_position_ids(StrictDummyProcessor(), input_ids, attention_mask, {})

    assert position_ids.shape == (1, 3, 3)
