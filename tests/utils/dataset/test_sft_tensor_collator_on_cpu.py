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

import pytest
import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.model import extract_multi_modal_inputs


@pytest.mark.parametrize("image_first", [False, True])
def test_sft_collator_preserves_optional_multimodal_input_and_sample_order(image_first):
    text = {"input_ids": torch.tensor([1, 2]), "loss_mask": torch.tensor([0, 1])}
    pixels = torch.arange(6).reshape(2, 3)
    image = {
        "input_ids": torch.tensor([3, 4, 5]),
        "loss_mask": torch.tensor([0, 0, 1]),
        "multi_modal_inputs": {"pixel_values": pixels},
    }
    samples = [image, text] if image_first else [text, image]

    batch = SFTTensorCollator(DatasetPadMode.NO_PADDING)(samples)

    for key in ("input_ids", "loss_mask"):
        assert all(
            torch.equal(actual, sample[key]) for actual, sample in zip(batch[key].unbind(), samples, strict=True)
        )
    text_index = 1 if image_first else 0
    assert batch["multi_modal_inputs"][text_index].data is None
    assert torch.equal(extract_multi_modal_inputs(batch["multi_modal_inputs"])["pixel_values"], pixels)


@pytest.mark.parametrize("missing_index", [0, 1])
def test_sft_collator_does_not_treat_missing_tensor_fields_as_optional(missing_index):
    samples = [
        {"input_ids": torch.tensor([1, 2]), "loss_mask": torch.tensor([0, 1])},
        {"input_ids": torch.tensor([3, 4, 5]), "loss_mask": torch.tensor([0, 0, 1])},
    ]
    del samples[missing_index]["input_ids"]

    with pytest.raises(KeyError, match="input_ids"):
        SFTTensorCollator(DatasetPadMode.NO_PADDING)(samples)


@pytest.mark.parametrize(
    ("sample_order", "indices"),
    [
        ([0, 0], None),
        ([0, 1], None),
        ([1, 0], None),
        ([1, 1], None),
        ([1, 2], None),
        ([2, 1], None),
        ([2, 2], None),
        ([0, 1, 2], [2, 0, 1]),
        ([0, 1, 2], [0]),
        ([0, 1, 2], [2]),
    ],
)
def test_sft_image_crop_lists_keep_sample_crop_and_image_size_order(sample_order, indices):
    samples = [
        {"input_ids": torch.tensor([1, 2]), "loss_mask": torch.tensor([0, 1])},
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "loss_mask": torch.tensor([0, 0, 1]),
            "multi_modal_inputs": {
                "pixel_values": [torch.full((3, 4, 4), 10), torch.full((3, 4, 4), 11)],
                "image_sizes": torch.tensor([[4, 4]]),
            },
        },
        {
            "input_ids": torch.tensor([6, 7, 8, 9]),
            "loss_mask": torch.tensor([0, 0, 0, 1]),
            "multi_modal_inputs": {
                "pixel_values": [torch.full((3, 4, 4), value) for value in range(20, 25)],
                "image_sizes": torch.tensor([[4, 4], [4, 8]]),
            },
        },
    ]
    ordered_samples = [samples[index] for index in sample_order]
    batch = SFTTensorCollator(DatasetPadMode.NO_PADDING)(ordered_samples)

    merged = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []), indices=indices)

    selected = ordered_samples if indices is None else [ordered_samples[index] for index in indices]
    expected_mm = [sample["multi_modal_inputs"] for sample in selected if "multi_modal_inputs" in sample]
    if not expected_mm:
        assert merged == {}
        return
    expected_crops = [crop for mm in expected_mm for crop in mm["pixel_values"]]
    assert isinstance(merged["pixel_values"], list)
    assert all(actual is expected for actual, expected in zip(merged["pixel_values"], expected_crops, strict=True))
    assert torch.equal(merged["image_sizes"], torch.cat([mm["image_sizes"] for mm in expected_mm]))


def test_sft_image_bound_keeps_existing_per_sample_pixel_lists():
    first = [torch.zeros(3, 4, 4)]
    second = [torch.ones(3, 4, 4)]
    merged = extract_multi_modal_inputs(
        [
            {"pixel_values": first, "image_bound": torch.tensor([[1, 2]])},
            {"pixel_values": second, "image_bound": torch.tensor([[3, 4]])},
        ]
    )

    assert merged["pixel_values"][0] is first
    assert merged["pixel_values"][1] is second


@pytest.mark.parametrize("list_first", [False, True])
def test_sft_image_crops_do_not_silently_flatten_mixed_tensor_and_list_values(list_first):
    tensor = {"pixel_values": torch.zeros(2, 3, 4, 4)}
    crops = {"pixel_values": [torch.ones(3, 4, 4)]}

    with pytest.raises(TypeError, match="expected Tensor"):
        extract_multi_modal_inputs([crops, tensor] if list_first else [tensor, crops])
