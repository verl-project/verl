# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.workers.utils.losses import sft_loss


def _dataset(rows, *, max_length=4, truncation="right", pad_mode=DatasetPadMode.NO_PADDING):
    dataset = object.__new__(MultiTurnSFTDataset)
    dataset.dataframe = pd.DataFrame(rows)
    dataset.max_length = max_length
    dataset.truncation = truncation
    dataset.pad_mode = pad_mode
    dataset.tokenizer = SimpleNamespace(pad_token_id=99)
    return dataset


@pytest.mark.parametrize("pad_mode", [DatasetPadMode.NO_PADDING, DatasetPadMode.RIGHT])
@pytest.mark.parametrize(
    ("truncation", "expected_ids", "expected_mask"),
    [("left", [12, 13, 14, 15], [0, 0, 1, 1]), ("right", [10, 11, 12, 13], [0, 1, 1, 0])],
)
def test_truncation_keeps_original_token_mask_alignment(pad_mode, truncation, expected_ids, expected_mask):
    dataset = _dataset(
        [{"input_ids": list(range(10, 16)), "loss_mask": [0, 1, 1, 0, 1, 1], "sample": 7}],
        truncation=truncation,
        pad_mode=pad_mode,
    )
    sample = dataset[0]
    assert sample["input_ids"].tolist() == expected_ids
    assert sample["loss_mask"].tolist() == expected_mask
    assert sample["position_ids"].tolist() == [0, 1, 2, 3]
    assert sample["sample"] == 7
    if pad_mode == DatasetPadMode.RIGHT:
        assert sample["attention_mask"].tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("truncation", ["left", "right"])
def test_truncation_without_explicit_mask_excludes_first_retained_token(truncation):
    sample = _dataset([{"input_ids": list(range(6))}], truncation=truncation)[0]
    assert sample["loss_mask"].tolist() == [0, 1, 1, 1]


def test_short_row_right_padding_excludes_unpredictable_and_padding_tokens():
    sample = _dataset([{"input_ids": [10, 11], "loss_mask": [1, 1]}], pad_mode=DatasetPadMode.RIGHT)[0]
    assert sample["input_ids"].tolist() == [10, 11, 99, 99]
    assert sample["attention_mask"].tolist() == [1, 1, 0, 0]
    assert sample["loss_mask"].tolist() == [0, 1, 0, 0]


def test_masks_are_validated_before_truncation():
    dataset = _dataset([{"input_ids": list(range(6)), "loss_mask": [0, 1, 1, 1]}])
    with pytest.raises(ValueError, match="4 entries for 6 tokens"):
        dataset[0]


def test_teacher_rows_are_validated_against_original_tokens():
    row = {
        "input_ids": list(range(6)),
        "teacher_ids": [[1, 2]] * 6,
        "teacher_logprobs": [[-0.2, -1.7]] * 6,
    }
    assert _dataset([row], truncation="left")[0]["input_ids"].tolist() == [2, 3, 4, 5]
    row["teacher_ids"] = row["teacher_ids"][:4]
    row["teacher_logprobs"] = row["teacher_logprobs"][:4]
    with pytest.raises(ValueError, match="align with the 6 tokens"):
        _dataset([row])[0]


def test_error_truncation_still_rejects_long_input():
    with pytest.raises(ValueError, match="6 tokens > max_length=4"):
        _dataset([{"input_ids": list(range(6))}], truncation="error")[0]


def test_all_one_input_masks_do_not_train_on_packed_sample_boundaries():
    dataset = _dataset([{"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}, {"input_ids": [4, 5], "loss_mask": [1, 1]}])
    masks = torch.nested.nested_tensor([dataset[index]["loss_mask"] for index in range(2)], layout=torch.jagged)
    log_probs = torch.tensor([-1.0, -2.0, -100.0, -3.0, -200.0], requires_grad=True)
    nested_log_probs = torch.nested.nested_tensor_from_jagged(log_probs, masks.offsets())
    data = TensorDict({"loss_mask": masks}, batch_size=[2])
    data["dp_size"] = NonTensorData(1)
    data["batch_num_tokens"] = NonTensorData(masks.values().sum())
    loss, _ = sft_loss(None, {"log_probs": nested_log_probs}, data)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert log_probs.grad.tolist() == pytest.approx([-1 / 3, -1 / 3, 0, -1 / 3, 0])
