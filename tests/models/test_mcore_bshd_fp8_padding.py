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

from unittest.mock import patch

import torch

from verl.models.mcore.util import postprocess_bshd, preprocess_bshd


def test_preprocess_bshd_fp8_padding_aligns_local_tokens_and_roundtrips():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 0, 0],
            [20, 21, 22, 23, 24],
            [30, 31, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand_as(input_ids)

    with (
        patch("verl.models.mcore.util.mpu.get_context_parallel_world_size", return_value=1),
        patch("verl.models.mcore.util.mpu.get_tensor_model_parallel_world_size", return_value=2),
    ):
        input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd(
            input_ids,
            attention_mask,
            position_ids,
            sequence_parallel=True,
            pre_process=True,
            use_fp8_padding=True,
        )

    assert input_ids_bshd.shape == (3, 256)
    assert position_ids_bshd.shape == (3, 256)
    assert attention_mask_bshd.shape == (3, 256)
    assert (input_ids_bshd.shape[0] * input_ids_bshd.shape[1] // 2) % 128 == 0

    restored = postprocess_bshd(input_ids_bshd, attention_mask_bshd, attention_mask, input_ids.shape[1])
    torch.testing.assert_close(restored, input_ids)
