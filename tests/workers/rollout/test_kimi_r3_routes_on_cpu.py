# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import SYNTHETIC_PADDING_SEQ_LEN, construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.schemas import MODEL_ROW_ROUTE_SEMANTICS
from verl.workers.utils.padding import left_right_2_no_padding


@pytest.mark.parametrize("expert_ids", [[0, 7], [256, 895]])
@pytest.mark.parametrize("per_sample_tag", [False, True])
def test_model_row_padding_preserves_int32_routes(expert_ids, per_sample_tag):
    ids = torch.tensor(expert_ids, dtype=torch.int32).reshape(1, 1, 1, 2).expand(2, 5, 1, 2)
    data = TensorDict(
        dict(
            input_ids=torch.ones(2, 4, dtype=torch.long),
            attention_mask=torch.ones(2, 4),
            response_mask=torch.ones(2, 2),
            position_ids=torch.arange(4).expand(2, 4),
            routed_experts=ids,
        ),
        batch_size=[2],
    )
    tag = [MODEL_ROW_ROUTE_SEMANTICS] * 2 if per_sample_tag else MODEL_ROW_ROUTE_SEMANTICS
    tu.assign_non_tensor_data(data, "routed_experts_transport_semantics", tag)
    result = left_right_2_no_padding(data)
    torch.testing.assert_close(result["routed_experts"].values(), ids.flatten(0, 1))
    assert result["routed_experts"].dtype == torch.int32
    assert result["routed_experts"].offsets().tolist() == [0, 5, 10]


def test_synthetic_padding_keeps_model_axis_and_valid_source_ids():
    source = {
        "routed_experts": torch.tensor([[[256, 895]], [[1, 2]]], dtype=torch.int32),
        "routed_experts_transport_semantics": MODEL_ROW_ROUTE_SEMANTICS,
    }
    sample, tag = construct_minimal_padding_template(source, {}, eos_token_id=7)
    assert tag["is_padding"]
    assert sample["routed_experts_transport_semantics"] == MODEL_ROW_ROUTE_SEMANTICS
    assert sample["routed_experts"].shape == (SYNTHETIC_PADDING_SEQ_LEN - 1, 1, 2)
    torch.testing.assert_close(sample["routed_experts"][0], source["routed_experts"][0])
    assert not sample["loss_mask"].any()
