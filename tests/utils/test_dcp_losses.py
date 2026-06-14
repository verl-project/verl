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

import torch
from tensordict import TensorDict

from verl.workers.config import CriticConfig
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import no_padding_2_padding


def _nested(parts):
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def test_value_loss_masks_nonlocal_dcp_tokens():
    data = TensorDict(
        {
            "prompts": _nested([torch.tensor([1, 2])]),
            "responses": _nested([torch.tensor([3, 4, 5])]),
            "values": torch.zeros(1, 3),
            "returns": torch.zeros(1, 3),
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "values": _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _loss, metrics = value_loss(CriticConfig(strategy="megatron", ppo_micro_batch_size_per_gpu=1), model_output, data)

    assert metrics["critic/vpred_mean"] == torch.tensor(505.0).item()


def test_dcp_no_padding_uses_response_mask_for_response_span():
    data = TensorDict(
        {
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, False, False, True, False]]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])])

    padded = no_padding_2_padding(model_output, data)

    torch.testing.assert_close(padded, torch.tensor([[10.0, 100.0, 1000.0]]))
