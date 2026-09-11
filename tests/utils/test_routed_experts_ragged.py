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
"""CUDA unpad of ragged vs dense routed_experts must agree."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.routed_experts import ROUTER_REPLAY_UNRECORDED, pack_padded_routed_experts
from verl.workers.utils.padding import left_right_2_no_padding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

LAYERS, TOPK = 2, 3
SENTINEL = ROUTER_REPLAY_UNRECORDED


def test_cuda_ragged_unpad_matches_dense_and_keeps_sentinel():
    device = torch.device("cuda")
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
        dtype=torch.int64,
        device=device,
    )
    batch_size, seq_len = attention_mask.shape
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64, device=device).reshape(batch_size, seq_len)
    response_mask = torch.zeros(batch_size, 2, dtype=torch.int64, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    dense = torch.arange(batch_size * seq_len * LAYERS * TOPK, dtype=torch.int16, device=device).reshape(
        batch_size, seq_len, LAYERS, TOPK
    )
    # Last attended token of each row is unrecorded (the sampled token never re-entered MoE).
    for i in range(batch_size):
        attended_idx = attention_mask[i].nonzero(as_tuple=False).flatten()
        dense[i, attended_idx[-1]] = SENTINEL

    packed = pack_padded_routed_experts(list(dense.detach().cpu()), attention_mask.cpu())

    dense_td = TensorDict(
        {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "response_mask": response_mask.clone(),
            "position_ids": position_ids.clone(),
            "routed_experts": dense.clone(),
        },
        batch_size=[batch_size],
    )
    ragged_proto = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "response_mask": response_mask.clone(),
                "position_ids": position_ids.clone(),
            },
            batch_size=[batch_size],
        ),
        non_tensor_batch={"routed_experts": packed},
    )

    dense_out = left_right_2_no_padding(dense_td)
    ragged_out = left_right_2_no_padding(ragged_proto.to_tensordict())

    assert dense_out["routed_experts"].is_nested
    assert ragged_out["routed_experts"].is_nested
    assert dense_out["routed_experts"].values().device.type == "cuda"
    assert ragged_out["routed_experts"].values().device.type == "cuda"
    torch.testing.assert_close(
        ragged_out["routed_experts"].values(),
        dense_out["routed_experts"].values(),
    )
    values = dense_out["routed_experts"].values()
    offsets = dense_out["routed_experts"].offsets()
    for i in range(batch_size):
        last = values[int(offsets[i + 1].item()) - 1]
        assert bool((last == SENTINEL).all())
