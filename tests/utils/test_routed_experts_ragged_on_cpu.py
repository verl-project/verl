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
"""Ragged routed-experts wire format: pack, nested round-trip, dense adapter, DataProto ops."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.padding_utils import build_padding_routed_experts
from verl.utils.attention_utils import unpad_input
from verl.utils.routed_experts import (
    ROUTER_REPLAY_UNRECORDED,
    has_rollout_routed_experts,
    pack_padded_routed_experts,
    ragged_routed_experts_to_nested,
    rollout_and_actor_routed_experts_conflict,
    unrecorded_fill_value,
)
from verl.workers.utils.padding import left_right_2_no_padding

LAYERS, TOPK = 2, 3
SENTINEL = ROUTER_REPLAY_UNRECORDED


def _padded_row(attended: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scatter ``attended`` into a config-length row, filling pad with the sentinel."""
    seq = int(mask.numel())
    row = torch.full((seq, LAYERS, TOPK), SENTINEL, dtype=torch.int16)
    row[mask.bool()] = attended
    return row


def test_unrecorded_fill_value_does_not_wrap_unsigned():
    assert unrecorded_fill_value(torch.int16) == SENTINEL
    assert unrecorded_fill_value(torch.uint8) == 0


def test_pack_strips_padding_and_keeps_final_token_sentinel():
    # Left-padded prompt + right-padded response. The last attended token is the
    # sampled token that never re-entered MoE, so it stays the sentinel.
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    row0 = torch.tensor(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[12, 13, 14], [15, 16, 17]],
            [[SENTINEL] * TOPK, [SENTINEL] * TOPK],
        ],
        dtype=torch.int16,
    )
    row1 = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[SENTINEL] * TOPK, [SENTINEL] * TOPK],
        ],
        dtype=torch.int16,
    )
    padded = [_padded_row(row0, mask[0]).unsqueeze(0), _padded_row(row1, mask[1])]
    packed = pack_padded_routed_experts(padded, mask)

    assert packed.dtype == object
    assert packed.shape == (2,)
    np.testing.assert_array_equal(packed[0], row0.numpy())
    np.testing.assert_array_equal(packed[1], row1.numpy())
    assert int(packed[0][-1, 0, 0]) == SENTINEL
    assert int(packed[1][-1, 0, 0]) == SENTINEL


def test_pack_rejects_partial_payload():
    mask = torch.ones(2, 4, dtype=torch.int64)
    row = torch.zeros(1, 4, LAYERS, TOPK, dtype=torch.int16)
    with pytest.raises(ValueError, match="partial routing payload"):
        pack_padded_routed_experts([row, None], mask)


def test_pack_rejects_seq_mismatch():
    mask = torch.ones(1, 5, dtype=torch.int64)
    row = torch.zeros(1, 3, LAYERS, TOPK, dtype=torch.int16)
    with pytest.raises(ValueError, match="expected"):
        pack_padded_routed_experts([row], mask)


def test_ragged_to_nested_matches_unpad_offsets():
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.int64)
    input_ids = torch.arange(10, dtype=torch.int64).reshape(2, 5)
    rows = [
        torch.arange(3 * LAYERS * TOPK, dtype=torch.int16).reshape(3, LAYERS, TOPK).numpy(),
        torch.arange(100, 100 + 2 * LAYERS * TOPK, dtype=torch.int16).reshape(2, LAYERS, TOPK).numpy(),
    ]
    packed = np.empty(2, dtype=object)
    packed[:] = rows
    _, _, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), mask)
    nested = ragged_routed_experts_to_nested(packed, cu_seqlens)

    assert nested.is_nested
    torch.testing.assert_close(nested.offsets(), cu_seqlens.to(dtype=nested.offsets().dtype))
    torch.testing.assert_close(nested.values(), torch.from_numpy(np.concatenate(rows, axis=0)))


def test_ragged_to_nested_rejects_row_count_drift():
    mask = torch.ones(2, 3, dtype=torch.int64)
    input_ids = torch.ones(2, 3, dtype=torch.int64)
    packed = np.empty(2, dtype=object)
    packed[0] = np.zeros((3, LAYERS, TOPK), dtype=np.int16)
    packed[1] = np.zeros((2, LAYERS, TOPK), dtype=np.int16)
    _, _, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), mask)
    with pytest.raises(ValueError, match="cu_seqlens"):
        ragged_routed_experts_to_nested(packed, cu_seqlens)


def test_dense_adapter_still_unpads():
    batch_size, seq_len = 2, 6
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
        dtype=torch.int64,
    )
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64).reshape(batch_size, seq_len)
    response_mask = torch.zeros(batch_size, 2, dtype=torch.int64)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    routed = torch.arange(batch_size * seq_len * LAYERS * TOPK, dtype=torch.int16).reshape(
        batch_size, seq_len, LAYERS, TOPK
    )
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "routed_experts": routed,
        },
        batch_size=[batch_size],
    )
    converted = left_right_2_no_padding(data)
    assert converted["routed_experts"].is_nested
    expected = torch.cat([routed[0, attention_mask[0].bool()], routed[1, attention_mask[1].bool()]], dim=0)
    torch.testing.assert_close(converted["routed_experts"].values(), expected)


def test_ragged_adapter_round_trips_through_left_right_2_no_padding():
    batch_size, seq_len = 2, 6
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
        dtype=torch.int64,
    )
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64).reshape(batch_size, seq_len)
    response_mask = torch.zeros(batch_size, 2, dtype=torch.int64)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    rows = [
        torch.arange(3 * LAYERS * TOPK, dtype=torch.int16).reshape(3, LAYERS, TOPK).numpy(),
        torch.arange(50, 50 + 3 * LAYERS * TOPK, dtype=torch.int16).reshape(3, LAYERS, TOPK).numpy(),
    ]
    packed = np.empty(batch_size, dtype=object)
    packed[:] = rows
    proto = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=[batch_size],
        ),
        non_tensor_batch={"routed_experts": packed},
    )
    converted = left_right_2_no_padding(proto.to_tensordict())
    assert converted["routed_experts"].is_nested
    torch.testing.assert_close(
        converted["routed_experts"].values(),
        torch.from_numpy(np.concatenate(rows, axis=0)),
    )


def test_dataproto_concat_reorder_chunk_keep_per_row_blocks():
    def _one(marker: int, n: int) -> DataProto:
        row = np.full((n, LAYERS, TOPK), marker, dtype=np.int16)
        packed = np.empty(1, dtype=object)
        packed[0] = row
        return DataProto(
            batch=TensorDict({"input_ids": torch.zeros(1, 2, dtype=torch.int64)}, batch_size=[1]),
            non_tensor_batch={"routed_experts": packed},
        )

    a, b, c = _one(1, 2), _one(2, 5), _one(3, 1)
    cat = DataProto.concat([a, b, c])
    assert cat.non_tensor_batch["routed_experts"].shape == (3,)
    assert int(cat.non_tensor_batch["routed_experts"][1][0, 0, 0]) == 2
    assert cat.non_tensor_batch["routed_experts"][1].shape[0] == 5

    cat.reorder(torch.tensor([2, 0, 1]))
    assert [int(row[0, 0, 0]) for row in cat.non_tensor_batch["routed_experts"]] == [3, 1, 2]

    chunks = cat.chunk(chunks=3)
    assert [int(c.non_tensor_batch["routed_experts"][0][0, 0, 0]) for c in chunks] == [3, 1, 2]


def test_build_padding_routed_experts_mirrors_numpy_and_tensor():
    src_np = np.zeros((4, LAYERS, TOPK), dtype=np.int16)
    pad_np = build_padding_routed_experts(src_np, 7)
    assert isinstance(pad_np, np.ndarray)
    assert pad_np.shape == (7, LAYERS, TOPK)
    assert bool((pad_np == SENTINEL).all())

    src_t = torch.zeros(4, LAYERS, TOPK, dtype=torch.int16)
    pad_t = build_padding_routed_experts(src_t, 7)
    assert pad_t.shape == (7, LAYERS, TOPK)
    assert bool((pad_t == SENTINEL).all())


def test_r2_r3_conflict_sees_non_tensor_rollout_payload():
    rollout = DataProto(
        batch=TensorDict({"input_ids": torch.zeros(1, 2, dtype=torch.int64)}, batch_size=[1]),
        non_tensor_batch={"routed_experts": np.empty(1, dtype=object)},
    )
    actor = DataProto(
        batch=TensorDict({"routed_experts": torch.zeros(1, 2, 1, 1, dtype=torch.int16)}, batch_size=[1]),
    )
    assert has_rollout_routed_experts(rollout)
    assert rollout_and_actor_routed_experts_conflict(rollout, actor)

    empty = DataProto(batch=TensorDict({"input_ids": torch.zeros(1, 2, dtype=torch.int64)}, batch_size=[1]))
    assert not rollout_and_actor_routed_experts_conflict(empty, actor)
