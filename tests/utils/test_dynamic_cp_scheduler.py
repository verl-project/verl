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

import pytest
import torch
from tensordict import TensorDict

import verl.utils.dynamic_cp_scheduler as dcp_module
from verl.utils import tensordict_utils as tu
from verl.utils.dynamic_cp_scheduler import (
    DCP_GROUP_LEADER,
    DCP_LOCAL_NUM_TOKENS,
    DCP_PADDING_MASK,
    DCP_SAMPLE_IDS,
    DynamicCPScheduler,
    _local_padding_mask,
    postprocess_dynamic_cp_batch,
)


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _nested(parts):
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def _batch(lengths):
    return TensorDict(
        {
            "input_ids": _nested([torch.arange(length) for length in lengths]),
            "sample": torch.arange(len(lengths)),
        },
        batch_size=[len(lengths)],
    )


def test_schedule_adapts_mcore_assignments(monkeypatch):
    assignments = [[[0], [0], [1], [2]]]

    class FakeMCoreScheduler:
        kwargs = None

        def __init__(self, **kwargs):
            FakeMCoreScheduler.kwargs = kwargs

        def get_groups_and_subsamples(self, samples):
            assert samples == [(0, 7), (1, 3), (2, 5)]
            return assignments

    monkeypatch.setattr(dcp_module, "get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMCoreScheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=4, dp_size=2, cp_size=2)

    expected = [
        ([0], 2, True, 3, [False, False, False, True]),
        ([0], 2, False, 4, [False, False, False, False]),
        ([1], 1, True, 3, [False, False, False]),
        ([2], 1, True, 5, [False, False, False, False, False]),
    ]
    for rank, (sample_ids, cp_size, leader, local_tokens, padding_mask) in enumerate(expected):
        micro_batches = scheduler.schedule(_batch([7, 3, 5]), _FakeGroup(4, rank), tp_size=1)
        assert len(micro_batches) == 1
        micro_batch = micro_batches[0]
        assert micro_batch["sample"].tolist() == sample_ids
        assert tu.get_non_tensor_data(micro_batch, "local_cp_size", None) == cp_size
        assert tu.get_non_tensor_data(micro_batch, DCP_SAMPLE_IDS, None) == sample_ids
        assert tu.get_non_tensor_data(micro_batch, DCP_GROUP_LEADER, None) is leader
        assert tu.get_non_tensor_data(micro_batch, DCP_LOCAL_NUM_TOKENS, None) == local_tokens
        assert tu.get_non_tensor_data(micro_batch, DCP_PADDING_MASK, None).tolist() == padding_mask

    assert FakeMCoreScheduler.kwargs == {
        "max_seqlen_per_dp_cp_rank": 4,
        "cp_size": 2,
        "dp_size": 2,
        "microbatch_group_size_per_vp_stage": None,
    }


@pytest.mark.parametrize("cp_size,tp_size", [(2, 1), (4, 2)])
def test_local_token_count_partitions_real_tokens(cp_size, tp_size):
    masks = [
        _local_padding_mask(
            [7, 3, 17],
            cp_size=cp_size,
            cp_rank=rank,
            tp_size=tp_size,
        )
        for rank in range(cp_size)
    ]
    counts = [mask.numel() - int(mask.sum().item()) for mask in masks]
    assert sum(counts) == 27


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seqlen_per_dp_cp_rank": 0, "dp_size": 1, "cp_size": 1},
        {"max_seqlen_per_dp_cp_rank": 4, "dp_size": True, "cp_size": 1},
        {"max_seqlen_per_dp_cp_rank": 4, "dp_size": 1, "cp_size": -1},
    ],
)
def test_scheduler_rejects_invalid_topology(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        DynamicCPScheduler(**kwargs)


def test_scheduler_rejects_unaligned_mcore_group(monkeypatch):
    class FakeMCoreScheduler:
        def __init__(self, **kwargs):
            pass

        def get_groups_and_subsamples(self, samples):
            return [[[0], [1], [1], [2]]]

    monkeypatch.setattr(dcp_module, "get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMCoreScheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=4, dp_size=2, cp_size=2)
    with pytest.raises(RuntimeError, match="not aligned"):
        scheduler.schedule(_batch([3, 5, 7]), _FakeGroup(4, rank=1), tp_size=1)


def test_postprocess_restores_original_sample_order(monkeypatch):
    local_outputs = [
        {
            DCP_GROUP_LEADER: True,
            DCP_SAMPLE_IDS: [1],
            "model_output": {"log_probs": _nested([torch.tensor([10.0, 11.0])])},
            "loss": 2.0,
            "metrics": {"metric": 1.0},
        }
    ]
    remote_records = {
        1: [
            {
                "sample_ids": [0],
                "model_output": {"log_probs": [torch.tensor([0.0])]},
                "loss": 1.0,
                "metrics": {"metric": 0.0},
            }
        ],
        2: [
            {
                "sample_ids": [2],
                "model_output": {"log_probs": [torch.tensor([20.0, 21.0, 22.0])]},
                "loss": 3.0,
                "metrics": {"metric": 2.0},
            }
        ],
        3: [],
    }

    def fake_all_gather_object(output, local, group):
        output[0] = local
        for rank, records in remote_records.items():
            output[rank] = records

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    result = postprocess_dynamic_cp_batch(local_outputs, batch_size=3, dcp_group=_FakeGroup(4))

    assert [part.tolist() for part in result["model_output"]["log_probs"].unbind()] == [
        [0.0],
        [10.0, 11.0],
        [20.0, 21.0, 22.0],
    ]
    assert result["loss"] == [1.0, 2.0, 3.0]
    assert result["metrics"] == {"metric": [0.0, 1.0, 2.0]}
