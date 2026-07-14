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
from verl.utils.dynamic_cp_scheduler import (
    DynamicCPScheduler,
    _build_reverse_routing_plans,
    _classify_routed_batch_fields,
    _get_response_lengths,
    _reconstruct_compact_sample,
    _reroute_samples,
    broadcast_dcp_metadata_to_pp,
    reverse_route_outputs,
)
from verl.workers.engine.utils import prepare_micro_batches


def test_dcp_scheduler_delegates_grouping_to_megatron(monkeypatch):
    calls = {}

    class FakeMegatronScheduler:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def get_groups_and_subsamples(self, samples):
            calls["samples"] = samples
            return [[[0], [1]]]

    monkeypatch.setattr(dcp_module, "_get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMegatronScheduler)
    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        dp_size=1,
        cp_size=2,
        microbatch_group_size_per_vp_stage=4,
    )

    groups = scheduler._get_groups_and_subsamples([(0, 900), (1, 300)])

    assert groups == [[[0], [1]]]
    assert calls == {
        "kwargs": {
            "max_seqlen_per_dp_cp_rank": 1024,
            "cp_size": 2,
            "dp_size": 1,
            "min_cp_size": 1,
            "microbatch_group_size_per_vp_stage": 4,
        },
        "samples": [(0, 900), (1, 300)],
    }


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_seqlen_per_dp_cp_rank": 0, "dp_size": 1}, "max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": 1024, "dp_size": 0}, "dp_size"),
        ({"max_seqlen_per_dp_cp_rank": 1024, "dp_size": 1, "cp_size": False}, "cp_size"),
        ({"max_seqlen_per_dp_cp_rank": 1024, "dp_size": 1, "min_cp_size": 0}, "min_cp_size"),
    ],
)
def test_dcp_scheduler_rejects_non_positive_topology(kwargs, name):
    with pytest.raises(ValueError, match=name):
        DynamicCPScheduler(**kwargs)


def test_dcp_scheduler_rejects_min_cp_larger_than_group():
    with pytest.raises(ValueError, match="cannot exceed"):
        DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=2, min_cp_size=3)


def test_dcp_batch_field_classification_requires_nested_input_ids():
    batch = TensorDict({"input_ids": torch.ones(2, 4, dtype=torch.long)}, batch_size=[2])

    with pytest.raises(ValueError, match="input_ids to be a NestedTensor"):
        _classify_routed_batch_fields(batch, ["input_ids"], {"temperature"}, _FakeGroup([0], rank=0))


def test_dcp_batch_field_classification_requires_one_scalar_per_sample():
    batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(2), torch.arange(3)], layout=torch.jagged),
            "temperature": torch.ones(2, 2),
        },
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="exactly one dense value per sample"):
        _classify_routed_batch_fields(
            batch,
            ["input_ids", "temperature"],
            {"temperature"},
            _FakeGroup([0], rank=0),
        )


def test_dcp_scheduler_rejects_empty_megatron_rank(monkeypatch):
    class FakeMegatronScheduler:
        def __init__(self, **_kwargs):
            pass

        def get_groups_and_subsamples(self, _samples):
            return [[[0], []]]

    monkeypatch.setattr(dcp_module, "_get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMegatronScheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=2)

    with pytest.raises(RuntimeError, match="empty rank"):
        scheduler._get_groups_and_subsamples([(0, 900)])


def test_dcp_scheduler_rejects_sample_set_mismatch(monkeypatch):
    class FakeMegatronScheduler:
        def __init__(self, **_kwargs):
            pass

        def get_groups_and_subsamples(self, _samples):
            return [[[0], [0]]]

    monkeypatch.setattr(dcp_module, "_get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMegatronScheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=2)

    with pytest.raises(RuntimeError, match="preserve the input sample set"):
        scheduler._get_groups_and_subsamples([(0, 900), (1, 300)])


def test_dcp_scheduler_delegates_non_power_of_two_grouping_to_megatron(monkeypatch):
    calls = {}

    class FakeMegatronScheduler:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def get_groups_and_subsamples(self, samples):
            calls["samples"] = samples
            return [[[0, 1]] * 6]

    monkeypatch.setattr(dcp_module, "_get_megatron_dynamic_cp_scheduler_cls", lambda: FakeMegatronScheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=128, dp_size=3, cp_size=2)

    groups = scheduler._get_groups_and_subsamples([(0, 200), (1, 190)])

    assert groups == [[[0, 1]] * 6]
    assert calls["kwargs"]["dp_size"] == 3
    assert calls["kwargs"]["cp_size"] == 2
    assert calls["samples"] == [(0, 200), (1, 190)]


def test_non_power_of_two_scheduler_preserves_vpp_alignment(monkeypatch):
    class FakeMegatronScheduler:
        def __init__(self, **kwargs):
            assert kwargs["microbatch_group_size_per_vp_stage"] == 2

        def get_groups_and_subsamples(self, _samples):
            return [
                [[0]] * 6,
                [[1]] * 6,
            ]

    monkeypatch.setattr(
        dcp_module,
        "_get_megatron_dynamic_cp_scheduler_cls",
        lambda: FakeMegatronScheduler,
    )
    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=256,
        dp_size=3,
        cp_size=2,
        microbatch_group_size_per_vp_stage=2,
    )

    groups = scheduler._get_groups_and_subsamples([(0, 303), (1, 62)])

    assert len(groups) == 2
    assert all(rank_ids for group in groups for rank_ids in group)
    assigned_ids = [sample_id for group in groups for rank_ids in group for sample_id in rank_ids]
    assert assigned_ids.count(0) == assigned_ids.count(1) == 6


def test_reverse_routing_gathers_all_dynamic_cp_shards_on_canonical_cp_rank(monkeypatch):
    class FakeGroup:
        def __init__(self, ranks, rank):
            self.ranks = ranks
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return len(self.ranks)

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    dcp_group = FakeGroup(list(range(8)), rank=0)
    dp_group = FakeGroup([0, 4], rank=0)
    routing_info = {
        "sample_id_groups": [
            [[0], [0], [1], [1], [2], [2], [3], [3]],
        ],
        "offsets": torch.tensor([0, 2, 4], dtype=torch.int32),
        "global_ids_this_rank": torch.tensor([0, 1], dtype=torch.int32),
    }

    send_by_dest, recv_by_src, _my_gids, _send_ids, recv_ids = _build_reverse_routing_plans(
        routing_info, dp_group, dcp_group
    )

    assert send_by_dest[0] == [0]
    assert recv_by_src[:4] == [[0], [0], [1], [1]]
    assert recv_ids == [0, 0, 1, 1]


def test_prepare_micro_batches_uses_unified_dynamic_cp_entrypoint(monkeypatch):
    calls = {}

    class FakeGroup:
        def size(self):
            return 3

    class FakeScheduler:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def schedule(self, **kwargs):
            calls["schedule"] = kwargs
            return ["microbatch"], {"routing": True}

    monkeypatch.setattr(dcp_module, "DynamicCPScheduler", FakeScheduler)
    data = TensorDict({}, batch_size=[])
    dp_group = FakeGroup()
    dcp_group = FakeGroup()

    result = prepare_micro_batches(
        data,
        dp_group=dp_group,
        dynamic_context_parallel=True,
        dcp_group=dcp_group,
        max_seqlen_per_dp_cp_rank=2048,
        cp_size=4,
        num_batches_divided_by=2,
        non_tensor_data={"compute_loss": True},
    )

    assert result == (["microbatch"], {"routing": True})
    assert calls["kwargs"] == {
        "max_seqlen_per_dp_cp_rank": 2048,
        "dp_size": 3,
        "cp_size": 4,
        "min_cp_size": 1,
        "microbatch_group_size_per_vp_stage": 2,
    }
    assert calls["schedule"] == {
        "batch": data,
        "dp_group": dp_group,
        "dcp_group": dcp_group,
        "non_tensor_data": {"compute_loss": True},
    }


def test_reverse_route_outputs_requires_dp_group():
    with pytest.raises(ValueError, match="data-parallel process group"):
        reverse_route_outputs({}, {}, dp_group=None)


def test_dcp_schedule_does_not_mutate_caller_metadata(monkeypatch):
    class FakeGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1)
    group = FakeGroup()
    metadata = {"compute_loss": False}
    batch = TensorDict(
        {"input_ids": torch.nested.as_nested_tensor([torch.arange(2)], layout=torch.jagged)},
        batch_size=[1],
    )
    received_metadata = {}

    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        dcp_module,
        "_get_global_seqlens_and_ids",
        lambda _local_seqlens, _dp_group: (
            [(0, 2)],
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(scheduler, "_get_groups_and_subsamples", lambda _samples: [[[0]]])
    monkeypatch.setattr(
        dcp_module,
        "_reroute_samples",
        lambda *_args, **_kwargs: {0: {"input_ids": torch.arange(2)}},
    )
    monkeypatch.setattr(
        dcp_module,
        "_build_micro_batches_from_samples",
        lambda *_args, **_kwargs: ([[{"input_ids": torch.arange(2)}]], [1]),
    )

    def fake_to_tensordict(*_args, non_tensor_data, **_kwargs):
        received_metadata.update(non_tensor_data)
        return TensorDict({}, batch_size=[])

    monkeypatch.setattr(dcp_module, "_samples_to_nested_tensor_batch", fake_to_tensordict)

    scheduler.schedule(batch, dp_group=group, dcp_group=group, non_tensor_data=metadata)

    assert metadata == {"compute_loss": False}
    assert received_metadata == {"compute_loss": False, "_dcp_scheduled": True}


def test_dcp_response_lengths_use_attention_span_not_response_mask_sum():
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(8), torch.arange(6)],
        layout=torch.jagged,
    )
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": torch.ones(2, 3, dtype=torch.long),
            "responses": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0],
                ]
            ),
        },
        batch_size=[2],
    )

    response_lens = _get_response_lengths(batch, input_ids.offsets().diff())

    torch.testing.assert_close(response_lens, torch.tensor([5, 4]))


def test_dcp_response_lengths_use_nested_mask_offsets_with_internal_zeros():
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(8), torch.arange(6)],
        layout=torch.jagged,
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 0, 0, 1, 1]), torch.tensor([1, 0, 1, 0])],
        layout=torch.jagged,
    )
    batch = TensorDict(
        {"input_ids": input_ids, "response_mask": response_mask},
        batch_size=[2],
    )

    response_lens = _get_response_lengths(batch, input_ids.offsets().diff())

    torch.testing.assert_close(response_lens, torch.tensor([5, 4]))


def test_dcp_routing_keys_are_minimal_for_forward_only():
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=4)
    batch = {
        "input_ids": object(),
        "position_ids": object(),
        "loss_mask": object(),
        "prompts": object(),
        "responses": object(),
        "attention_mask": object(),
        "response_mask": object(),
        "old_log_probs": object(),
        "advantages": object(),
        "temperature": object(),
    }

    assert scheduler._routing_key_order(batch, {"compute_loss": False}) == ["input_ids", "temperature"]


def test_dcp_routing_keys_include_train_loss_and_optional_model_inputs():
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=4)
    batch = {
        "input_ids": object(),
        "position_ids": object(),
        "loss_mask": object(),
        "prompts": object(),
        "responses": object(),
        "attention_mask": object(),
        "response_mask": object(),
        "old_log_probs": object(),
        "advantages": object(),
        "rollout_is_weights": object(),
        "ref_log_prob": object(),
        "values": object(),
        "returns": object(),
        "routed_experts": object(),
        "teacher_logprobs": object(),
        "teacher_ids": object(),
        "temperature": object(),
    }

    keys = scheduler._routing_key_order(
        batch,
        {
            "compute_loss": True,
            "enable_routing_replay": True,
            "distillation_use_topk": True,
            "_dcp_route_attention_mask": True,
        },
    )

    assert keys == [
        "input_ids",
        "loss_mask",
        "response_mask",
        "old_log_probs",
        "advantages",
        "rollout_is_weights",
        "ref_log_prob",
        "values",
        "returns",
        "routed_experts",
        "teacher_logprobs",
        "teacher_ids",
        "attention_mask",
        "temperature",
    ]


def test_dcp_routes_distillation_tensors_without_topk_flag():
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=2)
    batch = {
        "input_ids": object(),
        "teacher_logprobs": object(),
        "teacher_ids": object(),
    }

    assert scheduler._routing_key_order(batch, {"compute_loss": True, "distillation_use_topk": False}) == [
        "input_ids",
        "teacher_logprobs",
        "teacher_ids",
    ]


def test_dcp_routes_loss_mask_for_forward_only_mtp_when_requested():
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=1024, dp_size=1, cp_size=2)
    batch = {"input_ids": object(), "loss_mask": object()}

    assert scheduler._routing_key_order(
        batch,
        {"compute_loss": False, "_dcp_route_loss_mask": True},
    ) == ["input_ids", "loss_mask"]


class _FakeGroup:
    def __init__(self, ranks, rank):
        self.ranks = ranks
        self._rank = rank

    def rank(self):
        return self._rank

    def size(self):
        return len(self.ranks)


def _copy_all_gather(outputs, input, **_kwargs):
    for output in outputs:
        output.copy_(input)


def test_reroute_empty_local_rank_uses_batch_schema_dtype(monkeypatch):
    dcp_group = _FakeGroup([0], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    payload_dtypes = []

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_to_all_single(output, input, **_kwargs):
        output.copy_(input)
        payload_dtypes.append(input.dtype)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)

    result = _reroute_samples(
        local_samples=[],
        global_ids_this_rank=torch.empty(0, dtype=torch.int32),
        sample_id_groups=[[[]]],
        offsets=torch.tensor([0], dtype=torch.int32),
        dp_group=dp_group,
        dcp_group=dcp_group,
        tensor_keys=["input_ids"],
        scalar_keys=[],
        key_dtypes={"input_ids": torch.int64},
    )

    assert result == {}
    # numel metadata, shape metadata, then the empty tensor payload. The last
    # dtype regressed to float32 before the source batch schema was propagated.
    assert payload_dtypes == [torch.int64, torch.int64, torch.int64]


def test_reroute_preserves_present_zero_length_tensor(monkeypatch):
    dcp_group = _FakeGroup([0], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "all_to_all_single", lambda output, input, **_kwargs: output.copy_(input))

    result = _reroute_samples(
        local_samples=[
            {"input_ids": torch.empty(0, dtype=torch.int64)},
            {"input_ids": torch.tensor([1, 2], dtype=torch.int64)},
        ],
        global_ids_this_rank=torch.tensor([0, 1], dtype=torch.int32),
        sample_id_groups=[[[0, 1]]],
        offsets=torch.tensor([0, 2], dtype=torch.int32),
        dp_group=dp_group,
        dcp_group=dcp_group,
        tensor_keys=["input_ids"],
        scalar_keys=[],
        key_dtypes={"input_ids": torch.int64},
    )

    assert result[0]["input_ids"].shape == (0,)
    torch.testing.assert_close(result[1]["input_ids"], torch.tensor([1, 2]))


def test_routing_schema_mismatch_is_rejected_before_all_to_all(monkeypatch):
    dcp_group = _FakeGroup([0, 1], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_gather(gathered, signature, *, group):
        assert group is dcp_group
        gathered[0].copy_(signature)
        gathered[1].copy_(signature)
        gathered[1][-1] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("all-to-all must not be reached")),
    )

    with pytest.raises(ValueError, match="schema differs across ranks before all-to-all"):
        _reroute_samples(
            local_samples=[{"input_ids": torch.tensor([1, 2])}],
            global_ids_this_rank=torch.tensor([0], dtype=torch.int32),
            sample_id_groups=[[[0], [0]]],
            offsets=torch.tensor([0, 1], dtype=torch.int32),
            dp_group=dp_group,
            dcp_group=dcp_group,
            tensor_keys=["input_ids"],
            scalar_keys=[],
            key_dtypes={"input_ids": torch.int64},
        )


def test_reverse_route_empty_canonical_owner_returns_empty_batch(monkeypatch):
    dcp_group = _FakeGroup([0, 1, 2, 3], rank=2)
    dp_group = _FakeGroup([0, 2], rank=1)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "all_gather", _copy_all_gather)

    def fake_all_to_all_single(output, input, **_kwargs):
        del input
        output.zero_()

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)
    scheduled_foreign_output = torch.nested.as_nested_tensor(
        [torch.tensor([1.0, 2.0])],
        layout=torch.jagged,
    )

    result = reverse_route_outputs(
        {"log_probs": scheduled_foreign_output},
        {
            "sample_id_groups": [[[0], [0], [0], [0]]],
            "offsets": torch.tensor([0, 1, 1], dtype=torch.int32),
            "global_ids_this_rank": torch.empty(0, dtype=torch.int32),
        },
        dp_group=dp_group,
        dcp_group=dcp_group,
        merge_duplicate_gids=True,
    )

    assert result["log_probs"].is_nested
    assert len(result["log_probs"].unbind()) == 0


def test_reverse_route_preserves_present_zero_length_output(monkeypatch):
    dcp_group = _FakeGroup([0], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "all_to_all_single", lambda output, input, **_kwargs: output.copy_(input))
    model_output = {
        "log_probs": torch.nested.as_nested_tensor(
            [torch.empty(0), torch.tensor([1.0, 2.0])],
            layout=torch.jagged,
        )
    }

    result = reverse_route_outputs(
        model_output,
        {
            "sample_id_groups": [[[0, 1]]],
            "offsets": torch.tensor([0, 2], dtype=torch.int32),
            "global_ids_this_rank": torch.tensor([0, 1], dtype=torch.int32),
        },
        dp_group=dp_group,
        dcp_group=dcp_group,
    )

    parts = result["log_probs"].unbind()
    assert parts[0].shape == (0,)
    torch.testing.assert_close(parts[1], torch.tensor([1.0, 2.0]))


def test_reconstruct_compact_sample_requires_exact_token_coverage():
    result = _reconstruct_compact_sample(
        7,
        [torch.tensor([[10.0], [40.0]]), torch.tensor([[20.0], [30.0]])],
        [torch.tensor([0, 3]), torch.tensor([1, 2])],
        [torch.tensor([4]), torch.tensor([4])],
    )

    torch.testing.assert_close(result, torch.tensor([[10.0], [20.0], [30.0], [40.0]]))


@pytest.mark.parametrize(
    "indices, full_len",
    [
        ([torch.tensor([0]), torch.tensor([2])], 3),
        ([torch.tensor([0, 1]), torch.tensor([1, 2])], 4),
        ([torch.tensor([0, 1]), torch.tensor([2, 4])], 4),
    ],
)
def test_reconstruct_compact_sample_rejects_missing_duplicate_or_out_of_range_indices(indices, full_len):
    values = [torch.arange(part.numel(), dtype=torch.float32) for part in indices]

    with pytest.raises(ValueError, match="must cover"):
        _reconstruct_compact_sample(
            3,
            values,
            indices,
            [torch.tensor([full_len]) for _ in indices],
        )


def test_reconstruct_compact_sample_rejects_inconsistent_full_lengths():
    with pytest.raises(ValueError, match="inconsistent full sequence lengths"):
        _reconstruct_compact_sample(
            3,
            [torch.tensor([1.0]), torch.tensor([2.0])],
            [torch.tensor([0]), torch.tensor([1])],
            [torch.tensor([2]), torch.tensor([3])],
        )


def test_reverse_route_rejects_output_batch_cardinality_mismatch(monkeypatch):
    dcp_group = _FakeGroup([0], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("all-to-all must not be reached")),
    )
    empty_output = torch.nested.nested_tensor_from_jagged(
        torch.empty(0),
        offsets=torch.tensor([0]),
        min_seqlen=0,
        max_seqlen=0,
    )

    with pytest.raises(ValueError, match="batch size must match the scheduled local sample count"):
        reverse_route_outputs(
            {"log_probs": empty_output},
            {
                "sample_id_groups": [[[0]]],
                "offsets": torch.tensor([0, 1], dtype=torch.int32),
                "global_ids_this_rank": torch.tensor([0], dtype=torch.int32),
            },
            dp_group=dp_group,
            dcp_group=dcp_group,
        )


def test_reverse_route_rejects_scalar_tensor_before_all_to_all(monkeypatch):
    dcp_group = _FakeGroup([0], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("all-to-all must not be reached")),
    )

    with pytest.raises(ValueError, match="must have a batch dimension"):
        reverse_route_outputs(
            {"loss": torch.tensor(1.0)},
            {
                "sample_id_groups": [[[0]]],
                "offsets": torch.tensor([0, 1], dtype=torch.int32),
                "global_ids_this_rank": torch.tensor([0], dtype=torch.int32),
            },
            dp_group=dp_group,
            dcp_group=dcp_group,
        )


def test_reverse_routing_schema_mismatch_is_rejected_before_all_to_all(monkeypatch):
    dcp_group = _FakeGroup([0, 1], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_all_gather(gathered, signature, *, group):
        assert group is dcp_group
        gathered[0].copy_(signature)
        gathered[1].copy_(signature)
        gathered[1][-1] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("all-to-all must not be reached")),
    )

    with pytest.raises(ValueError, match="schema differs across ranks before all-to-all"):
        reverse_route_outputs(
            {"log_probs": torch.nested.as_nested_tensor([torch.tensor([1.0])], layout=torch.jagged)},
            {
                "sample_id_groups": [[[0], [0]]],
                "offsets": torch.tensor([0, 1], dtype=torch.int32),
                "global_ids_this_rank": torch.tensor([0], dtype=torch.int32),
            },
            dp_group=dp_group,
            dcp_group=dcp_group,
        )


def test_reverse_route_keeps_full_sequence_routed_experts_with_compact_metadata(monkeypatch):
    dcp_group = _FakeGroup([0, 1], rank=0)
    dp_group = _FakeGroup([0], rank=0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "all_gather", _copy_all_gather)

    def fake_all_to_all_single(output, input, **_kwargs):
        # Simulate the second CP rank returning the same replicated full-sequence
        # router map (and matching compact metadata for the other outputs).
        output.copy_(input.repeat(2))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)

    routed_experts = torch.arange(8, dtype=torch.int64).reshape(4, 2, 1)
    model_output = {
        "routed_experts": torch.nested.as_nested_tensor([routed_experts], layout=torch.jagged),
        "_dcp_local_token_indices": torch.nested.as_nested_tensor(
            [torch.tensor([0, 3], dtype=torch.int64)], layout=torch.jagged
        ),
        "_dcp_full_seq_lens": torch.nested.as_nested_tensor(
            [torch.tensor([4], dtype=torch.int64)], layout=torch.jagged
        ),
    }
    routing_info = {
        "sample_id_groups": [[[0], [0]]],
        "offsets": torch.tensor([0, 1], dtype=torch.int32),
        "global_ids_this_rank": torch.tensor([0], dtype=torch.int32),
    }

    result = reverse_route_outputs(
        model_output,
        routing_info,
        dp_group=dp_group,
        dcp_group=dcp_group,
        merge_duplicate_gids=True,
    )

    torch.testing.assert_close(result["routed_experts"].unbind()[0], routed_experts)


def test_pp_metadata_broadcast_preserves_existing_routed_fields(monkeypatch):
    pp_group = _FakeGroup([0, 1, 2], rank=1)
    serialized = torch.tensor([1, 2, 3, 0, 4, 10], dtype=torch.int32)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(dcp_module, "get_device_id", lambda: torch.device("cpu"))

    def fake_broadcast(tensor, _src, group):
        assert group is pp_group
        if tensor.numel() == 1:
            tensor.fill_(serialized.numel())
        else:
            tensor.copy_(serialized)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    routed_experts = torch.nested.as_nested_tensor(
        [torch.ones(4, 2, 1, dtype=torch.int64), torch.ones(6, 2, 1, dtype=torch.int64)],
        layout=torch.jagged,
    )
    existing = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.ones(4), torch.ones(6)], layout=torch.jagged),
            "routed_experts": routed_experts,
        },
        batch_size=[2],
    )

    result = broadcast_dcp_metadata_to_pp([existing], pp_group)

    assert result[0] is existing
    torch.testing.assert_close(result[0]["routed_experts"].values(), routed_experts.values())
    assert result[0]["input_ids"].offsets().tolist() == [0, 4, 10]
    assert dcp_module.tu.get_non_tensor_data(result[0], key="local_cp_size", default=None) == 2
