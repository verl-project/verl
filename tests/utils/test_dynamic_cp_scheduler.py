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

from collections import Counter, deque

from verl.utils.dynamic_cp_scheduler import (
    DynamicCPScheduler,
    dcp_gpus_needed,
    next_hdp_group,
)


def _sample_replication_counts(sample_id_groups):
    counts = Counter()
    for group in sample_id_groups:
        for rank_ids in group:
            counts.update(rank_ids)
    return counts


def _cp_group_counts(sample_id_groups):
    counts = Counter()
    for group in sample_id_groups:
        rank = 0
        while rank < len(group):
            rank_ids = group[rank]
            if not rank_ids:
                rank += 1
                continue
            first_id = rank_ids[0]
            cp_size = 0
            while rank + cp_size < len(group) and first_id in group[rank + cp_size]:
                cp_size += 1
            counts[cp_size] += 1
            rank += cp_size
    return counts


def test_dcp_scheduler_uses_megatron_sequence_cap_for_packing():
    sample_id_seqlens = [(idx, 2500) for idx in range(8)]

    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1536,
        dp_size=1,
        cp_size=4,
    )

    groups = scheduler._get_groups_and_subsamples(sample_id_seqlens)

    assert dcp_gpus_needed(2500, 1536) == 2
    assert len(groups) == 4
    assert _sample_replication_counts(groups) == {idx: 2 for idx in range(8)}


def test_dcp_scheduler_does_not_promote_short_tail_into_cp4_groups():
    long_samples = [(idx, 4633) for idx in range(8)]
    short_samples = [(idx, 150) for idx in range(8, 32)]

    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=2304,
        dp_size=1,
        cp_size=4,
    )

    groups = scheduler._get_groups_and_subsamples(long_samples + short_samples)
    replication_counts = _sample_replication_counts(groups)

    assert _cp_group_counts(groups) == {1: 4, 4: 8}
    assert {idx: replication_counts[idx] for idx in range(8)} == {idx: 4 for idx in range(8)}
    assert {idx: replication_counts[idx] for idx in range(8, 32)} == {idx: 1 for idx in range(8, 32)}


def test_dcp_scheduler_accepts_explicit_larger_packing_budget():
    sample_id_seqlens = [(idx, 2500) for idx in range(8)]

    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1536,
        max_token_len_per_rank=5000,
        dp_size=1,
        cp_size=4,
    )

    groups = scheduler._get_groups_and_subsamples(sample_id_seqlens)

    assert dcp_gpus_needed(2500, 1536) == 2
    assert len(groups) == 1
    assert _sample_replication_counts(groups) == {idx: 2 for idx in range(8)}


def test_dcp_scheduler_scans_past_full_packing_group():
    buckets = [
        deque([(0, 2500), (1, 2500)]),
        deque([(2, 1000), (3, 1000), (4, 1000)]),
    ]

    def make_buckets_equal_fn(_sample_seqlens, _compute_estimator):
        return [deque(bucket) for bucket in buckets]

    def gpus_needed_fn(seq_len):
        return 2 if seq_len > 1536 else 1

    def get_total_workload_fn(seq_len, cp_size=None):
        return seq_len / (cp_size or gpus_needed_fn(seq_len))

    micro_batches, leftovers, _exec_times, sample_ids = next_hdp_group(
        [(idx, seq_len) for bucket in buckets for idx, seq_len in bucket],
        compute_estimator=get_total_workload_fn,
        total_gpus=3,
        gpus_needed_fn=gpus_needed_fn,
        make_buckets_equal_fn=make_buckets_equal_fn,
        max_seq_len_per_rank=1536,
        get_total_workload_fn=get_total_workload_fn,
        max_token_len_per_rank=2000,
    )

    assert micro_batches == [[2500], [2500], [1000, 1000]]
    assert sample_ids == [[0], [0], [2, 3]]
    assert leftovers == [(1, 2500), (4, 1000)]


def test_dcp_routing_keys_are_minimal_for_forward_only():
    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        dp_size=1,
        cp_size=4,
    )
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
    scheduler = DynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024,
        dp_size=1,
        cp_size=4,
    )
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
