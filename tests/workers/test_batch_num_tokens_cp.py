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

"""Test that batch_num_tokens aggregation uses the DP+CP group when context parallelism > 1.

Regression test for https://github.com/verl-project/verl/issues/5983:
When CP > 1, each CP rank holds 1/CP of the sequence. Reducing batch_num_tokens
over pure DP misses other CP ranks' partial token counts, undercounting by factor CP.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _init_process_groups(rank, world_size, dp_size, cp_size, rendezvous_file):
    """Initialize process groups mimicking Megatron's DP and DP+CP groups."""
    init_method = f"file://{rendezvous_file}"
    dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=world_size)

    # Build pure DP group: ranks that share the same CP position.
    # Layout: ranks are assigned round-robin to CP positions.
    # With DP=2, CP=2, world=4: CP0={0,2}, CP1={1,3}; pure DP groups: {0,2} and {1,3}.
    dp_groups = []
    for cp_pos in range(cp_size):
        dp_ranks = list(range(cp_pos, world_size, cp_size))
        group = dist.new_group(dp_ranks)
        dp_groups.append((dp_ranks, group))

    # Build DP+CP group: all ranks (when TP=1, PP=1)
    dp_cp_group = dist.new_group(list(range(world_size)))

    cp_rank = rank % cp_size
    my_dp_group = dp_groups[cp_rank][1]

    return my_dp_group, dp_cp_group, dp_size, dp_size * cp_size


def _worker_fn(rank, world_size, dp_size, cp_size, rendezvous_file, results_dict):
    """Worker that computes batch_num_tokens with both pure DP and DP+CP groups."""
    my_dp_group, dp_cp_group, pure_dp_size, dp_cp_size = _init_process_groups(
        rank, world_size, dp_size, cp_size, rendezvous_file
    )

    # Simulate: each rank has a different number of valid tokens in its CP slice
    local_tokens = torch.tensor((rank + 1) * 10, dtype=torch.int64)

    # --- Buggy path: pure DP group ---
    buggy_count = local_tokens.clone()
    dist.all_reduce(buggy_count, op=dist.ReduceOp.SUM, group=my_dp_group)

    # --- Fixed path: DP+CP group ---
    correct_count = local_tokens.clone()
    dist.all_reduce(correct_count, op=dist.ReduceOp.SUM, group=dp_cp_group)

    results_dict[rank] = {
        "local_tokens": local_tokens.item(),
        "buggy_count": buggy_count.item(),
        "correct_count": correct_count.item(),
        "pure_dp_size": pure_dp_size,
        "dp_cp_size": dp_cp_size,
    }

    dist.destroy_process_group()


def test_batch_num_tokens_with_context_parallel():
    """With CP=2, pure DP group undercounts tokens; DP+CP group gives correct total."""
    dp_size, cp_size = 2, 2
    world_size = dp_size * cp_size  # 4 processes

    with tempfile.TemporaryDirectory() as tmp_dir:
        rendezvous_file = os.path.join(tmp_dir, "rendezvous")
        results = mp.Manager().dict()
        mp.spawn(
            _worker_fn,
            args=(world_size, dp_size, cp_size, rendezvous_file, results),
            nprocs=world_size,
            join=True,
        )

    # Expected total across ALL ranks: 10+20+30+40 = 100
    expected_total = 100

    for rank in range(world_size):
        r = results[rank]
        # DP+CP group should give the global total
        assert r["correct_count"] == expected_total, (
            f"Rank {rank}: DP+CP all_reduce should give {expected_total}, got {r['correct_count']}"
        )
        # Pure DP group gives less: {0,2}→10+30=40; {1,3}→20+40=60
        assert r["buggy_count"] < expected_total, (
            f"Rank {rank}: pure DP all_reduce should undercount, got {r['buggy_count']}"
        )
        # Verify dp_size values
        assert r["pure_dp_size"] == dp_size
        assert r["dp_cp_size"] == dp_size * cp_size


def test_batch_num_tokens_without_context_parallel():
    """With CP=1, both groups are equivalent — no regression."""
    dp_size, cp_size = 4, 1
    world_size = dp_size * cp_size  # 4 processes

    with tempfile.TemporaryDirectory() as tmp_dir:
        rendezvous_file = os.path.join(tmp_dir, "rendezvous")
        results = mp.Manager().dict()
        mp.spawn(
            _worker_fn,
            args=(world_size, dp_size, cp_size, rendezvous_file, results),
            nprocs=world_size,
            join=True,
        )

    expected_total = 100  # 10+20+30+40

    for rank in range(world_size):
        r = results[rank]
        # Both groups should give the same result when CP=1
        assert r["correct_count"] == expected_total
        assert r["buggy_count"] == expected_total
        # dp_size values should be equal when CP=1
        assert r["pure_dp_size"] == r["dp_cp_size"]


if __name__ == "__main__":
    test_batch_num_tokens_with_context_parallel()
    print("PASSED: test_batch_num_tokens_with_context_parallel")
    test_batch_num_tokens_without_context_parallel()
    print("PASSED: test_batch_num_tokens_without_context_parallel")
