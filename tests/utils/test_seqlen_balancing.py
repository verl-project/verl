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

import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.model import create_random_mask
from verl.utils.seqlen_balancing import (
    ceildiv,
    get_equal_size_group_partitions,
    get_reverse_idx,
    prepare_dynamic_batch,
    rearrange_micro_batches,
    restore_dynamic_batch,
)


def test_seqlen_balancing():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = rearrange_micro_batches(dataproto.batch, max_token_len=300)
    batch = torch.cat(micro_batches)
    micro_bsz_idx = []
    for idx in micro_bsz_idx_lst:
        micro_bsz_idx.extend(idx)
    reverse_idx_map = get_reverse_idx(micro_bsz_idx)
    reverse_idx_map = torch.tensor(reverse_idx_map)
    new_batch = batch[reverse_idx_map]
    torch.testing.assert_close(new_batch, dataproto.batch)


def test_micro_batches_respect_max_token_len():
    input_ids = torch.zeros((8, 7), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    dataproto = DataProto.from_single_dict({"input_ids": input_ids, "attention_mask": attention_mask})

    micro_batches, _ = rearrange_micro_batches(dataproto.batch, max_token_len=8)

    assert len(micro_batches) == 8
    assert all(micro_batch["attention_mask"].sum().item() <= 8 for micro_batch in micro_batches)


def test_dynamic_batch():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = prepare_dynamic_batch(dataproto, max_token_len=300)
    input_ids = torch.cat([micro_batch.batch["input_ids"] for micro_batch in micro_batches], dim=0)
    input_ids = restore_dynamic_batch(input_ids, micro_bsz_idx_lst)
    torch.testing.assert_close(input_ids, dataproto.batch["input_ids"])


def _worker(rank, world_size, init_method, max_token_len, use_same_dp, min_mb):
    # 1) init process group & CUDA
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    # 2) build a small random batch (each rank different length to force mismatch)
    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, 10, (20 + rank * 5, 100), device=f"{get_device_name()}:{rank}")
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0.1,
        max_ratio_of_valid_token=0.9,
        min_ratio_of_valid_token=0.5,
    )
    dp = {"input_ids": input_ids, "attention_mask": attention_mask}
    proto = DataProto.from_single_dict(dp)
    batch = proto.batch

    # 3) call rearrange_micro_batches with one of the two params under test
    micros, idx_lst = rearrange_micro_batches(
        batch,
        max_token_len=max_token_len,
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=use_same_dp,
        min_num_micro_batch=min_mb,
    )

    # 4) check the enforced counts and token limit
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    minimum = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_mb is not None:
        minimum = max(minimum, min_mb)
    assert len(micros) >= minimum
    assert all(micro["attention_mask"].sum().item() <= max_token_len for micro in micros)

    if use_same_dp:
        # All ranks must use the same final count, including any upward search.
        counts = [torch.zeros(1, device=f"{get_device_name()}:{rank}") for _ in range(world_size)]
        counts[rank].fill_(len(micros))
        dist.all_gather(counts, counts[rank])
        assert len({int(count.item()) for count in counts}) == 1

    # 5) reconstruction sanity: concat→reverse_idx→orig
    flat = torch.cat(micros, dim=0)
    idx = []
    for sub in idx_lst:
        idx.extend(sub)
    inv = get_reverse_idx(idx)
    inv = torch.tensor(inv, device=flat.device)
    reconstructed = flat[inv]
    torch.testing.assert_close(reconstructed, batch)

    dist.destroy_process_group()


def _constraint_error_worker(rank, world_size, init_method, scenario):
    import verl.utils.seqlen_balancing as seqlen_balancing

    dist.init_process_group(backend="gloo", init_method=init_method, world_size=world_size, rank=rank)
    seqlen_balancing.get_device_name = lambda: "cpu"

    if scenario == "oversized_group":
        seq_lens = [6, 6, 1, 1] if rank == 0 else [4, 4, 1, 1]
        force_group_size = 2
        expected_error = "forced group exceeds max_token_len"
    else:
        seq_lens = [1, 1] if rank == 0 else [6, 6, 6]
        force_group_size = 1
        expected_error = "same micro-batch count across DP ranks"

    max_seq_len = max(seq_lens)
    attention_mask = torch.zeros((len(seq_lens), max_seq_len), dtype=torch.long)
    for i, seq_len in enumerate(seq_lens):
        attention_mask[i, :seq_len] = 1
    batch = DataProto.from_single_dict(
        {"input_ids": torch.zeros_like(attention_mask), "attention_mask": attention_mask}
    ).batch

    try:
        seqlen_balancing.rearrange_micro_batches(
            batch,
            max_token_len=10,
            dp_group=dist.group.WORLD,
            same_micro_num_in_dp=True,
            force_group_size=force_group_size,
        )
    except ValueError as error:
        assert expected_error in str(error)
    else:
        raise AssertionError("Expected a synchronized ValueError")

    dist.barrier()
    dist.destroy_process_group()


def _mask_batch(seq_lens):
    attention_mask = torch.zeros((len(seq_lens), max(seq_lens)), dtype=torch.long)
    for i, seq_len in enumerate(seq_lens):
        attention_mask[i, :seq_len] = 1
    return DataProto.from_single_dict(
        {"input_ids": torch.zeros_like(attention_mask), "attention_mask": attention_mask}
    ).batch


def _splittable_groups_worker(rank, world_size, init_method, scenario):
    """Prefix-packed groups: a rank with fewer groups must split them to match the agreed micro-batch count."""
    import verl.utils.seqlen_balancing as seqlen_balancing

    dist.init_process_group(backend="gloo", init_method=init_method, world_size=world_size, rank=rank)
    seqlen_balancing.get_device_name = lambda: "cpu"

    if rank == 0:
        # One group of 4 samples sharing a 51-token prompt: packed 50 + 4 * 10 = 90 tokens, so it asks for
        # one micro-batch. Rank 1 forces more, and this rank has to split the group to keep up.
        seq_lens = [60, 60, 60, 60] if scenario == "two_splits" else [60, 60]
        kwargs = dict(group_sizes=[len(seq_lens)], group_token_lens=[50 + 10 * len(seq_lens)], group_trunk_lens=[50])
    else:
        # Four singleton groups of 55 tokens. ceil(220 / 100) = 3, but no 3-way partition fits 100 tokens,
        # so the loop raises the count to 4 after the first partition attempt.
        seq_lens = [55, 55, 55, 55]
        kwargs = dict(group_sizes=[1] * 4, group_token_lens=[55] * 4, group_trunk_lens=[0] * 4)

    batch = _mask_batch(seq_lens)
    try:
        micro_batches, idx_lists = seqlen_balancing.rearrange_micro_batches(
            batch, max_token_len=100, dp_group=dist.group.WORLD, same_micro_num_in_dp=True, **kwargs
        )
    except ValueError as error:
        assert scenario == "too_few_samples", str(error)
        assert "a rank has only 2 samples" in str(error)
    else:
        assert scenario == "two_splits"
        # Rank 0 asked for 1 micro-batch, was raised to 3 by the initial MAX (first split), then to 4 when
        # rank 1's 3-way partition went over budget (second split, inside the loop).
        assert len(micro_batches) == 4
        assert sorted(i for idx in idx_lists for i in idx) == list(range(len(seq_lens)))
        assert all(int(mb["attention_mask"].sum()) <= 100 for mb in micro_batches)

    dist.barrier()
    dist.destroy_process_group()


def test_splittable_groups_follow_dp_micro_batch_count(tmp_path):
    world_size = 2
    for scenario in ("two_splits", "too_few_samples"):
        init_file = tmp_path / f"dist_init_{scenario}"
        init_file.write_text("")
        mp.spawn(
            _splittable_groups_worker,
            args=(world_size, f"file://{init_file}", scenario),
            nprocs=world_size,
            join=True,
        )


def test_splittable_group_meets_min_num_micro_batch():
    """Single rank: min_num_micro_batch above the group count splits the group instead of raising."""
    from verl.utils.seqlen_balancing import rearrange_micro_batches

    batch = _mask_batch([60, 60, 60, 60])
    group_sizes = [4]
    micro_batches, idx_lists = rearrange_micro_batches(
        batch,
        max_token_len=100,
        min_num_micro_batch=3,
        group_sizes=group_sizes,
        group_token_lens=[90],
        group_trunk_lens=[50],
    )
    assert len(micro_batches) == 3
    assert sorted(len(idx) for idx in idx_lists) == [1, 1, 2]
    assert sorted(i for idx in idx_lists for i in idx) == [0, 1, 2, 3]
    assert group_sizes == [4], "caller's group_sizes must not be mutated"

    with pytest.raises(ValueError, match="forced groups atomic"):
        rearrange_micro_batches(batch, max_token_len=100, min_num_micro_batch=3, group_sizes=[4], group_token_lens=[90])


def test_dataproto_split_uneven():
    """Test DataProto.split with uneven splits"""
    # Create test data with 10 items
    input_ids = torch.randint(low=0, high=10, size=(10, 5))
    attention_mask = torch.ones(10, 5)
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)

    # Test split with size 3 (should create chunks of [3, 3, 3, 1])
    splits = dataproto.split(3)
    assert len(splits) == 4
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3
    assert len(splits[2]) == 3
    assert len(splits[3]) == 1

    reconstructed = DataProto.concat(splits)
    torch.testing.assert_close(reconstructed.batch["input_ids"], dataproto.batch["input_ids"])
    torch.testing.assert_close(reconstructed.batch["attention_mask"], dataproto.batch["attention_mask"])

    # Test split with size equal to length (should create one chunk)
    splits = dataproto.split(10)
    assert len(splits) == 1
    assert len(splits[0]) == 10

    # Test split with size larger than length (should create one chunk with all data)
    splits = dataproto.split(15)
    assert len(splits) == 1
    assert len(splits[0]) == 10

    # Test with non-tensor batch data
    import numpy as np

    data_with_non_tensor = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": np.array([f"label_{i}" for i in range(10)], dtype=object),
    }
    dataproto_with_non_tensor = DataProto.from_single_dict(data_with_non_tensor)

    splits = dataproto_with_non_tensor.split(3)
    assert len(splits) == 4
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3
    assert len(splits[2]) == 3
    assert len(splits[3]) == 1

    # Verify non-tensor data integrity
    reconstructed = DataProto.concat(splits)
    np.testing.assert_array_equal(
        reconstructed.non_tensor_batch["labels"], dataproto_with_non_tensor.non_tensor_batch["labels"]
    )


def test_seqlen_balancing_distributed_params(tmp_path):
    world_size = 2
    init_file = tmp_path / "dist_init"
    init_file.write_text("")  # empty file
    init_method = f"file://{init_file}"

    # test min_num_micro_batch only
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, False, 4),
        nprocs=world_size,
        join=True,
    )

    # test same_micro_num_in_dp only
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, True, None),
        nprocs=world_size,
        join=True,
    )


def test_seqlen_balancing_distributed_constraint_errors(tmp_path):
    world_size = 2
    for scenario in ("oversized_group", "insufficient_groups"):
        init_file = tmp_path / f"dist_init_{scenario}"
        init_file.write_text("")
        mp.spawn(
            _constraint_error_worker,
            args=(world_size, f"file://{init_file}", scenario),
            nprocs=world_size,
            join=True,
        )


def test_group_balanced_partitions():
    """Test group-level balancing keeps same-uid samples together."""
    from verl.utils.seqlen_balancing import get_group_balanced_partitions

    # Create test data: 4 groups with different sizes
    # Group 0 (uid=0): indices 0,1,2,3 with seqlens [100, 100, 100, 100]
    # Group 1 (uid=1): indices 4,5,6,7 with seqlens [200, 200, 200, 200]
    # Group 2 (uid=2): indices 8,9,10,11 with seqlens [150, 150, 150, 150]
    # Group 3 (uid=3): indices 12,13,14,15 with seqlens [50, 50, 50, 50]
    seqlen_list = [100] * 4 + [200] * 4 + [150] * 4 + [50] * 4
    uid_list = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4

    # Partition into 2 groups
    partitions = get_group_balanced_partitions(seqlen_list, uid_list, k_partitions=2)

    assert len(partitions) == 2

    # Verify all indices are covered
    all_indices = set()
    for partition in partitions:
        all_indices.update(partition)
    assert all_indices == set(range(16))

    # Verify same-uid samples stay together
    for partition in partitions:
        uids_in_partition = set(uid_list[i] for i in partition)
        for uid in uids_in_partition:
            # All samples with this uid should be in this partition
            uid_indices = [i for i, u in enumerate(uid_list) if u == uid]
            assert all(i in partition for i in uid_indices), f"uid {uid} samples split across partitions"


def test_group_balanced_partitions_single_sample_groups():
    """Test group balancing with single-sample groups (n=1)."""
    from verl.utils.seqlen_balancing import get_group_balanced_partitions

    # Each sample is its own group
    seqlen_list = [100, 200, 150, 50, 300, 250]
    uid_list = [0, 1, 2, 3, 4, 5]

    partitions = get_group_balanced_partitions(seqlen_list, uid_list, k_partitions=2)

    assert len(partitions) == 2
    all_indices = set()
    for partition in partitions:
        all_indices.update(partition)
    assert all_indices == set(range(6))


def test_group_balanced_partitions_equal_size():
    """Test group balancing with equal_size constraint simulation."""
    from verl.utils.seqlen_balancing import get_group_balanced_partitions

    # 8 groups, partition into 4 (simulating world_size=4)
    # Each group has 2 samples
    seqlen_list = [100, 100, 200, 200, 150, 150, 50, 50, 300, 300, 250, 250, 180, 180, 120, 120]
    uid_list = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

    partitions = get_group_balanced_partitions(seqlen_list, uid_list, k_partitions=4)

    assert len(partitions) == 4

    # Verify all indices are covered
    all_indices = set()
    for partition in partitions:
        all_indices.update(partition)
    assert all_indices == set(range(16))

    # Verify same-uid samples stay together
    for partition in partitions:
        uids_in_partition = set(uid_list[i] for i in partition)
        for uid in uids_in_partition:
            uid_indices = [i for i, u in enumerate(uid_list) if u == uid]
            assert all(i in partition for i in uid_indices)


class TestEqualSizeGroupPartitions:
    @pytest.mark.parametrize("seed", range(10))
    def test_groups_stay_whole_and_partitions_are_equal(self, seed):
        rng = random.Random(seed)
        n, dp, num_prompts = 8, 4, 16
        group_ids = [f"uid{p}" for p in range(num_prompts) for _ in range(n)]
        rng.shuffle(group_ids)
        workloads = [rng.randint(100, 5000) for _ in group_ids]
        parts = get_equal_size_group_partitions(workloads, group_ids, dp)
        assert sorted(i for p in parts for i in p) == list(range(len(group_ids)))
        assert all(len(p) == len(group_ids) // dp for p in parts)
        for part in parts:
            ids = [group_ids[i] for i in part]
            for gid in set(ids):
                assert ids.count(gid) == n, "a group was split although every group fits"
                first = ids.index(gid)
                assert ids[first : first + n] == [gid] * n, "group members are not adjacent"

    def test_uneven_groups_still_give_equal_partitions(self):
        group_ids = ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["pad"]
        parts = get_equal_size_group_partitions([1] * 16, group_ids, 2)
        assert [len(p) for p in parts] == [8, 8]
        assert sorted(i for p in parts for i in p) == list(range(16))

    def test_group_larger_than_partition_is_split(self):
        group_ids = ["a"] * 10 + ["b"] * 2
        parts = get_equal_size_group_partitions([1] * 12, group_ids, 3)
        assert [len(p) for p in parts] == [4, 4, 4]
        assert sorted(i for p in parts for i in p) == list(range(12))
        assert sum(1 for p in parts if any(group_ids[i] == "a" for i in p)) == 3
