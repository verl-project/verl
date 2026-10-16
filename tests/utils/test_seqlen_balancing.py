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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.model import create_random_mask
from verl.utils.seqlen_balancing import (
    ceildiv,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    karmarkar_karp,
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


def _spread(seqlen_list, partitions):
    """Max minus min of per-partition workload sums."""
    sums = [sum(seqlen_list[i] for i in p) for p in partitions]
    return max(sums) - min(sums)


def test_get_seqlen_balanced_partitions_equal_size_invariants():
    """The equal_size path is what all three trainers use to balance DP ranks.

    get_seqlen_balanced_partitions(..., equal_size=True) is called in ray_trainer,
    v1/trainer_base and sft_trainer_ray with k_partitions=dp_size, yet only
    get_group_balanced_partitions had direct coverage. Pin the contract here.
    """
    seqlen_list = [97, 32, 55, 12, 80, 3, 44, 61, 25, 70, 18, 90]
    k = 4
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=k, equal_size=True)

    # k partitions, each with exactly len / k items
    assert len(partitions) == k
    assert all(len(p) == len(seqlen_list) // k for p in partitions)

    # every index covered exactly once
    flat = sorted(i for p in partitions for i in p)
    assert flat == list(range(len(seqlen_list)))

    # each partition's indices are returned sorted (documented behaviour)
    for p in partitions:
        assert p == sorted(p)


def test_get_seqlen_balanced_partitions_unequal_size_covers_and_nonempty():
    """equal_size=False must still cover all indices and leave no empty partition."""
    seqlen_list = [10, 20, 30, 40, 50, 60, 70]
    k = 3
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=k, equal_size=False)

    assert len(partitions) == k
    assert all(len(p) > 0 for p in partitions)
    flat = sorted(i for p in partitions for i in p)
    assert flat == list(range(len(seqlen_list)))


def test_karmarkar_karp_beats_naive_contiguous_split():
    """The whole point of KK is a smaller workload spread than a naive chunking.

    A degenerate implementation that returned contiguous chunks would still pass
    the coverage/size invariants above, so this pins the balancing property itself.
    """
    # Sorted-ascending lengths make a contiguous split maximally unbalanced.
    seqlen_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    k = 3

    kk_partitions = karmarkar_karp(seqlen_list, k_partitions=k, equal_size=True)

    # Naive contiguous equal-size split for comparison.
    per = len(seqlen_list) // k
    naive_partitions = [list(range(i * per, (i + 1) * per)) for i in range(k)]

    assert _spread(seqlen_list, kk_partitions) < _spread(seqlen_list, naive_partitions)


def test_karmarkar_karp_is_deterministic():
    """Same input must give the same partition — DP ranks rely on this agreeing."""
    seqlen_list = [5, 5, 5, 3, 9, 1, 7, 2, 8, 4, 6, 0]
    first = karmarkar_karp(seqlen_list, k_partitions=4, equal_size=True)
    second = karmarkar_karp(seqlen_list, k_partitions=4, equal_size=True)
    assert first == second


def test_karmarkar_karp_all_equal_lengths_is_perfectly_balanced():
    """Equal lengths admit a zero-spread partition; KK must find it."""
    seqlen_list = [7] * 12
    partitions = karmarkar_karp(seqlen_list, k_partitions=4, equal_size=True)
    assert _spread(seqlen_list, partitions) == 0
