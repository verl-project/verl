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

    # 4) check the enforced counts
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    local = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))

    if min_mb is not None:
        expected = max(local, min_mb)
        assert len(micros) == expected
    if use_same_dp:
        # gather all local_counts
        counts = [torch.zeros(1, device=f"{get_device_name()}:{rank}") for _ in range(world_size)]
        counts[rank].fill_(local)
        dist.all_gather(counts, counts[rank])
        expected = max(int(c.item()) for c in counts)
        assert len(micros) == expected
    else:
        # if neither, we get the local natural count
        assert len(micros) == local

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
