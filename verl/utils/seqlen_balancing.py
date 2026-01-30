# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import copy
import heapq
from itertools import chain

import torch
from torch import distributed as dist

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name


def calculate_workload(seqlen_list: torch.Tensor) -> torch.Tensor:
    """Calculate approximate computational workload for transformer attention.

    Estimates FLOPs for dense transformer blocks based on sequence length using
    the formula: FLOPs ≈ 12 * hidden_size² * seqlen + 2 * hidden_size * seqlen²

    The constants are calibrated for a 7B model (hidden_size=4096), yielding:
    workload ∝ 24576 * seqlen + seqlen²

    Args:
        seqlen_list: Sequence lengths as a tensor.

    Returns:
        torch.Tensor: Estimated workload values proportional to actual FLOPs.

    Note:
        The returned values are relative workloads, not actual FLOP counts.
        Useful for balancing computation across data parallel ranks.
    """
    return 24576 * seqlen_list + seqlen_list**2


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using the Karmarkar-Karp differencing method.

    Implements the Largest Differencing Method (LDM) algorithm for balanced
    multi-way number partitioning. This heuristic produces near-optimal partitions
    by iteratively combining the sets with the largest difference.

    Args:
        seqlen_list: Values to partition (typically sequence lengths or workloads).
        k_partitions: Number of partitions to create.
        equal_size: If True, each partition will have exactly len(seqlen_list) / k_partitions
            items. If False, partitions may have different sizes.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.

    See Also:
        https://en.wikipedia.org/wiki/Largest_differencing_method

    Note:
        When equal_size=True, len(seqlen_list) must be divisible by k_partitions.
    """

    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using a greedy assignment strategy.

    Assigns each item to the partition with the smallest current sum, iterating
    through items in order. Simpler but typically less optimal than Karmarkar-Karp.

    Args:
        seqlen_list: Values to partition (typically sequence lengths or workloads).
        k_partitions: Number of partitions to create.
        equal_size: If True, adds a bias to ensure equal partition sizes.
            Requires len(seqlen_list) to be divisible by k_partitions.

    Returns:
        list[list[int]]: List of k partitions, each containing indices into seqlen_list.

    Note:
        When equal_size=True, a large bias is added to encourage equal distribution
        of items before considering the actual values.
    """
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def ceildiv(a: int, b: int) -> int:
    """Compute ceiling division of a by b.

    Returns the smallest integer greater than or equal to a/b.
    Uses the identity: ceil(a/b) = floor((a + b - 1) / b) = -(-a // b)

    Args:
        a: Dividend (numerator).
        b: Divisor (denominator), must be non-zero.

    Returns:
        int: Ceiling of a divided by b.

    Example:
        >>> ceildiv(7, 3)  # ceil(7/3) = ceil(2.33) = 3
        3
        >>> ceildiv(6, 3)  # ceil(6/3) = ceil(2.0) = 2
        2
    """
    return -(a // -b)


def roundup_divisible(a: int, b: int) -> int:
    """Round up a to the nearest multiple of b.

    Returns the smallest multiple of b that is >= a.

    Args:
        a: Value to round up.
        b: Divisor to round to (must be positive).

    Returns:
        int: Smallest multiple of b that is >= a.

    Example:
        >>> roundup_divisible(7, 4)  # nearest multiple of 4 >= 7 is 8
        8
        >>> roundup_divisible(8, 4)  # 8 is already a multiple of 4
        8
    """
    return ((a + b - 1) // b) * b


def rearrange_micro_batches(
    batch,
    max_token_len,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    # this is per local micro_bsz
    input_ids = batch["input_ids"]
    if input_ids.is_nested:
        seq_len_effective: torch.Tensor = input_ids.offsets().diff()
        max_seq_len = max(seq_len_effective)
    else:
        max_seq_len = batch["attention_mask"].shape[-1]
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)

    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    assert num_micro_batches <= len(seq_len_effective)

    # upcast to int64 to avoid potential overflow im `calculate_workload` computation.
    seq_len_effective = seq_len_effective.long()
    # note that seq_len_effective is a GPU tensor. We need to make it a list to avoid D2H!
    workloads = calculate_workload(seq_len_effective).cpu().tolist()
    micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)

    if use_dynamic_bsz_balance:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(workloads[idx] for idx in partition),
                partition[0] if partition else 0,
            ),
            reverse=True,
        )
        # Place smaller micro-batches at both ends to reduce the bubbles exposed during the warm-up and cool-down.
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = tu.index_select_tensor_dict(batch, partition)
        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map


def prepare_dynamic_batch(
    data: DataProto,
    max_token_len: int,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
) -> tuple[list[DataProto], list[list[int]]]:
    """
    Prepare a batch for dynamic batching.

    Args:
        data (DataProto): The input data.
        max_token_len (int): The maximum token length for dynamic batching.

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
    """
    batch, batch_idx_list = rearrange_micro_batches(
        data.batch,
        max_token_len=max_token_len,
        dp_group=dp_group,
        num_batches_divided_by=num_batches_divided_by,
        same_micro_num_in_dp=same_micro_num_in_dp,
        min_num_micro_batch=min_num_micro_batch,
        use_dynamic_bsz_balance=use_dynamic_bsz_balance,
    )
    micro_batches = []
    for i, batch_idx in enumerate(batch_idx_list):
        tensors = dict(batch[i])
        non_tensors = {key: value[batch_idx] for key, value in data.non_tensor_batch.items()}
        meta_info = copy.deepcopy(data.meta_info)
        micro_batches.append(DataProto.from_dict(tensors, non_tensors, meta_info=meta_info))

    return micro_batches, batch_idx_list


def restore_dynamic_batch(data: torch.Tensor, batch_idx_list: list[list[int]]) -> torch.Tensor:
    """
    Restore a batch from dynamic batching.

    Args:
        data (torch.Tensor): The input data.
        batch_idx_list (List[List[int]]): The list of index lists.

    Returns:
        torch.Tensor: The restored data.
    """
    indices = list(chain.from_iterable(batch_idx_list))
    batch_size = data.shape[0]
    assert len(indices) == batch_size, f"{len(indices)} vs. {batch_size}"
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)

    if data.is_nested:
        data_lst = data.unbind()
        tensors = [data_lst[i] for i in revert_indices]
        reverted_data = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    else:
        reverted_data = data[revert_indices]

    return reverted_data


def get_group_balanced_partitions(
    seqlen_list: list[int],
    uid_list: list,
    k_partitions: int,
) -> list[list[int]]:
    """
    Partition samples into k groups while keeping samples with the same uid together.

    Args:
        seqlen_list: List of sequence lengths for each sample.
        uid_list: List of uids identifying which samples share the same prefix.
                  Samples with the same uid will be kept together.
        k_partitions: Number of partitions (typically world_size).

    Returns:
        List of k lists, each containing sample indices assigned to that partition.
        Samples with the same uid are guaranteed to be in the same partition.
    """
    assert len(seqlen_list) == len(uid_list), "seqlen_list and uid_list must have same length"

    # Build groups: each group contains indices of samples with the same uid
    # Assumes samples with same uid are contiguous
    groups = []  # List of (group_indices, group_total_seqlen)
    current_uid = None
    current_indices = []
    current_seqlen = 0

    for i, (seqlen, uid) in enumerate(zip(seqlen_list, uid_list, strict=False)):
        if uid != current_uid:
            if current_indices:
                groups.append((current_indices, current_seqlen))
            current_uid = uid
            current_indices = [i]
            current_seqlen = seqlen
        else:
            current_indices.append(i)
            current_seqlen += seqlen

    # Don't forget the last group
    if current_indices:
        groups.append((current_indices, current_seqlen))

    num_groups = len(groups)
    assert num_groups >= k_partitions, (
        f"Number of uid groups ({num_groups}) must be >= k_partitions ({k_partitions}). "
        f"Consider reducing world_size or increasing batch_size."
    )

    # Calculate workload for each group (as integers for partitioning)
    group_workloads = []
    for indices, total_seqlen in groups:
        # Use sum of individual workloads for more accurate estimation
        workload = sum(int(calculate_workload(torch.tensor([seqlen_list[i]])).item()) for i in indices)
        group_workloads.append(workload)

    # Use Karmarkar-Karp to partition groups
    # equal_size=True ensures each partition gets the same number of groups,
    # which is required when each group has the same number of samples (rollout.n)
    group_partitions = get_seqlen_balanced_partitions(
        seqlen_list=group_workloads,
        k_partitions=k_partitions,
        equal_size=True,
    )

    # Convert group partitions to sample partitions
    sample_partitions = []
    for group_partition in group_partitions:
        sample_indices = []
        for group_idx in group_partition:
            sample_indices.extend(groups[group_idx][0])
        sample_partitions.append(sorted(sample_indices))

    return sample_partitions
