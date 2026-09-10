from typing import Dict, List, Optional, Tuple


def grab_exact_from_heterogeneous_queue(
    queue: List[Dict[str, List]], batch_size: int
) -> Tuple[Optional[List], List]:
    """
    Grabs a batch of size batchsize from a queue of different sized items

    e.g. queue = [{"tokens": [[1, 2, 3],[4, 5, 6, 7, 8]]}, {"tokens": [[9, 10]]}]

    without going over the batchsize. This function will return a batch of size batchsize, and the new queue.

    Because all groups are a common denominator of the batchsize, and all groups are a power of 2,
    we can simplify a bit by assuming we can grab groups of groups to be equal to the maximum group size.
    Note that we cannot drop items from groups, so we must grab the entire group if we grab it.

    There may be a more efficient clearing mechanism by grouping these smaller groups heterogeneously, but
    forcing them all into powers of two groups is a simple way to ensure we can grab a batch of the correct size.

    :param queue:
    :param batch_size:
    :return: batch, new_queue
    """

    # Pass 1: precompute group sizes, total tokens and early exit if not enough tokens.
    total_groups = len(queue)
    if total_groups == 0:
        return None, queue

    group_sizes = []
    lengths = []
    total_tokens = 0
    max_group_size = 0

    for item in queue:
        length = len(item["tokens"])
        lengths.append(length)
        group_sizes.append(length)
        total_tokens += length
        if length > max_group_size:
            max_group_size = length

    if total_tokens < batch_size:
        return None, queue

    group_sizes_set = set(group_sizes)
    group_batching_storage = {size: [] for size in group_sizes_set}

    # Index into the queue and batch related indices into "packs"
    potential_batch_indices = []
    for i, group_size in enumerate(group_sizes):
        group_batching_storage[group_size].append(i)
        if len(group_batching_storage[group_size]) * group_size == max_group_size:
            potential_batch_indices.extend(group_batching_storage[group_size])
            group_batching_storage[group_size].clear()  # much faster than = []

    # Calculate total batch tokens only once (avoid repeated sums)
    potential_batch_token_total = sum(lengths[i] for i in potential_batch_indices)
    if potential_batch_token_total < batch_size:
        return None, queue

    # Batch selection
    batch = []
    batch_indices = []
    running_tokens = 0
    for idx in potential_batch_indices:
        group = queue[idx]
        batch.append(group)
        batch_indices.append(idx)
        running_tokens += lengths[idx]
        if running_tokens == batch_size:
            break
        elif running_tokens > batch_size:
            # Should never happen due to problem constraints, but sanity check
            return None, queue

    if running_tokens != batch_size:
        return None, queue

    # Construct new_queue with a single pass, using a set for O(1) lookup
    batch_indices_set = set(batch_indices)
    new_queue = [item for i, item in enumerate(queue) if i not in batch_indices_set]
    return batch, new_queue
