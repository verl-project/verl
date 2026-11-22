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

"""
Tests for prepare_micro_batches with force_group_size > 1 and use_dynamic_bsz=True.

Focuses on verifying that:
1. Samples within the same group (consecutive force_group_size samples) always
   end up in the same micro-batch.
2. All original samples are covered exactly once across all micro-batches.
3. The returned batch_idx_list correctly maps micro-batch positions back to
   original batch positions.
4. Token budget (max_token_len) is respected per micro-batch.
"""

from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from verl.workers.engine.utils import prepare_micro_batches


def _make_batch(seq_lens: list[int], force_group_size: int, max_token_len_per_gpu: int) -> TensorDict:
    """Build a minimal TensorDict accepted by prepare_micro_batches.

    Args:
        seq_lens: Effective sequence length for each sample.
        force_group_size: Group size constraint to embed in the batch.
        max_token_len_per_gpu: Token budget per GPU to embed in the batch.

    Returns:
        A TensorDict with ``input_ids``, ``attention_mask``, and the required
        non-tensor metadata fields.
    """
    batch_size = len(seq_lens)
    max_len = max(seq_lens)

    # Build padded attention_mask: each row has seq_lens[i] ones followed by zeros.
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, sl in enumerate(seq_lens):
        attention_mask[i, :sl] = 1

    input_ids = torch.randint(1, 100, (batch_size, max_len))

    batch = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        batch_size=[batch_size],
    )

    # Embed metadata that prepare_micro_batches reads via get_non_tensor_data.
    tu.assign_non_tensor_data(batch, "use_dynamic_bsz", True)
    tu.assign_non_tensor_data(batch, "sp_size", 1)
    tu.assign_non_tensor_data(batch, "force_group_size", force_group_size)
    tu.assign_non_tensor_data(batch, "max_token_len_per_gpu", max_token_len_per_gpu)

    return batch


def _verify_group_integrity(batch_idx_list: list[list[int]], force_group_size: int, batch_size: int):
    """Assert that every group of force_group_size consecutive samples stays together.

    Args:
        batch_idx_list: Index lists returned by prepare_micro_batches.
        force_group_size: Expected group size.
        batch_size: Total number of samples in the original batch.
    """
    # Build a mapping: original_sample_idx -> micro_batch_id
    sample_to_mb = {}
    for mb_id, indices in enumerate(batch_idx_list):
        for idx in indices:
            assert idx not in sample_to_mb, f"Sample {idx} appears in multiple micro-batches"
            sample_to_mb[idx] = mb_id

    # Every sample must be assigned.
    assert set(sample_to_mb.keys()) == set(range(batch_size)), (
        f"Not all samples covered. Missing: {set(range(batch_size)) - set(sample_to_mb.keys())}"
    )

    # Samples within the same group must share the same micro-batch.
    num_groups = batch_size // force_group_size
    for g in range(num_groups):
        start = g * force_group_size
        group_indices = list(range(start, start + force_group_size))
        mb_ids = {sample_to_mb[i] for i in group_indices}
        assert len(mb_ids) == 1, f"Group {g} (samples {group_indices}) was split across micro-batches {mb_ids}"


def test_force_group_size_2_basic():
    """Basic test: batch_size=8, force_group_size=2, dynamic bsz enabled."""
    # 4 groups of 2; alternating short/long sequences within each group.
    seq_lens = [50, 60, 80, 90, 40, 45, 100, 110]
    force_group_size = 2
    batch_size = len(seq_lens)
    max_token_len_per_gpu = 220

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None, "batch_idx_list must not be None when use_dynamic_bsz=True"
    assert len(micro_batches) > 0

    _verify_group_integrity(batch_idx_list, force_group_size, batch_size)


def test_force_group_size_4_basic():
    """Test with force_group_size=4 (e.g., 4 responses per prompt in RM training)."""
    # 4 groups of 4 samples each.
    seq_lens = [
        100,
        110,
        90,
        95,  # group 0
        200,
        210,
        190,
        205,  # group 1
        50,
        55,
        45,
        60,  # group 2
        150,
        160,
        140,
        155,  # group 3
    ]
    force_group_size = 4
    batch_size = len(seq_lens)
    max_token_len_per_gpu = 850

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None
    assert len(micro_batches) > 0

    _verify_group_integrity(batch_idx_list, force_group_size, batch_size)


def test_force_group_size_reconstruction():
    """Verify that micro-batches can be reconstructed back to the original batch order."""
    seq_lens = [80, 85, 120, 130, 60, 65, 200, 210]
    force_group_size = 2
    max_token_len_per_gpu = 420

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None

    # Flatten micro-batches and index lists.
    flat_input_ids = torch.cat([mb["input_ids"] for mb in micro_batches], dim=0)
    flat_indices = [idx for indices in batch_idx_list for idx in indices]

    # Build reverse mapping and reconstruct.
    reverse_idx = [0] * len(flat_indices)
    for new_pos, orig_pos in enumerate(flat_indices):
        reverse_idx[orig_pos] = new_pos

    reconstructed = flat_input_ids[torch.tensor(reverse_idx)]
    torch.testing.assert_close(reconstructed, batch["input_ids"])


def test_force_group_size_single_micro_batch():
    """When all samples fit in one micro-batch, grouping constraint is trivially satisfied."""
    seq_lens = [10, 12, 15, 11, 8, 9, 14, 13]
    force_group_size = 2
    max_token_len_per_gpu = 10000  # very large budget

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None
    # All samples should be in a single micro-batch.
    assert len(micro_batches) == 1
    assert len(batch_idx_list[0]) == len(seq_lens)

    _verify_group_integrity(batch_idx_list, force_group_size, len(seq_lens))


def test_force_group_size_large_group():
    """Test with a larger batch and force_group_size=3."""
    # 6 groups of 3 samples each.
    seq_lens = [
        100,
        105,
        95,  # group 0
        200,
        205,
        195,  # group 1
        50,
        55,
        45,  # group 2
        150,
        155,
        145,  # group 3
        80,
        85,
        75,  # group 4
        120,
        125,
        115,  # group 5
    ]
    force_group_size = 3
    batch_size = len(seq_lens)
    max_token_len_per_gpu = 650

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None
    assert len(micro_batches) > 0

    _verify_group_integrity(batch_idx_list, force_group_size, batch_size)


def test_force_group_size_1_unchanged():
    """force_group_size=1 should behave identically to the default (no grouping constraint)."""
    seq_lens = [100, 200, 50, 150, 80, 120]
    force_group_size = 1
    max_token_len_per_gpu = 300

    batch = _make_batch(seq_lens, force_group_size, max_token_len_per_gpu)
    micro_batches, batch_idx_list = prepare_micro_batches(batch)

    assert batch_idx_list is not None
    assert len(micro_batches) > 0

    # All samples covered exactly once.
    all_indices = [idx for indices in batch_idx_list for idx in indices]
    assert sorted(all_indices) == list(range(len(seq_lens)))


def test_force_group_exceeding_token_limit_raises():
    batch = _make_batch(seq_lens=[200, 210], force_group_size=2, max_token_len_per_gpu=300)

    with pytest.raises(ValueError, match="forced group exceeds max_token_len"):
        prepare_micro_batches(batch)


def test_nested_dense_token_budget():
    seq_lens = [63, 96] + [2] * 6
    ids = torch.nested.as_nested_tensor([torch.full((n,), i) for i, n in enumerate(seq_lens)], layout=torch.jagged)
    batch = tu.get_tensordict({"input_ids": ids}, {"max_token_len_per_gpu": 200})
    assert len(prepare_micro_batches(batch)[0]) == 1  # Effective tokens fit, but the dense batch does not.
    micro_batches, indices = prepare_micro_batches(batch, use_dense_token_budget=True)
    assert len(micro_batches) == 2
    for mb, partition in zip(micro_batches, indices, strict=True):
        assert len(partition) * max(seq_lens[i] for i in partition) <= 200
        assert all(torch.equal(row, ids[i]) for row, i in zip(mb["input_ids"].unbind(), partition, strict=True))


def test_static_chunking_skips_token_budget_selection():
    ids = torch.nested.as_nested_tensor([torch.arange(n) for n in [8, 8, 1, 1]], layout=torch.jagged)
    batch = tu.get_tensordict({"input_ids": ids}, {"use_dynamic_bsz": False, "micro_batch_size_per_gpu": 2})
    micro_batches, indices = prepare_micro_batches(
        batch, use_dense_token_budget=lambda: pytest.fail("static budget lookup")
    )

    assert indices is None
    assert [len(micro_batch) for micro_batch in micro_batches] == [2, 2]


@pytest.mark.parametrize(
    "group_lengths,group_size,budget_per_group,options,expected_count",
    [
        ([100, 100, 80, 45, 45, 45], 1, 140, {"num_batches_divided_by": 4}, 4),
        ([40, 40, 40, 60, 60, 100, 100], 2, 120, {"num_batches_divided_by": 4}, 4),
        ([5, 5, 40, 40, 60, 80], 1, 160, {"min_num_micro_batch": 3}, 3),
        ([4, 0, 0], 1, 4, {}, 2),
        ([0, 0], 1, 4, {}, 1),
    ],
    ids=["vpp-dense-overflow", "vpp-effective-overflow", "requested-count", "trailing-zero", "all-zero"],
)
def test_dense_budget_partitions(group_lengths, group_size, budget_per_group, options, expected_count):
    seq_lens = [length for length in group_lengths for _ in range(group_size)]
    max_token_len = budget_per_group * group_size
    batch = _make_batch(seq_lens, group_size, max_token_len)

    _, indices = prepare_micro_batches(batch, use_dense_token_budget=True, **options)

    assert len(indices) == expected_count
    _verify_group_integrity(indices, group_size, len(seq_lens))
    assert all(len(partition) * max(seq_lens[i] for i in partition) <= max_token_len for partition in indices)


def test_dense_budget_rejects_infeasible_vpp_partition():
    batch = _make_batch([100] * 5 + [10], force_group_size=1, max_token_len_per_gpu=140)

    with pytest.raises(ValueError, match="Cannot satisfy max_token_len"):
        prepare_micro_batches(batch, num_batches_divided_by=4, use_dense_token_budget=True)


def test_dense_lower_bound_skips_impossible_counts():
    batch = _make_batch([60, 1, 50, 1, 40, 1, 20, 1], force_group_size=2, max_token_len_per_gpu=200)
    with patch(
        "verl.utils.seqlen_balancing.get_seqlen_balanced_partitions", wraps=get_seqlen_balanced_partitions
    ) as partitioner:
        _, indices = prepare_micro_batches(batch, use_dense_token_budget=True)
    partitioner.assert_called_once()
    assert len(indices) == 3


def test_force_group_exceeding_dense_token_limit_raises():
    batch = _make_batch(seq_lens=[100, 160], force_group_size=2, max_token_len_per_gpu=300)

    with pytest.raises(ValueError, match="forced group exceeds max_token_len"):
        prepare_micro_batches(batch, use_dense_token_budget=True)
