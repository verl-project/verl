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

"""Regression test for https://github.com/verl-project/verl/issues/6259.

``FSDPModelMerger._merge_by_placement`` handles a ``Replicate()`` placement by returning
the shard-0 ``_local_tensor`` as-is. That local tensor is loaded straight from a
checkpoint file and can be a *view* into a storage shared with some other tensor from the
same checkpoint (e.g. if the checkpoint writer packed multiple parameters into one
contiguous allocation). Returning it unmodified lets that storage aliasing leak into the
merged state_dict, so two unrelated keys can end up sharing one underlying storage --
independently of whatever ``transformers.save_pretrained`` does with that at write time,
this is not a state the merged state_dict should be in. The fix clones the tensor so every
key in the merged state_dict always owns independent storage.

This uses real ``torch.distributed.tensor`` placements and plain CPU tensors -- no
mocking of ``verl``, ``torch``, or ``transformers`` internals, and no GPU / process group
is needed since ``_merge_by_placement`` is a pure function of its arguments.
"""

import torch
from torch.distributed.tensor import Replicate, Shard

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _merger() -> FSDPModelMerger:
    # `_merge_by_placement` doesn't touch `self`, so a bare, un-`__init__`-ed instance
    # (same pattern as `_TestModelMerger` in test_output_validation_on_cpu.py) is enough.
    return FSDPModelMerger.__new__(FSDPModelMerger)


def test_replicate_placement_clones_to_independent_storage():
    """The historical bug: two keys sharing one FSDP shard's storage end up aliased."""
    merger = _merger()
    # One buffer standing in for a checkpoint shard that packed two parameters together;
    # each "parameter" is a disjoint, non-overlapping view into it.
    shared_buffer = torch.arange(8, dtype=torch.float32)
    tensors_for_key_a = [shared_buffer[0:4]]
    tensors_for_key_b = [shared_buffer[4:8]]

    merged_a = merger._merge_by_placement(tensors_for_key_a, Replicate())
    merged_b = merger._merge_by_placement(tensors_for_key_b, Replicate())

    assert torch.equal(merged_a, shared_buffer[0:4])
    assert torch.equal(merged_b, shared_buffer[4:8])
    # The fix: cloning gives each key independent storage, so the merged state_dict never
    # depends on two distinct keys resolving to non-overlapping (or, in a more degenerate
    # case, identical) regions of one shared allocation.
    assert merged_a.untyped_storage().data_ptr() != merged_b.untyped_storage().data_ptr()
    assert merged_a.untyped_storage().data_ptr() != shared_buffer.untyped_storage().data_ptr()


def test_replicate_placement_preserves_values_and_dtype():
    merger = _merger()
    original = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)

    merged = merger._merge_by_placement([original], Replicate())

    assert torch.equal(merged, original)
    assert merged.dtype == original.dtype
    assert merged.shape == original.shape


def test_shard_placement_still_concatenates_without_cloning_behavior_change():
    """Non-regression: only the replicate path changed; sharded merging is untouched."""
    merger = _merger()
    shard0 = torch.tensor([1.0, 2.0])
    shard1 = torch.tensor([3.0, 4.0])

    merged = merger._merge_by_placement([shard0, shard1], Shard(0))

    assert torch.equal(merged, torch.tensor([1.0, 2.0, 3.0, 4.0]))
