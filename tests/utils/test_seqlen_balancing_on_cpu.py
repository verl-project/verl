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

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import rearrange_micro_batches


def _uneven_effective_length_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=5),
    )
    try:
        effective_length = (16, 21)[rank]
        input_ids = torch.arange(128).reshape(1, 128)
        attention_mask = torch.zeros((1, 128), dtype=torch.long)
        attention_mask[:, -effective_length:] = 1
        batch = DataProto.from_single_dict({"input_ids": input_ids, "attention_mask": attention_mask}).batch
        tu.assign_non_tensor_data(batch, "use_remove_padding", True)

        with (
            patch("verl.utils.seqlen_balancing.get_device_name", return_value="cpu"),
            pytest.raises(AssertionError, match="max_seq_len=21"),
        ):
            rearrange_micro_batches(batch, max_token_len=20, dp_group=dist.group.WORLD, same_micro_num_in_dp=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_effective_length_guard_is_synchronized_across_dp_ranks(tmp_path):
    world_size = 2
    rendezvous_file = str(tmp_path / "seqlen_balancing_rdzv")
    mp.spawn(
        _uneven_effective_length_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )
