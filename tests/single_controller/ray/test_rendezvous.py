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

import importlib.util

import pytest
import ray
import torch


@ray.remote
class RendezvousWorker:
    def __init__(self, rank: int, world_size: int, group_name: str) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.communicator = None

    def init(self) -> None:
        from verl.utils.rendezvous.ray_backend import create_nccl_communicator_in_ray

        self.communicator = create_nccl_communicator_in_ray(self.rank, self.world_size, self.group_name)

    def communicator_rank(self) -> int:
        return self.communicator.rank_id()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or importlib.util.find_spec("cupy") is None,
    reason="requires two CUDA devices and CuPy",
)
def test_legacy_rendezvous_path_creates_ranked_communicators():
    ray.init()
    try:
        workers = [RendezvousWorker.options(num_gpus=1).remote(rank, 2, "test_group") for rank in range(2)]
        ray.get([worker.init.remote() for worker in workers])
        assert ray.get([worker.communicator_rank.remote() for worker in workers]) == [0, 1]
    finally:
        ray.shutdown()
