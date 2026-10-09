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
"""
In client, we can get the server handler and send RPC request
"""

import ray
import torch
from server import Trainer
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


if __name__ == "__main__":
    ray.init(address="auto", namespace="verl")
    # get the worker group using names
    worker_names = ["trainerTrainer_0:0", "trainerTrainer_0:1"]
    cls_with_init_args = RayClassWithInitArgs(cls=Trainer)
    worker_group = RayWorkerGroup.from_detached(worker_names=worker_names, ray_cls_with_init=cls_with_init_args)

    batch_size = 16
    sequence_length = 1024

    # give Trainer some data to train
    input_ids = torch.randint(low=0, high=256, size=(batch_size, sequence_length), dtype=torch.int64, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    position_ids = compute_position_id_with_mask(attention_mask)

    data = DataProto(
        batch=TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            batch_size=batch_size,
        ),
        meta_info={},
    )

    output = worker_group.train_model(data)

    print(output)
