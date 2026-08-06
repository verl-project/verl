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

import os

import torch

from verl.utils.torch_functional import allgather_dict_into_dict

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    metrics_dict = {"loss": [0 + rank, 1 + rank, 2 + rank], "grad_norm": rank}

    result = allgather_dict_into_dict(data=metrics_dict, group=None)

    assert result["loss"] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert result["grad_norm"] == [0, 1, 2, 3]

    print(result)
