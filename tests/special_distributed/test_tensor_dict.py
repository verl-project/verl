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

import os

os.environ["NCCL_DEBUG"] = "WARN"

import numpy as np
import torch
import torch.distributed

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group


def test_all_gather_data_proto():
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        get_device_name(), mesh_shape=[2, 2], mesh_dim_names=["dp", "tp"]
    )

    global_rank = torch.distributed.get_rank()

    obs = torch.tensor([[1 * global_rank, 2 * global_rank + 1], [3 * global_rank, 4 * global_rank + 1]])

    labels = ["a", "b"] if global_rank % 2 == 0 else ["b", "a"]
    labels = np.array(labels, dtype=object)
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    all_gather_data_proto(data=data, process_group=device_mesh.get_group("dp"))

    if global_rank == 0:
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device=get_device_name())
        expected_labels = ["a", "b", "a", "b"]
    elif global_rank == 1:
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device=get_device_name())
        expected_labels = ["b", "a", "b", "a"]
    elif global_rank == 2:
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device=get_device_name())
        expected_labels = ["a", "b", "a", "b"]
    elif global_rank == 3:
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device=get_device_name())
        expected_labels = ["b", "a", "b", "a"]

    torch.testing.assert_close(data.batch["obs"], expected_obs, atol=0, rtol=0)
    assert (data.non_tensor_batch["labels"] == expected_labels).all()
    assert data.meta_info == {"info": "test_info"}


def test_vocab_parallel_entropy():
    from megatron.core import parallel_state as mpu

    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
    from verl.utils.profiler import log_gpu_memory_usage
    from verl.utils.torch_functional import entropy_from_logits

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None
    )

    batch_size = 2
    seqlen = 128
    vocab_size = 155136

    logits = torch.randn(batch_size * seqlen, vocab_size, device=get_device_name(), requires_grad=True)
    target = torch.randint(
        low=0, high=vocab_size, size=(batch_size * seqlen,), device=get_device_name(), dtype=torch.int64
    )

    # broadcast across tp
    torch.distributed.broadcast(
        logits, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
    )
    torch.distributed.broadcast(
        target, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
    )

    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_size_per_tp = vocab_size // mpu.get_tensor_model_parallel_world_size()

    # get the local logits of each tp
    vocab_parallel_logits = (
        logits.clone().detach()[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp].requires_grad_()
    )
    logits.grad = None
    vocab_parallel_logits.grad = None

    log_gpu_memory_usage("begin")
    output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
    log_gpu_memory_usage("after forward")
    grad_output = torch.randn_like(output_entropy)
    output_entropy.backward(grad_output)
    log_gpu_memory_usage("after backward")

    target_entropy = entropy_from_logits(logits)
    torch.testing.assert_close(output_entropy, target_entropy)
    target_entropy.backward(grad_output)
    torch.testing.assert_close(
        logits.grad[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits.grad
    )
    # make sure logits is not altered
    torch.testing.assert_close(
        logits[:, tp_rank * vocab_size_per_tp : (tp_rank + 1) * vocab_size_per_tp], vocab_parallel_logits
    )

    if mpu.get_tensor_model_parallel_rank() == 0:
        print("test_vocab_parallel_entropy passes")

    mpu.destroy_model_parallel()


if __name__ == "__main__":
    local_rank, rank, world_size = initialize_global_process_group()
    test_all_gather_data_proto()
    test_vocab_parallel_entropy()
