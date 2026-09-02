# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Regression test for non-blocking FSDP2 full-parameter model transfers.

PyTorch currently places the CPU tensors produced by a non-blocking CUDA-to-CPU
``Module.to`` in pinned memory. This is required for the following non-blocking
CPU-to-CUDA transfer, but is not strictly guaranteed by the public docs. Keep
that behavior covered here since the colocated actor moves the full-parameter
FSDP2 model around both old-log-prob and actor-update phases.

Launch:
    torchrun --nproc-per-node=2 --standalone \
        tests/special_distributed/test_fsdp2_pinned_model_transfer.py
"""

import torch
import torch.distributed
from torch.distributed import init_device_mesh
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    load_fsdp2_model_to_gpu,
    offload_fsdp2_model_to_cpu,
)


def _local_tensor(tensor):
    return tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor


def _build_full_parameter_fsdp2_model(device_mesh):
    config = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
    )
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16)
        model = model.to(device="cuda")

    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        ),
    }
    apply_fsdp2(model, fsdp_kwargs, {})
    assert all(param.requires_grad for param in model.parameters())
    return model


def main():
    if get_device_name() != "cuda":
        print("test_fsdp2_pinned_model_transfer skipped: pinned transfer behavior is CUDA-specific")
        return
    assert get_torch_device().device_count() >= 2, "need at least 2 GPUs for test"
    _, rank, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    model = _build_full_parameter_fsdp2_model(device_mesh)

    expected_local_params = [_local_tensor(param).detach().clone() for param in model.parameters()]

    offload_fsdp2_model_to_cpu(model, empty_cache=False)
    for param in model.parameters():
        local_param = _local_tensor(param)
        assert local_param.device.type == "cpu"
        assert local_param.is_pinned(), "non-blocking FSDP2 D2H copy must produce pinned CPU parameters"

    load_fsdp2_model_to_gpu(model)
    torch.cuda.synchronize()
    for param, expected in zip(model.parameters(), expected_local_params, strict=True):
        local_param = _local_tensor(param)
        assert local_param.device.type == "cuda"
        torch.testing.assert_close(local_param, expected, atol=0.0, rtol=0.0)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("test_fsdp2_pinned_model_transfer passed")


if __name__ == "__main__":
    main()
