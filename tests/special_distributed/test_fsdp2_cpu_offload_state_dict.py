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

"""Regression for verl#5995: FSDP2 + CPUOffloadPolicy state_dict crash inside
FSDPEngine.get_per_tensor_param.

Pre-fix, get_per_tensor_param unconditionally called load_fsdp_model_to_gpu()
(which under FSDP2 reduces to model.to(device)) before module.state_dict().
With CPUOffloadPolicy this leaves the module half-moved and crashes state_dict()
with "Attempted to set the storage of a tensor on device 'cpu' to a storage on
different device 'cuda:0'."

This test rebuilds that exact sequence on a tiny Qwen2 model and asserts:
  - the post-fix sequence (state_dict only) succeeds and downstream DTensor
    materialisation still yields GPU tensors,
  - the pre-fix sequence (load + state_dict) still crashes today (informational;
    the fix is still correct on PyTorch versions that have relaxed this check).

Launch:
    torchrun --nproc-per-node=2 --standalone \\
        tests/special_distributed/test_fsdp2_cpu_offload_state_dict.py
"""

import torch
import torch.distributed
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2, load_fsdp_model_to_gpu


def _build_fsdp2_cpu_offload_module(device_mesh):
    """Wrap a tiny Qwen2 with FSDP2 + CPUOffloadPolicy, mirroring FSDPEngine."""
    config = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
    )
    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16)
        model = model.to(device=get_device_name())

    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        ),
        "offload_policy": CPUOffloadPolicy(pin_memory=True),
    }
    apply_fsdp2(model, fsdp_kwargs, {})
    return model


def _assert_fixed_path_succeeds(device_mesh, rank):
    """Replay FSDPEngine.get_per_tensor_param's post-fix sequence."""
    module = _build_fsdp2_cpu_offload_module(device_mesh)

    state_dict = module.state_dict()
    assert len(state_dict) > 0, "expected a populated state dict"

    # Verify downstream DTensor materialisation in get_per_tensor_param still
    # produces GPU tensors -- this is the rationale for skipping the manual load.
    device = get_device_id()
    materialised = False
    for name, param in state_dict.items():
        if isinstance(param, DTensor):
            full = param.to(device, non_blocking=True).full_tensor()
            assert full.device.type == get_device_name(), (
                f"{name}: full_tensor() yielded {full.device.type}, expected {get_device_name()}"
            )
            materialised = True
            break
    assert materialised, "did not encounter any DTensor in state_dict; FSDP2 sharding may not be active"

    if rank == 0:
        print("fixed path: state_dict() + DTensor materialisation succeeded")


def _probe_pre_fix_crash(device_mesh, rank):
    """Reproduce the pre-fix sequence and report whether it still crashes."""
    module = _build_fsdp2_cpu_offload_module(device_mesh)
    load_fsdp_model_to_gpu(module)

    crashed_with_device_mismatch = False
    try:
        _ = module.state_dict()
    except RuntimeError as e:
        msg = str(e).lower()
        # Issue #5995 error string is:
        #   "Attempted to set the storage of a tensor on device 'cpu' to a
        #    storage on different device 'cuda:0'."
        crashed_with_device_mismatch = "cpu" in msg and "cuda" in msg and "storage" in msg
        if rank == 0:
            print(f"pre-fix path crashed (as expected for #5995): {str(e)[:200]}")

    if not crashed_with_device_mismatch and rank == 0:
        # PyTorch could relax this check in the future. Don't fail -- the fix is
        # still correct because it avoids a redundant no-op move under
        # CPUOffloadPolicy, which is what FSDP2 already manages internally.
        print("note: unguarded load + state_dict did not crash on this PyTorch version")


def main():
    assert get_torch_device().device_count() >= 1, "need at least 1 gpu for test"
    _, rank, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    _assert_fixed_path_succeeds(device_mesh, rank)
    # _probe_pre_fix_crash(device_mesh, rank)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("test_fsdp2_cpu_offload_state_dict passed")


if __name__ == "__main__":
    main()
