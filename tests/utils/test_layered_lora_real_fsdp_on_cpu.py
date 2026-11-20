# Copyright 2026 Individual Contributor
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
"""Two-rank native CPU/Gloo coverage for layered FSDP1 LoRA collection.

FSDP, lazy initialization, PEFT filtering, sharding, and collectives are real.
Only the unrelated device-cache helper is replaced with a CPU no-op. This
test checks export correctness and storage ownership, not GPU peak memory.
"""

import os
import platform
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.multiprocessing.spawn import ProcessRaisedException


class _TinyLoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 4, bias=False)

    def forward(self, inputs):
        return self.q_proj(inputs)


def _assert_export(actual, expected):
    assert set(actual) == set(expected)
    assert all("_flat_param" not in name and "_fsdp_wrapped_module" not in name for name in actual)
    for name, tensor in actual.items():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.shape == expected[name].shape
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def _collect_on_cpu_rank(rank, rendezvous):
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict

    from verl.utils import fsdp_utils

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    original_device_helper = fsdp_utils.get_torch_device
    fsdp_utils.get_torch_device = lambda: SimpleNamespace(empty_cache=lambda: None)
    try:
        torch.manual_seed(1234)
        model = get_peft_model(_TinyLoraModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
        with torch.no_grad():
            for index, (name, tensor) in enumerate(model.named_parameters(), start=1):
                if "lora_" in name:
                    tensor.copy_(torch.arange(tensor.numel(), dtype=tensor.dtype).reshape_as(tensor) + index)
        expected = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
        assert set(expected) == {
            "base_model.model.q_proj.lora_A.weight",
            "base_model.model.q_proj.lora_B.weight",
        }

        fsdp_options = {"device_id": torch.device("cpu"), "use_orig_params": False, "sync_module_states": False}
        adapter = model.base_model.model.q_proj
        # Keep frozen and trainable parameters in separate real FSDP units.
        # The parent owns B while the nested adapter leaf owns A. Gathering
        # that parent's named_parameters() also exercises FSDP's real removal
        # of wrapper segments, which differs from its pre-summon namespace.
        adapter.base_layer = FSDP(adapter.base_layer, **fsdp_options)
        adapter.lora_A["default"] = FSDP(adapter.lora_A["default"], **fsdp_options)
        parent = FSDP(adapter, **fsdp_options)
        model.base_model.model.q_proj = parent
        root = FSDP(model, **fsdp_options)
        assert root._is_root is None
        assert parent._is_root is None

        actual = fsdp_utils.layered_summon_lora_params(root)

        _assert_export(actual, expected)
        assert root._is_root is True
        for _, unit in root.named_modules():
            if isinstance(unit, FSDP) and unit is not root:
                assert unit._is_root is False

        # Every live parameter is now a local flat shard. The returned full
        # adapter must remain valid after reshards and later parameter writes.
        live_parameters = list(root.parameters())
        assert live_parameters
        live_storage = {tensor.untyped_storage().data_ptr() for tensor in live_parameters}
        for tensor in actual.values():
            assert tensor.untyped_storage().data_ptr() not in live_storage
        with torch.no_grad():
            for tensor in live_parameters:
                tensor.add_(1000)
        _assert_export(actual, expected)
    finally:
        fsdp_utils.get_torch_device = original_device_helper
        dist.destroy_process_group()


def _stop_workers(context):
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    for error_file in getattr(context, "error_files", ()):
        if os.path.exists(error_file):
            os.unlink(error_file)


def test_layered_lora_native_two_rank_cpu_gloo(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("native CPU FSDP test requires the Gloo process-group backend")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if platform.system() == "Darwin" else "lo")
    rendezvous = str(tmp_path / "gloo_filestore")
    context = mp.spawn(_collect_on_cpu_rank, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("two-rank CPU/Gloo LoRA collection exceeded its 120-second timeout")
    except ProcessRaisedException as error:
        # Some PyTorch builds disallow explicit-CPU FSDP. Do not turn an export,
        # namespace, PEFT-version, or collective failure into an environment skip.
        unsupported_cpu_messages = (
            "FSDP needs a non-CPU accelerator device",
            "FSDP does not support CPU only execution",
        )
        if any(message in str(error) for message in unsupported_cpu_messages):
            pytest.skip("this PyTorch build does not support native CPU FSDP: " + str(error))
        raise
    finally:
        _stop_workers(context)
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)
