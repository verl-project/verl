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

"""Exercise real sharded Adam states through FSDPEngine's automatic train context.

Launch with torchrun --standalone --nproc-per-node=2
    tests/special_distributed/test_fsdp2_optimizer_offload_step.py
"""

import io
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def build_engine(mesh, enabled, param_offload, foreach, fused):
    torch.manual_seed(5282)
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 8))
    engine.module.to(get_device_name())
    fully_shard(engine.module, mesh=mesh)
    assert all(isinstance(p, DTensor) for p in engine.module.parameters())
    engine.optimizer = torch.optim.AdamW(engine.module.parameters(), lr=0.01, foreach=foreach, fused=fused)
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.engine_config = FSDPEngineConfig(
        strategy="fsdp2", optimizer_offload=True, optimizer_offload_step=enabled, param_offload=param_offload
    )
    engine._is_offload_param = param_offload
    engine._is_offload_optimizer = True
    engine._qat_enabled = False
    engine.scaler = None
    engine.mode = None
    engine.ulysses_parallel_group = None
    engine.to(device="cpu", model=param_offload, optimizer=True, grad=param_offload)
    return engine


def snapshot(engine):
    # Every rank checks its entire parameter and optimizer shards. The collective
    # exit barrier makes a failure on either rank fail the distributed test.
    get_torch_device().synchronize()

    def local_copy(tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor.detach().cpu().clone()

    result = {f"param/{name}": local_copy(p) for name, p in engine.module.named_parameters()}
    for index, state in enumerate(engine.optimizer.state.values()):
        for name, value in state.items():
            result[f"optim/{index}/{name}"] = local_copy(value) if torch.is_tensor(value) else value
    return result


def assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def assert_state_device(engine, device):
    for state in engine.optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            assert isinstance(state[name], DTensor), "test must cover actual sharded optimizer states"
            assert state[name].to_local().device.type == device


def update(engine, index, nonfinite=False, nested=False):
    with engine.train_mode():
        if engine.optimizer.state:
            assert_state_device(engine, "cpu" if engine.engine_config.optimizer_offload_step else get_device_name())
        with engine.train_mode(disable_auto_offload=True) if nested else nullcontext():
            generator = torch.Generator(device=get_device_name()).manual_seed(1000 + index + dist.get_rank())
            # Two backward calls verify that state residency covers gradient accumulation.
            for _ in range(2):
                inputs = torch.randn(4, 32, device=get_device_name(), generator=generator)
                engine.module(inputs).square().mean().div(2).backward()
            if nonfinite and dist.get_rank() == 0:
                next(engine.module.parameters()).grad.to_local().fill_(float("inf"))
            grad_norm = engine.optimizer_step()
            assert torch.isfinite(torch.tensor(grad_norm)).item() is not nonfinite
            assert_state_device(engine, "cpu" if engine.engine_config.optimizer_offload_step else get_device_name())
    assert_state_device(engine, "cpu")
    assert engine._optimizer_offload_step is False
    assert all(p.grad is None for p in engine.module.parameters())
    return snapshot(engine)


def round_trip(engine, enabled):
    # Serialize real DTensor optimizer state, then switch the option on resume.
    buffer = io.BytesIO()
    torch.save(engine.optimizer.state_dict(), buffer)
    buffer.seek(0)
    engine.optimizer.load_state_dict(torch.load(buffer, weights_only=False))
    engine.to(device="cpu", model=False, optimizer=True, grad=False)
    engine.engine_config = replace(engine.engine_config, optimizer_offload_step=enabled)


def main():
    _, rank, world_size = initialize_global_process_group()
    mesh = init_device_mesh(get_device_name(), (world_size,), mesh_dim_names=("dp",))
    for param_offload in (False, True):
        for foreach, fused in ((False, False), (True, False), (False, True)):
            baseline = build_engine(mesh, False, param_offload, foreach, fused)
            offloaded = build_engine(mesh, True, param_offload, foreach, fused)
            for index in range(4):
                assert_equal(update(offloaded, index), update(baseline, index))
            before = snapshot(offloaded)
            assert_equal(update(offloaded, 4, nonfinite=True), before)
            # Both directions resume against an uninterrupted baseline.
            round_trip(offloaded, False)
            assert_equal(update(offloaded, 5), update(baseline, 5))
            round_trip(offloaded, True)
            assert_equal(update(offloaded, 6), update(baseline, 6))
            assert_equal(update(offloaded, 7, nested=True), update(baseline, 7, nested=True))
            if rank == 0:
                print(
                    f"PASS world_size={world_size} param_offload={param_offload} foreach={foreach} fused={fused}",
                    flush=True,
                )
            del baseline, offloaded
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("test_fsdp2_optimizer_offload_step passed", flush=True)


if __name__ == "__main__":
    main()
