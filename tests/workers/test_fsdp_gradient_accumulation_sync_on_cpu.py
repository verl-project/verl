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

from contextlib import contextmanager, nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _FSDP1Module:
    def __init__(self):
        self.events = []

    @contextmanager
    def no_sync(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


class _FSDP2Module:
    def __init__(self):
        self.events = []

    def set_requires_gradient_sync(self, enabled):
        self.events.append(enabled)


def _make_engine(module):
    engine = object.__new__(FSDPEngine)
    engine.module = module
    return engine


@pytest.mark.parametrize("version,module_cls", [(1, _FSDP1Module), (2, _FSDP2Module)])
def test_gradient_sync_context_skips_non_final_micro_batch(monkeypatch, version, module_cls):
    module = module_cls()
    engine = _make_engine(module)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: version)

    with engine._gradient_sync_context(is_last_micro_batch=False):
        module.events.append("backward")

    expected = ["enter", "backward", "exit"] if version == 1 else [False, "backward", True]
    assert module.events == expected


def test_gradient_sync_context_restores_fsdp2_after_error(monkeypatch):
    module = _FSDP2Module()
    engine = _make_engine(module)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)

    with pytest.raises(RuntimeError, match="backward failed"):
        with engine._gradient_sync_context(is_last_micro_batch=False):
            raise RuntimeError("backward failed")

    assert module.events == [False, True]


def test_gradient_sync_context_keeps_sync_for_final_micro_batch(monkeypatch):
    module = _FSDP2Module()
    engine = _make_engine(module)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)

    with engine._gradient_sync_context(is_last_micro_batch=True):
        module.events.append("backward")

    assert module.events == ["backward"]


def test_forward_backward_batch_syncs_only_final_micro_batch(monkeypatch):
    engine = _make_engine(_FSDP2Module())
    engine.ulysses_sequence_parallel_size = 1
    engine.scaler = None
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1
    sync_states = []

    @contextmanager
    def record_sync(*, is_last_micro_batch):
        sync_states.append(is_last_micro_batch)
        yield

    engine._gradient_sync_context = record_sync
    engine.forward_step = lambda micro_batch, loss_function, forward_only: (
        micro_batch["loss"].sum(),
        {"metrics": {}},
    )

    micro_batches = [
        TensorDict({"loss": torch.tensor([float(i)], requires_grad=True)}, batch_size=[1]) for i in range(3)
    ]
    monkeypatch.setattr(
        transformer_impl,
        "prepare_micro_batches",
        lambda **_: (micro_batches, None),
    )
    monkeypatch.setattr(
        transformer_impl,
        "postprocess_batch_func",
        lambda output_lst, indices, data: output_lst,
    )
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    data = TensorDict({"loss_mask": torch.ones(3)}, batch_size=[3])
    output = engine.forward_backward_batch(data, loss_function=lambda: None)

    assert sync_states == [False, False, True]
    assert len(output) == 3
    assert all(micro_batch["loss"].grad is not None for micro_batch in micro_batches)


def _build_distributed_model(strategy, mesh):
    torch.manual_seed(2026)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    if strategy == "fsdp":
        return FSDP(
            model,
            device_id=torch.device("cpu"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
    fully_shard(model, mesh=mesh)
    return model


def _run_distributed_step(model, inputs, targets, strategy, defer_sync):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    engine = _make_engine(model)
    for micro_batch_idx, (inputs_micro_batch, targets_micro_batch) in enumerate(zip(inputs, targets, strict=True)):
        if defer_sync:
            is_last_micro_batch = micro_batch_idx == len(inputs) - 1
            sync_ctx = engine._gradient_sync_context(is_last_micro_batch=is_last_micro_batch)
        else:
            # Baseline: synchronize gradients on every micro-batch (pre-optimization behavior).
            sync_ctx = nullcontext()
        with sync_ctx:
            output = model(inputs_micro_batch)
            torch.nn.functional.mse_loss(output, targets_micro_batch, reduction="sum").backward()
    optimizer.step()

    if strategy == "fsdp":
        with FSDP.summon_full_params(model):
            return [parameter.detach().clone() for parameter in model.parameters()]
    return [parameter.full_tensor().detach().clone() for parameter in model.parameters()]


def _distributed_equivalence_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        inputs = (torch.arange(16, dtype=torch.float32).view(4, 4) + rank) / 10
        targets = torch.arange(8, dtype=torch.float32).view(4, 2) / 7
        input_micro_batches = inputs.chunk(2)
        target_micro_batches = targets.chunk(2)

        for strategy in ("fsdp", "fsdp2"):
            baseline = _build_distributed_model(strategy, mesh)
            optimized = _build_distributed_model(strategy, mesh)
            baseline_parameters = _run_distributed_step(
                baseline,
                input_micro_batches,
                target_micro_batches,
                strategy,
                defer_sync=False,
            )
            optimized_parameters = _run_distributed_step(
                optimized,
                input_micro_batches,
                target_micro_batches,
                strategy,
                defer_sync=True,
            )

            for baseline_parameter, optimized_parameter in zip(baseline_parameters, optimized_parameters, strict=True):
                torch.testing.assert_close(baseline_parameter, optimized_parameter, rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_distributed_accumulation_matches_per_micro_batch_sync(tmp_path):
    world_size = 2
    rendezvous_file = str(tmp_path / "fsdp_rdzv")
    mp.spawn(
        _distributed_equivalence_worker,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )
