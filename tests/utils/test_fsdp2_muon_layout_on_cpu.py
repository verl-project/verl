# Copyright 2026 Individual Contributor: yue-zeng-yue
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
"""CPU DTensor layout/guard tests; these do not exercise FSDP2 fully_shard or CUDA NS."""

import copy
import io
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from verl.utils.fsdp2_muon import _parameter_mesh, build_fsdp2_muon, group_fsdp2_muon_parameters


def _config():
    """Return fixed Muon and AdamW settings shared by layout and reference tests."""
    return SimpleNamespace(
        optimizer_impl="torch.optim",
        override_optimizer_config=None,
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        muon_momentum=0.95,
        muon_epsilon=1e-8,
        muon_adjust_lr="spectral_norm",
        muon_nesterov=False,
        muon_use_polar_express=False,
    )


def _optimizer(model):
    """Build the real optional Dion-backed optimizer or skip when Dion is absent."""
    pytest.importorskip("dion")
    return build_fsdp2_muon(model, _config(), SimpleNamespace(strategy="fsdp2"))


def _algorithm_by_id(model):
    """Map parameter identities to their production Muon or AdamW classification."""
    groups = group_fsdp2_muon_parameters(model)
    return {id(p): group["algorithm"] for group in groups for p in group["params"]}


def test_shared_embedding_and_exotic_output_head_win_over_linear_aliases():
    """Keep special shared weights on AdamW while covering each trainable parameter once."""

    class Model(torch.nn.Module):
        """Expose embedding and output-head weights through ordinary Linear aliases."""

        def __init__(self):
            """Create shared special weights, an ordinary projection and frozen parameters."""
            super().__init__()
            self.a_projection = torch.nn.Linear(4, 4, bias=False)
            self.token_table = torch.nn.Embedding(4, 4)
            self.a_projection.weight = self.token_table.weight
            self.hidden = torch.nn.Linear(4, 4)
            self.z_vocab_projection = torch.nn.Linear(4, 4, bias=False)
            self.alias = self.z_vocab_projection
            self.frozen = torch.nn.Linear(4, 4)
            self.frozen.requires_grad_(False)

        def get_output_embeddings(self):
            """Identify the output head despite its nonstandard module name."""
            return self.z_vocab_projection

    model = Model()
    groups = group_fsdp2_muon_parameters(model)
    params = [p for group in groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    algorithms = _algorithm_by_id(model)
    assert algorithms[id(model.a_projection.weight)] == "adamw"
    assert algorithms[id(model.z_vocab_projection.weight)] == "adamw"
    assert algorithms[id(model.hidden.weight)] == "muon"
    assert algorithms[id(model.hidden.bias)] == "adamw"


def test_custom_input_embedding_identity_is_excluded_from_muon():
    """Honor input-embedding ownership even when the embedding is implemented as Linear."""

    class Model(torch.nn.Module):
        """Use a one-hot Linear projection as an input embedding."""

        def __init__(self):
            """Create distinct input-embedding and hidden-projection weights."""
            super().__init__()
            # A one-hot token projection is an embedding even with a Linear implementation.
            self.token_projection = torch.nn.Linear(8, 4, bias=False)
            self.hidden = torch.nn.Linear(4, 4, bias=False)

        def get_input_embeddings(self):
            """Expose the token projection as the model's input embedding."""
            return self.token_projection

    model = Model()
    algorithms = _algorithm_by_id(model)
    assert algorithms[id(model.token_projection.weight)] == "adamw"
    assert algorithms[id(model.hidden.weight)] == "muon"


@pytest.mark.parametrize("field", ["param_names", "algorithm", "verl_muon_layout", "verl_muon_definition"])
def test_checkpoint_metadata_rejected_before_state_or_lr_changes(field):
    """Reject incompatible checkpoint metadata without modifying momentum or cached LR."""
    model = torch.nn.Linear(4, 4)
    optimizer = _optimizer(model)
    saved = copy.deepcopy(optimizer.state_dict())
    saved["param_groups"][0][field] = "incompatible"
    saved["param_groups"][0]["lr"] = 0.75
    lr_tensor = optimizer.param_groups[0]["lr"]
    before = float(lr_tensor)
    with pytest.raises(ValueError, match=f"checkpoint mismatch: {field}"):
        optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]["lr"] is lr_tensor
    assert float(lr_tensor) == before
    for state in optimizer.state.values():
        assert torch.count_nonzero(state["momentum"]) == 0


def test_checkpoint_load_refreshes_existing_lr_cache():
    """Restore serialized learning rates into the existing device LR tensors."""
    model = torch.nn.Linear(4, 4)
    optimizer = _optimizer(model)
    saved = copy.deepcopy(optimizer.state_dict())
    cached_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in saved["param_groups"]:
        group["lr"] = 0.125
    optimizer.load_state_dict(saved)
    for group, cached in zip(optimizer.param_groups, cached_lrs, strict=True):
        assert group["lr"] is cached
        assert float(cached) == 0.125
    assert all(isinstance(group["lr"], float) for group in optimizer.state_dict()["param_groups"])


def _dtensor(local, mesh, placement=None, shape=(4, 4)):
    """Wrap a local tensor with explicit global matrix metadata and test placement."""
    if placement is None:
        placement = Shard(0)
    return DTensor.from_local(local, mesh, [placement], run_check=False, shape=shape, stride=(shape[1], 1))


def _distributed_layout_worker(rank, rendezvous):
    """Exercise CPU DTensor layout guards, collective errors and state round trips."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=12))
    try:
        parent = init_device_mesh("cpu", (1, 2), mesh_dim_names=("replicate", "fsdp"))
        mesh = parent["fsdp"]
        param = torch.nn.Parameter(_dtensor(torch.ones(2, 4), mesh))
        assert _parameter_mesh([{"params": [param]}]) == mesh
        for placement in (Shard(1), Replicate(), Partial()):
            local = torch.ones(4, 2) if placement == Shard(1) else torch.ones(4, 4)
            invalid = torch.nn.Parameter(_dtensor(local, mesh, placement))
            with pytest.raises(ValueError, match="single 1D FSDP Shard"):
                _parameter_mesh([{"params": [invalid]}])
        two_dim = DTensor.from_local(
            torch.ones(2, 4), parent, [Replicate(), Shard(0)], run_check=False, shape=(4, 4), stride=(4, 1)
        )
        with pytest.raises(ValueError, match="single 1D FSDP Shard"):
            _parameter_mesh([{"params": [torch.nn.Parameter(two_dim)]}])
        with pytest.raises(ValueError, match="mixed Tensor and DTensor"):
            _parameter_mesh([{"params": [param, torch.nn.Parameter(torch.ones(4))]}])
        with pytest.raises(ValueError, match="actual DTensor"):
            _parameter_mesh([{"params": [torch.nn.Parameter(torch.ones(4, 4))]}])

        model = torch.nn.Linear(4, 4)
        model.weight = param
        model.bias = torch.nn.Parameter(
            DTensor.from_local(torch.ones(2), mesh, [Shard(0)], run_check=False, shape=(4,), stride=(1,))
        )
        optimizer = _optimizer(model)
        before = param.to_local().detach().clone()
        param.grad = _dtensor(torch.ones(2, 4), mesh) if rank == 0 else None
        with pytest.raises(ValueError, match="grad=None mask differs"):
            optimizer.step()
        assert torch.equal(param.to_local(), before)
        assert optimizer.param_groups[0]["step"] == 0
        assert torch.count_nonzero(optimizer.state[param]["momentum"].to_local()) == 0

        # Exercise actual DTensor serialization with the same per-rank torch.save/load
        # mechanism as the engine, without claiming a fully_shard training test.
        for state in optimizer.state.values():
            state["momentum"].to_local().fill_(rank + 1)
            if "variance" in state:
                state["variance"].to_local().fill_(rank + 2)
                state["step_dev"].fill_(257)
        payload = io.BytesIO()
        torch.save(optimizer.state_dict(), payload)
        payload.seek(0)
        restored = _optimizer(model)
        restored.load_state_dict(torch.load(payload, weights_only=False))
        for state in restored.state.values():
            assert isinstance(state["momentum"], DTensor)
            assert state["momentum"].device_mesh == mesh
            assert state["momentum"].placements == (Shard(0),)
            assert state["momentum"].dtype == torch.float32
            assert torch.all(state["momentum"].to_local() == rank + 1)
            if "variance" in state:
                assert torch.all(state["variance"].to_local() == rank + 2)
                assert state["step_dev"].dtype == torch.float32
                assert state["step_dev"].item() == 257
        # Both ranks have gradients, but one rank has a replicated gradient layout.
        param.grad = _dtensor(torch.ones(2, 4), mesh) if rank == 0 else _dtensor(torch.ones(4, 4), mesh, Replicate())
        with pytest.raises(ValueError, match="dense gradients matching"):
            optimizer.step()
        assert torch.equal(param.to_local(), before)
        assert optimizer.param_groups[0]["step"] == 0
    finally:
        dist.destroy_process_group()


def test_true_dtensor_submesh_and_collective_guards_on_two_cpu_ranks(tmp_path):
    """Check submesh guards on two Gloo ranks and bound hangs during collective failures."""
    pytest.importorskip("dion")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU gloo backend unavailable")
    context = mp.spawn(_distributed_layout_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail("DTensor collective guard worker exceeded 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def _distributed_update_worker(rank, rendezvous):
    """Compare five DTensor updates with full-matrix Dion, including missing gradients."""
    from dion import Muon

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        mesh = init_device_mesh("cpu", (2,))
        torch.manual_seed(71)
        model = torch.nn.ModuleList([torch.nn.Linear(4, rows) for rows in (3, 3, 1, 4)])
        reference = copy.deepcopy(model)
        for module in model:
            for name, param in list(module.named_parameters()):
                setattr(module, name, torch.nn.Parameter(distribute_tensor(param.detach(), mesh, [Shard(0)])))
        optimizer = _optimizer(model)
        config = _config()
        ref_optimizer = Muon(
            [
                {"params": [m.weight for m in reference], "algorithm": "muon"},
                {"params": [m.bias for m in reference], "algorithm": "adamw"},
            ],
            lr=config.lr,
            mu=config.muon_momentum,
            betas=config.betas,
            weight_decay=config.weight_decay,
            epsilon=config.muon_epsilon,
            nesterov=config.muon_nesterov,
            adjust_lr=config.muon_adjust_lr,
            use_polar_express=False,
        )
        for step in range(5):
            generator = torch.Generator().manual_seed(100 + step)
            for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
                grad = torch.randn(ref_param.shape, generator=generator)
                ref_param.grad = None if step == 2 and ref_param.shape == (1, 4) else grad
                param.grad = distribute_tensor(grad, mesh, [Shard(0)]) if ref_param.grad is not None else None
            optimizer.step()
            ref_optimizer.step()
            for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
                # Require exact Muon matrices/momentum. CPU AdamW vector kernels
                # may round differently for a local slice; use the already
                # preregistered FP32 optimizer-state budget from GPU acceptance.
                atol, rtol = (0, 0) if param.ndim == 2 else (2e-7, 2e-6)
                torch.testing.assert_close(param.full_tensor(), ref_param, atol=atol, rtol=rtol)
                for key, value in optimizer.state[param].items():
                    actual = value.full_tensor() if isinstance(value, DTensor) else value
                    torch.testing.assert_close(actual, ref_optimizer.state[ref_param][key], atol=atol, rtol=rtol)
    finally:
        dist.destroy_process_group()


def test_gathered_dtensor_updates_match_full_matrices_with_uneven_and_empty_shards(tmp_path):
    """Match full-matrix updates with uneven or empty local shards on two CPU ranks."""
    pytest.importorskip("dion")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU gloo backend unavailable")
    context = mp.spawn(_distributed_update_worker, args=(f"file://{tmp_path / 'updates'}",), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail("DTensor update worker exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
