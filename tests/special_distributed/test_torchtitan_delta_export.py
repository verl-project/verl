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
"""Validate the torchtitan engine's sharded delta export against its full export.

The model is a real torchtitan Qwen3, parallelized by torchtitan's own
``parallelize_fn`` and re-keyed by its own state dict adapter, so the DTensor
placements and HF names under test are the ones a real run produces. The engine
methods under test are the real ones -- only the training loop around them is
stubbed out.

    torchrun --nproc_per_node=4 tests/special_distributed/test_torchtitan_delta_export.py
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models import qwen3 as tt_qwen3

from verl.checkpoint_engine.delta_sync.sparse_gather import gather_slot_entries_to_rank0
from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

FLAVOR = "debugmodel"


class _ExportOnlyEngine(TorchTitanEngine):
    """A TorchTitanEngine cut down to what the weight-export path reads.

    The export touches nothing but the model parts, the checkpointer's state dict
    adapter and the parallel dims, so bypassing ``__init__`` keeps the real methods
    under test without standing up a trainer, an optimizer or a dataloader.
    """

    def __init__(self, module, sd_adapter, parallel_dims):
        self.module = module
        self.checkpointer = type("_Ckpt", (), {"sd_adapter": sd_adapter})()
        self.parallel_dims = parallel_dims
        self._is_offload_param = False


def _build_sharded_model(parallel_dims):
    """Build a torchtitan Qwen3 and parallelize it the way torchtitan itself does."""
    spec = tt_qwen3.model_registry(FLAVOR)
    with torch.device("cuda"):
        model = spec.model.build()
    model.to(torch.float32)
    spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=parallel_dims.dp_shard,
            data_parallel_replicate_degree=parallel_dims.dp_replicate,
            tensor_parallel_degree=parallel_dims.tp,
            context_parallel_degree=parallel_dims.cp,
            pipeline_parallel_degree=parallel_dims.pp,
            expert_parallel_degree=parallel_dims.ep,
        ),
        compile_config=CompileConfig(enable=False),
        ac_config=None,
        dump_folder="/tmp/tt_delta_export_test",
    )
    adapter = spec.state_dict_adapter(spec.model, None) if spec.state_dict_adapter else None
    return [model], adapter


def _full_export(engine):
    """The full HF export as ``{name: bf16 flat tensor}`` -- the delta's reference."""
    gen, _ = engine.get_per_tensor_param()
    return {name: t.detach().to(torch.bfloat16).reshape(-1).clone() for name, t in gen}


def _perturb(model, shard_coord, step):
    """Nudge a sparse subset of every local shard, as an optimizer step would.

    Each shard touches different positions, so the gather has to stitch
    contributions from all of them: the exact thing that silently produced a
    partial result when a placement was misread. The seed is the *shard* coordinate
    rather than the global rank so that HSDP replicas of one shard stay identical,
    which is the invariant a real optimizer step preserves and the one the
    replica-skipping side of the export relies on.
    """
    g = torch.Generator(device="cuda").manual_seed(1234 + 97 * step + shard_coord)
    with torch.no_grad():
        for p in model.parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            flat = local.reshape(-1)
            if flat.numel() == 0:
                continue
            k = max(1, flat.numel() // 20)
            pos = torch.randint(0, flat.numel(), (k,), device=flat.device, generator=g)
            # big enough to survive the fp32 -> bf16 cast the export applies
            flat[pos] += 0.5 + torch.rand(k, device=flat.device, generator=g)


def _adamw_stepper(model):
    """A step function that takes a real AdamW step on the DTensor parameters.

    The sparse perturbation above is the easy regime for a delta: few positions, all
    inside one shard. A real step moves nearly every element, so this is the case
    where a wrong index lands inside the valid range instead of out of it, and where
    the fp32-master-to-bf16 cast decides what counts as changed at all. Gradients are
    seeded by shard coordinate for the same reason the perturbation is.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def _step(model, shard_coord, step):
        g = torch.Generator(device="cuda").manual_seed(4321 + 97 * step + shard_coord)
        with torch.no_grad():
            for p in model.parameters():
                local = p.to_local() if isinstance(p, DTensor) else p
                grad = torch.randn(local.shape, device=local.device, dtype=local.dtype, generator=g)
                if isinstance(p, DTensor):
                    grad = DTensor.from_local(grad, p.device_mesh, p.placements)
                p.grad = grad
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return _step


def _gather_delta_to_rank0(engine):
    """Consume the delta export the way the delta engine does: every rank walks the
    entries in lockstep and joins each gather; rank 0 keeps the assembled result."""
    gen, _ = engine.get_per_tensor_param_delta_shard()
    out = {}
    for slots, _dtype_str, counts, hf_idx, hf_val, pg in gen:
        gathered = gather_slot_entries_to_rank0(hf_idx, hf_val, counts.to(hf_idx.device), group=pg)
        if gathered is None:  # rank != 0
            continue
        for (name, _shape), (idx, val) in zip(slots, gathered, strict=True):
            out[name] = (idx, val)
    return out


def _compare(name, sharded, full_before, full_after):
    """A parameter passes when the gathered sparse delta is byte-identical to the
    diff of the full tensors before and after the step."""
    idx, val = sharded
    fb, fa = full_before[name], full_after[name]
    int_view = {2: torch.int16, 4: torch.int32}[fa.element_size()]
    bmask = fa.view(int_view) != fb.view(int_view)
    b_idx = bmask.nonzero(as_tuple=False).view(-1).to(torch.int64)
    b_val = fa[b_idx]

    so, bo = torch.argsort(idx.to(torch.int64)), torch.argsort(b_idx)
    idx_ok = idx.numel() == b_idx.numel() and torch.equal(idx.to(torch.int64)[so], b_idx[bo])
    val_ok = idx_ok and torch.equal(val[so].view(int_view), b_val[bo].view(int_view))
    return idx_ok and val_ok, int(b_idx.numel())


def _test_delta_matches_full(engine, model, shard_coord, rank, tag, step_fn, steps=2):
    """Seed, then take several steps, checking each step's delta against the full diff.

    Also keeps a rollout-side reconstruction: the seed's full tensors with every
    step's delta scattered into them, which is what the receiver ends up holding.
    Comparing that to the trainer's own weights closes the loop -- a delta can match
    a single step's diff and still drift if its coordinates are off relative to the
    HF tensor, or if the snapshot bookkeeping slips.

    More than one step on purpose: a delta export that forgets to refresh its
    snapshot still passes the first step and diverges from then on.
    """
    engine.prime_delta_snapshots()
    replica = _full_export(engine)  # what the seed sync hands the rollout side
    ok = True
    for step in range(steps):
        full_before = _full_export(engine)
        step_fn(model, shard_coord, step)
        full_after = _full_export(engine)
        gathered = _gather_delta_to_rank0(engine)
        if rank != 0:
            continue

        missing, extra = set(full_after) - set(gathered), set(gathered) - set(full_after)
        assert not missing and not extra, f"delta/full name mismatch: missing={sorted(missing)} extra={sorted(extra)}"

        failures, changed = [], 0
        for name in sorted(gathered):
            param_ok, nnz = _compare(name, gathered[name], full_before, full_after)
            changed += nnz
            if not param_ok:
                failures.append(name)
            idx, val = gathered[name]
            replica[name][idx.to(torch.int64)] = val

        drifted = [
            n for n in sorted(replica) if not torch.equal(replica[n].view(torch.int16), full_after[n].view(torch.int16))
        ]
        print(
            f"[delta vs full] {tag} step={step} params={len(gathered)} changed_elems={changed} "
            f"failures={len(failures)} replica_drift={len(drifted)}"
        )
        for name in (failures + drifted)[:5]:
            print("  FAIL", name)
        ok = ok and not failures and not drifted
    return ok


def _test_export_names_agree(engine, rank):
    """The full and the sharded export must enumerate the same HF tensors: the delta
    path pairs them by name, so a mismatch means a weight the rollout never updates."""
    full_names = [name for name, _ in engine.get_per_tensor_param()[0]]
    shard_names = [name for name, _, _ in engine.get_per_tensor_param_shard()[0]]
    ok = full_names == shard_names
    if rank == 0:
        print(f"[names] full={len(full_names)} sharded={len(shard_names)} identical_order={ok}")
        if not ok:
            print("  only in full   :", sorted(set(full_names) - set(shard_names))[:5])
            print("  only in sharded:", sorted(set(shard_names) - set(full_names))[:5])
        assert "lm_head.weight" in shard_names, "weight tying: lm_head must be re-added to the sharded export too"
    return ok


def _test_unsupported_layouts_rejected(world, rank):
    """TP / EP / PP must be named at the export boundary, not surface later as a
    placement error or, worse, a wrong translation."""
    cases = {}
    if world % 2 == 0:
        cases["tp"] = dict(dp_shard=world // 2, dp_replicate=1, cp=1, tp=2, pp=1, ep=1)
        cases["pp"] = dict(dp_shard=world // 2, dp_replicate=1, cp=1, tp=1, pp=2, ep=1)
        cases["ep"] = dict(dp_shard=world, dp_replicate=1, cp=1, tp=1, pp=1, ep=2)
    ok = True
    for tag, dims in cases.items():
        engine = _ExportOnlyEngine(None, None, ParallelDims(world_size=world, **dims))
        try:
            engine.get_per_tensor_param_shard()
            ok = False
            if rank == 0:
                print(f"[guard {tag}] FAIL: export accepted an unsupported layout")
        except NotImplementedError as e:
            if rank == 0:
                print(f"[guard {tag}] rejected: {str(e)[:80]}...")
    return ok


def _layouts(world):
    """The layouts the sharded export claims to support, as (tag, ParallelDims kwargs).

    HSDP earns its place: the replicate dim makes several ranks hold the same shard,
    and every one of them contributing would double-count. CP earns its place
    because the claim that it needs no special handling rests on torchtitan folding
    it into the ``fsdp`` mesh dim -- worth checking rather than asserting.
    """
    out = [("fsdp", dict(dp_shard=world, dp_replicate=1, cp=1))]
    if world % 2 == 0:
        out.append(("hsdp", dict(dp_shard=world // 2, dp_replicate=2, cp=1)))
        out.append(("fsdp+cp", dict(dp_shard=world // 2, dp_replicate=1, cp=2)))
    return out


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    all_ok = True

    for tag, dims in _layouts(world):
        parallel_dims = ParallelDims(tp=1, pp=1, ep=1, world_size=world, **dims)
        parallel_dims.build_mesh()
        module, adapter = _build_sharded_model(parallel_dims)
        engine = _ExportOnlyEngine(module, adapter, parallel_dims)
        # replicas of one shard must perturb identically, so seed by shard coordinate
        shard_coord = parallel_dims.get_mesh(["fsdp"]).get_local_rank()

        all_ok = _test_export_names_agree(engine, rank) and all_ok
        # sparse changes exercise the offset math, a real step the dense regime
        for label, step_fn in (("sparse", _perturb), ("adamw", _adamw_stepper(module[0]))):
            tagged = f"{tag}/{label}"
            all_ok = _test_delta_matches_full(engine, module[0], shard_coord, rank, tagged, step_fn) and all_ok
        del engine, module

    all_ok = _test_unsupported_layouts_rejected(world, rank) and all_ok

    if rank == 0:
        print("=" * 50)
        print(f"OVERALL: {'ALL PASS ✅' if all_ok else 'FAIL ❌'}")
        print("=" * 50)
    dist.barrier()
    dist.destroy_process_group()
    assert all_ok


if __name__ == "__main__":
    main()
