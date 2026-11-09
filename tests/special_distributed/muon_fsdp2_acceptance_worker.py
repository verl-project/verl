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
"""CUDA-only acceptance worker. Launch with run_muon_fsdp2_acceptance.sh.

The reference is unsharded Dion Muon, including its native AdamW groups. Both
paths store parameters and state in FP32; Dion's Newton-Schulz arithmetic casts
to BF16 even in the FP32 case. Fixed tolerances below allow BF16 batched GEMM
rounding; they are not fitted to measurements and must not be widened to get a
passing run. Restart comparisons are bitwise exact, with no numerical tolerance.
"""

import argparse
import copy
import datetime
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from dion import Muon
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import PretrainedConfig

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import MixedPrecisionPolicy, fsdp2_clip_grad_norm_, fully_shard
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

# Preregistered heuristic error budget, NOT an established numerical bound:
# BF16 round-to-nearest unit roundoff is u=2^-8 (~0.00391). Five Newton-Schulz
# iterations each evaluate three GEMMs plus polynomial additions/scalings. An
# uncorrelated-error estimate u*sqrt(15) is ~1.5%; twice that motivates the 3%
# synthetic-update relative L2 ceiling. Polynomial conditioning, BF16 casts,
# batched-GEMM choices and correlations invalidate treating this as a proof.
# The fixture uses lr=0.003; its per-coordinate matrix updates are typically at
# the 1e-4 to 1e-3 scale. A 3e-5 absolute weight ceiling covers percent-scale
# update rounding, supplemented by rtol=3e-4 on accumulated stored weights.
# Synthetic gradients are identical, so momentum/Adam states retain separate
# tight FP32 budgets rather than inheriting the BF16 update budget.
# The five-step training trajectory feeds rounded weights back into gradients.
# FP32 compute gets a 5% update ceiling; BF16 compute gets 10%, with larger
# absolute weight/state ceilings to cover that feedback and forward/backward
# rounding. Neither sqrt(steps) nor linear accumulation is a guaranteed bound.
# Every step (including 1 and 5) must pass both the absolute/relative tensor
# checks and the update-relative check. These thresholds were fixed WITHOUT
# GPU measurements: a failing run requires investigation, never widening them
# merely to pass. Checkpoint restart comparisons remain exact (atol=rtol=0).
TOLERANCES = {
    "optimizer": {"weights": (3e-5, 3e-4), "state": (2e-7, 2e-6), "update_relative_l2": 0.03},
    "fp32": {"weights": (2e-4, 2e-3), "state": (2e-5, 2e-2), "update_relative_l2": 0.05},
    "bf16": {"weights": (5e-4, 5e-3), "state": (2e-4, 5e-2), "update_relative_l2": 0.10},
}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.up = nn.Linear(32, 64)
        self.down = nn.Linear(64, 32)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(self.norm(x))))


class AcceptanceModel(nn.Module):
    """Repeated, uneven and empty shards plus embeddings, head, biases and norms."""

    def __init__(self, stochastic=False):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False)
        self.embed_tokens = nn.Embedding(64, 32)
        self.blocks = nn.ModuleList([Block(), Block()])
        # With two ranks these produce (2, 1) and (1, 0) row counts. Both
        # bottlenecks participate in the loss; they are not dead test parameters.
        self.uneven = nn.Linear(32, 3)
        self.uneven_out = nn.Linear(3, 32)
        self.single_row = nn.Linear(32, 1)
        self.single_row_out = nn.Linear(1, 32)
        self.dropout = nn.Dropout(0.15 if stochastic else 0.0)
        self.lm_head = nn.Linear(32, 64)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def can_generate(self):
        return False

    def forward(self, tokens):
        x = self.embed_tokens(tokens)
        for block in self.blocks:
            x = block(x)
        x = x + self.uneven_out(torch.tanh(self.uneven(x))) + self.single_row_out(torch.tanh(self.single_row(x)))
        return self.lm_head(self.dropout(x))


def full(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: full(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(full(item) for item in value)
    return copy.deepcopy(value)


def assert_tree_close(actual, expected, *, atol=0.0, rtol=0.0, path="root"):
    """Check every tensor, scalar, key, group field and sequence entry."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: dictionary keys differ"
        for key in expected:
            assert_tree_close(actual[key], expected[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
    elif isinstance(expected, tuple | list):
        assert type(actual) is type(expected) and len(actual) == len(expected), f"{path}: sequence differs"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_tree_close(left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def snapshot(model, optimizer, scheduler=None):
    names = {id(param): name for name, param in model.named_parameters()}
    # Preserve *all* state, not just a chosen list of expected momentum names.
    state = {names[id(param)]: full(value) for param, value in optimizer.state.items()}
    groups = []
    serialized = optimizer.state_dict()
    for runtime, saved in zip(optimizer.param_groups, serialized["param_groups"], strict=True):
        group = full({key: value for key, value in saved.items() if key != "params"})
        group["params"] = [names[id(param)] for param in runtime["params"]]
        groups.append(group)
    assert set(state) == set(names.values()), "Optimizer state is missing trainable parameters"
    weights = {name: full(param) for name, param in model.named_parameters()}
    for name, weight in weights.items():
        assert torch.isfinite(weight).all(), f"{name}: full weight contains nonfinite values"
    return {
        "weights": weights,
        "state": state,
        "groups": groups,
        "scheduler": full(scheduler.state_dict()) if scheduler is not None else None,
    }


def assert_storage(model, optimizer, step):
    assert isinstance(optimizer, Muon), "Acceptance must exercise Dion Muon kernels and state"
    assert {group["algorithm"] for group in optimizer.param_groups} == {"muon", "adamw"}
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name}: expected real FSDP2 DTensor"
        assert param.dtype == torch.float32, f"{name}: master weights must be FP32"
        assert param.to_local().device.type == "cuda", f"{name}: parameter is offloaded"
        for key, value in optimizer.state[param].items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == torch.float32, f"{name}.{key}: optimizer state must be FP32"
                local = value.to_local() if isinstance(value, DTensor) else value
                assert local.device.type == "cuda", f"{name}.{key}: state is offloaded"
                assert torch.isfinite(local).all(), f"{name}.{key}: state is nonfinite"
                if key != "step_dev":
                    assert isinstance(value, DTensor), f"{name}.{key}: state must retain DTensor metadata"
                    assert value.placements == param.placements
                    assert value.device_mesh == param.device_mesh
                else:
                    assert value.item() == step, f"{name}: Adam step counter differs"
    for group in optimizer.param_groups:
        assert group["step"] == step


def create_pair(precision, *, stochastic=False, engine_builder=True):
    torch.manual_seed(9123)
    reference = AcceptanceModel(stochastic=stochastic).cuda()
    model = copy.deepcopy(reference)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, cast_forward_inputs=True)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    if dist.get_world_size() == 2:
        rank = dist.get_rank()
        assert model.uneven.weight.to_local().shape == ((2, 32) if rank == 0 else (1, 32))
        assert model.single_row.weight.to_local().shape == ((1, 32) if rank == 0 else (0, 32))
    config = FSDPOptimizerConfig(
        optimizer="Muon",
        lr=3e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        clip_grad=0.25,
        muon_momentum=0.95,
        muon_nesterov=False,
        muon_adjust_lr="spectral_norm",
        muon_use_polar_express=False,
        muon_epsilon=1e-8,
    )
    engine_config = FSDPEngineConfig(strategy="fsdp2", model_dtype="fp32", use_torch_compile=False)
    if engine_builder:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

        context = SimpleNamespace(
            optimizer_config=config,
            engine_config=engine_config,
            model_config=SimpleNamespace(lora_rank=0, use_liger=False),
        )
        optimizer = FSDPEngine._build_optimizer(context, model)
    else:
        optimizer = build_optimizer(model.parameters(), config, model=model, engine_config=engine_config)
    # Build reference groups independently of production grouping logic, so an
    # incorrect embedding/head classification cannot make both sides agree.
    reference_groups = [
        {"params": [], "param_names": [], "algorithm": "muon"},
        {"params": [], "param_names": [], "algorithm": "adamw"},
    ]
    for name, param in sorted(reference.named_parameters()):
        matrix = (name.startswith("blocks.") and name.endswith(("up.weight", "down.weight"))) or name in {
            "uneven.weight",
            "uneven_out.weight",
            "single_row.weight",
            "single_row_out.weight",
        }
        group = reference_groups[0 if matrix else 1]
        group["params"].append(param)
        group["param_names"].append(name)
    # These fields are production checkpoint compatibility metadata, not Dion
    # algorithm inputs. Check their independently constructed expected values too.
    for group in reference_groups:
        group["verl_muon_layout"] = {
            "world_size": dist.get_world_size(),
            "sharded": True,
            "shapes": [tuple(param.shape) for param in group["params"]],
        }
        group["verl_muon_definition"] = {
            "dion_commit": "7692479288f928a3d61d1cc58137b4dbe4f71000",
            "state_dtype": "fp32",
            "polar_express": False,
            "matrix_evaluation": "unsharded_shape_batch_v1",
        }
    ref_optimizer = Muon(
        reference_groups,
        distributed_mesh=None,
        lr=config.lr,
        mu=config.muon_momentum,
        betas=config.betas,
        weight_decay=config.weight_decay,
        epsilon=config.muon_epsilon,
        nesterov=config.muon_nesterov,
        adjust_lr=config.muon_adjust_lr,
        use_polar_express=config.muon_use_polar_express,
    )
    expected = {name: group["algorithm"] for group in reference_groups for name in group["param_names"]}
    actual = {name: group["algorithm"] for group in optimizer.param_groups for name in group["param_names"]}
    assert actual == expected, "Production groups do not match independent module-role classification"
    assert_storage(model, optimizer, 0)
    return model, optimizer, reference, ref_optimizer, config


def max_abs_tensor_error(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (actual - expected).abs().max().item() if expected.numel() else 0.0
    if isinstance(expected, dict):
        return max((max_abs_tensor_error(actual[key], value) for key, value in expected.items()), default=0.0)
    if isinstance(expected, tuple | list):
        return max(
            (max_abs_tensor_error(left, right) for left, right in zip(actual, expected, strict=True)), default=0.0
        )
    return 0.0


def assert_nonzero_group_updates(optimizer, before, after):
    """A pair of no-op optimizers must not satisfy numerical alignment."""
    update_norms = {}
    for group in optimizer.param_groups:
        gradients = [full(param.grad) for param in group["params"] if param.grad is not None]
        active = any(torch.count_nonzero(grad).item() > 0 for grad in gradients)
        # The fixture intentionally exercises both Muon and AdamW on every step.
        assert active, f"{group['algorithm']}: fixture produced no effective gradient"
        squared_update = sum((after[name] - before[name]).double().square().sum() for name in group["param_names"])
        assert squared_update.item() > 0, f"{group['algorithm']}: nonzero gradients produced a zero group update"
        update_norms[group["algorithm"]] = squared_update.sqrt().item()
    return update_norms


def compare(model, optimizer, reference, ref_optimizer, before, ref_before, budget, step):
    actual, expected = snapshot(model, optimizer), snapshot(reference, ref_optimizer)
    # Group ordering is allowed to differ; group membership and every option are checked.
    actual["groups"].sort(key=lambda group: group["algorithm"])
    expected["groups"].sort(key=lambda group: group["algorithm"])
    assert_tree_close(actual["groups"], expected["groups"], path="optimizer.groups")
    actual_updates = assert_nonzero_group_updates(optimizer, before, actual["weights"])
    reference_updates = assert_nonzero_group_updates(ref_optimizer, ref_before, expected["weights"])
    metrics = {
        "budget": budget,
        "step": step,
        "max_abs_weight_error": max_abs_tensor_error(actual["weights"], expected["weights"]),
        "max_abs_state_error": max_abs_tensor_error(actual["state"], expected["state"]),
        "group_update_l2": actual_updates,
        "reference_group_update_l2": reference_updates,
    }
    # Emit diagnostics before assertions so failures retain every matrix's error,
    # including the uneven/empty shard fixtures. Acceptance budgets stay fixed.
    metrics["parameter_errors"] = {}
    for name, weight in actual["weights"].items():
        delta, ref_delta = weight - before[name], expected["weights"][name] - ref_before[name]
        metrics["parameter_errors"][name] = {
            "max_abs": max_abs_tensor_error(weight, expected["weights"][name]),
            "update_relative_l2": ((delta - ref_delta).norm() / ref_delta.norm().clamp_min(1e-12)).item(),
        }
    if dist.get_rank() == 0:
        print(json.dumps(metrics), flush=True)
    for part in ("weights", "state"):
        atol, rtol = TOLERANCES[budget][part]
        assert_tree_close(actual[part], expected[part], atol=atol, rtol=rtol, path=part)
    errors = []
    for name, weight in actual["weights"].items():
        delta, ref_delta = weight - before[name], expected["weights"][name] - ref_before[name]
        relative = (delta - ref_delta).norm() / ref_delta.norm().clamp_min(1e-12)
        assert relative <= TOLERANCES[budget]["update_relative_l2"], (
            f"{name}, step {step}: update relative L2 {relative.item():.6g} exceeds fixed {budget} budget"
        )
        errors.append(relative.item())
    assert_storage(model, optimizer, step)
    if dist.get_rank() == 0:
        print(json.dumps({"budget": budget, "step": step, "max_update_relative_l2": max(errors)}), flush=True)


def optimizer_alignment(precision, steps, *, engine_builder=True):
    model, optimizer, reference, ref_optimizer, _ = create_pair(precision, engine_builder=engine_builder)
    for step in range(1, steps + 1):
        before = {name: full(param) for name, param in model.named_parameters()}
        ref_before = {name: full(param) for name, param in reference.named_parameters()}
        generator = torch.Generator().manual_seed(1777 + step)
        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            gradient = torch.randn(param.shape, generator=generator, dtype=torch.float32).cuda()
            param.grad = distribute_tensor(gradient, device_mesh=param.device_mesh, placements=param.placements)
            ref_param.grad = gradient.clone()
        optimizer.step()
        ref_optimizer.step()
        compare(model, optimizer, reference, ref_optimizer, before, ref_before, "optimizer", step)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)


def batch(step, rank):
    generator = torch.Generator().manual_seed(501 + step * 37 + rank)
    return (
        torch.randint(64, (4, 8), generator=generator).cuda(),
        torch.randint(64, (4, 8), generator=generator).cuda(),
    )


def loss_for(model, tokens, labels, precision=None):
    if precision == "bf16":
        # FSDP2 casts every parameter for compute, including norms/embeddings.
        # Autocast alone would leave those FP32 and would be a different reference.
        params = {name: param.to(torch.bfloat16) for name, param in model.named_parameters()}
        logits = torch.func.functional_call(model, params, (tokens,))
    else:
        logits = model(tokens)
    return nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


def training_alignment(precision, steps):
    model, optimizer, reference, ref_optimizer, config = create_pair(precision)
    for step in range(1, steps + 1):
        before = {name: full(param) for name, param in model.named_parameters()}
        ref_before = {name: full(param) for name, param in reference.named_parameters()}
        tokens, labels = batch(step, dist.get_rank())
        loss = loss_for(model, tokens, labels)
        loss.backward()
        # Compute the exact global batch with the same microbatch GEMM shapes.
        # Accumulate *after* BF16 gradients cast back to FP32, as FSDP reduction does.
        reference_grads = [torch.zeros_like(param) for param in reference.parameters()]
        for rank in range(dist.get_world_size()):
            tokens, labels = batch(step, rank)
            ref_loss = loss_for(reference, tokens, labels, precision)
            grads = torch.autograd.grad(ref_loss, tuple(reference.parameters()))
            for total, grad in zip(reference_grads, grads, strict=True):
                total.add_(grad)
        for param, grad in zip(reference.parameters(), reference_grads, strict=True):
            param.grad = grad.div_(dist.get_world_size())
        norm = full(fsdp2_clip_grad_norm_(model.parameters(), max_norm=config.clip_grad))
        ref_norm = full(torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=config.clip_grad))
        assert torch.isfinite(norm) and torch.isfinite(ref_norm)
        assert norm.item() > config.clip_grad, "Fixture must exercise active gradient clipping"
        torch.testing.assert_close(norm, ref_norm, atol=2e-4, rtol=0.02 if precision == "bf16" else 0.002)
        optimizer.step()
        ref_optimizer.step()
        compare(model, optimizer, reference, ref_optimizer, before, ref_before, precision, step)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)


def save_snapshot(path, model, optimizer, scheduler):
    value = snapshot(model, optimizer, scheduler)
    torch.save(value, path / f"rank_{dist.get_rank()}.pt")


def checkpoint_phase(phase, precision, output):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    model, optimizer, _, _, config = create_pair(precision, stochastic=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    manager = FSDPCheckpointManager(model=model, optimizer=optimizer, lr_scheduler=scheduler)
    rank = dist.get_rank()
    random.seed(2718 + rank)
    np.random.seed(2718 + rank)
    torch.manual_seed(2718 + rank)
    checkpoint = output / "checkpoint_step_5"
    start, stop = (5, 10) if phase == "resume" else (0, 5 if phase == "save" else 10)
    if phase == "resume":
        manager.load_checkpoint(local_path=str(checkpoint))
        loaded = snapshot(model, optimizer, scheduler)
        saved = torch.load(output / "save" / f"rank_{rank}.pt", weights_only=False)
        assert_tree_close(loaded, saved, path="loaded_step_5")
        assert_storage(model, optimizer, 5)
    for step in range(start + 1, stop + 1):
        # Fresh RNG-driven batches plus dropout make RNG restoration observable.
        tokens = torch.randint(64, (4, 8), device="cuda")
        labels = torch.randint(64, (4, 8), device="cuda")
        loss_for(model, tokens, labels).backward()
        norm = full(fsdp2_clip_grad_norm_(model.parameters(), max_norm=config.clip_grad))
        assert torch.isfinite(norm)
        optimizer.step()
        reported_lr = FSDPEngine.lr_scheduler_step(SimpleNamespace(lr_scheduler=scheduler))
        assert type(reported_lr) is float and np.isfinite(reported_lr), "LR metrics must be host scalars"
        optimizer.zero_grad(set_to_none=True)
        assert_storage(model, optimizer, step)
    destination = output / phase
    destination.mkdir(parents=True, exist_ok=True)
    save_snapshot(destination, model, optimizer, scheduler)
    if phase == "save":
        manager.save_checkpoint(local_path=str(checkpoint), global_step=5)
        # The driver waits for this process to EXIT before launching resume.
        required = [
            f"{part}_world_size_{dist.get_world_size()}_rank_{rank}.pt" for part in ("model", "optim", "extra_state")
        ]
        assert all((checkpoint / name).is_file() for name in required)
    if phase == "resume":
        continuous = torch.load(output / "continuous" / f"rank_{rank}.pt", weights_only=False)
        resumed = snapshot(model, optimizer, scheduler)
        assert_tree_close(resumed, continuous, path="continuous10_vs_restart5_plus5")


def main():
    if not __debug__:
        raise RuntimeError("Acceptance must run with assertions enabled; do not use python -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["smoke", "alignment", "optimizer", "training", "continuous", "save", "resume"], required=True
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamo-recompile-limit", type=int)
    args = parser.parse_args()
    if args.dynamo_recompile_limit is not None:
        if args.dynamo_recompile_limit <= 0:
            parser.error("--dynamo-recompile-limit must be positive")
        # Dion compiles each matrix shape and layout. This fixture additionally
        # runs plain and FSDP parameters through the same compiled functions.
        # Keep this opt-in and visible; never disable compilation or change math.
        torch._dynamo.config.cache_size_limit = args.dynamo_recompile_limit
    print(f"Dynamo recompile limit: {torch._dynamo.config.cache_size_limit}", flush=True)
    required = 1 if args.case == "smoke" else 2
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if not torch.cuda.is_available() or torch.cuda.device_count() < required or world_size != required:
        raise RuntimeError(f"{args.case} requires exactly {required} torchrun ranks and at least {required} CUDA GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 acceptance requires CUDA hardware with BF16 support")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    try:
        if args.case == "smoke":
            optimizer_alignment(args.precision, 1, engine_builder=False)
            training_alignment(args.precision, 1)
        elif args.case == "alignment":
            optimizer_alignment(args.precision, 5)
            training_alignment(args.precision, 5)
        elif args.case == "optimizer":
            optimizer_alignment(args.precision, 5)
        elif args.case == "training":
            training_alignment(args.precision, 5)
        else:
            checkpoint_phase(args.case, args.precision, args.output)
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"MUON_FSDP2_WORKER_OK case={args.case} precision={args.precision}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
