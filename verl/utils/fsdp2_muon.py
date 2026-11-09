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
"""Optional native FSDP2 Muon configuration, parameter classification and guards."""

import importlib
import logging
import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DION_COMMIT = "7692479288f928a3d61d1cc58137b4dbe4f71000"


def validate_fsdp2_muon_config(config, engine_config, model_config=None):
    """Fail before model loading/sharding for unsupported training configurations."""
    if engine_config is None or engine_config.strategy != "fsdp2":
        raise ValueError("Muon requires the native fsdp2 strategy (FSDP1/fsdp_turbo are unsupported).")
    if config.optimizer_impl != "torch.optim" or config.override_optimizer_config:
        raise ValueError("Muon uses pinned Dion; optimizer_impl and override_optimizer_config must keep defaults.")
    for name in ("param_offload", "optimizer_offload", "offload_policy"):
        if getattr(engine_config, name, False):
            raise ValueError(f"FSDP2 Muon does not support CPU offload: {name}=True")
    if getattr(engine_config, "ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("FSDP2 Muon currently requires ulysses_sequence_parallel_size=1.")
    if getattr(engine_config, "model_dtype", None) not in (None, "fp32", "float32", torch.float32):
        raise ValueError("FSDP2 Muon requires FP32 stored parameters and optimizer states (model_dtype=fp32).")
    mp = getattr(engine_config, "mixed_precision", None) or {}
    for name in ("param_dtype", "reduce_dtype"):
        if mp.get(name, "fp32") not in ("fp32", "float32", "bf16", "bfloat16", torch.float32, torch.bfloat16):
            raise ValueError("FSDP2 Muon supports only FP32/BF16 mixed precision.")
    if model_config is not None:
        if getattr(model_config, "lora_rank", 0) > 0 or getattr(model_config, "lora_adapter_path", None):
            raise ValueError("FSDP2 Muon does not support LoRA.")
        _reject_moe(getattr(model_config, "hf_config", None))
    for name, value, valid in (
        ("muon_momentum", config.muon_momentum, 0 <= config.muon_momentum < 1),
        ("muon_epsilon", config.muon_epsilon, config.muon_epsilon > 0),
        ("lr", config.lr, config.lr >= 0),
        ("weight_decay", config.weight_decay, config.weight_decay >= 0),
    ):
        if not math.isfinite(value) or not valid:
            raise ValueError(f"Invalid FSDP2 Muon {name}: {value}")
    if len(config.betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in config.betas):
        raise ValueError("Muon AdamW betas must be finite and in [0, 1).")
    if config.muon_adjust_lr not in ("spectral_norm", "rms_norm", None):
        raise ValueError("muon_adjust_lr must be spectral_norm, rms_norm or null.")


def _reject_moe(config):
    if config is None:
        return
    values = config.to_dict() if hasattr(config, "to_dict") else vars(config)

    def visit(mapping):
        for key, value in mapping.items():
            if isinstance(value, dict):
                visit(value)
            elif key in ("num_experts", "num_local_experts", "n_routed_experts") and value and value > 1:
                raise ValueError("FSDP2 Muon currently supports dense models only, not MoE.")

    visit(values)


def group_fsdp2_muon_parameters(model):
    """Only ordinary Linear weights use Muon; any special alias wins over a Linear alias.

    Classification uses owning module types, HF output-embedding identity and conventional
    output-head names. All other parameters conservatively use AdamW, including unknown 2D
    tensors. Tied parameters are deduplicated by identity, sorted by their first alias.
    """
    _reject_moe(getattr(model, "config", None))
    special_ids = set()
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        get_module = getattr(model, accessor, None)
        special_module = get_module() if callable(get_module) else None
        if special_module is not None:
            special_ids.update(id(p) for p in special_module.parameters())
    # Hybrid/state-space attention has gates and other nonstandard projections.
    # Keep the entire special block on AdamW until its parameter semantics are audited.
    for name, module in model.named_modules(remove_duplicate=False):
        if "linear_attn" in name.split(".") or "gateddeltanet" in type(module).__name__.lower():
            special_ids.update(id(p) for p in module.parameters())
    records = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        if hasattr(module, "peft_config") or "lora" in type(module).__name__.lower():
            raise ValueError("FSDP2 Muon does not support LoRA/PEFT modules.")
        if "moe" in type(module).__name__.lower() or "experts" in module_name.split("."):
            raise ValueError("FSDP2 Muon currently supports dense models only, not MoE.")
        for name, param in module.named_parameters(recurse=False, remove_duplicate=False):
            if not param.requires_grad:
                continue
            fqn = f"{module_name}.{name}" if module_name else name
            is_head = module_name == "output" or any(
                part in {"lm_head", "head", "output_layer", "embed_out", "classifier", "score"}
                for part in module_name.split(".")
            )
            ordinary = (
                isinstance(module, torch.nn.Linear)
                and name == "weight"
                and param.ndim == 2
                and not is_head
                and id(param) not in special_ids
            )
            record = records.setdefault(id(param), {"param": param, "names": [], "ordinary": True})
            record["names"].append(fqn)
            record["ordinary"] &= ordinary
    groups = {algo: {"params": [], "param_names": [], "algorithm": algo} for algo in ("muon", "adamw")}
    for record in sorted(records.values(), key=lambda r: min(r["names"])):
        group = groups["muon" if record["ordinary"] else "adamw"]
        group["params"].append(record["param"])
        group["param_names"].append(min(record["names"]))
    for group in groups.values():
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "FSDP2 Muon grouping %s: %d tensors, %d global elements",
                group["algorithm"],
                len(group["params"]),
                sum(p.numel() for p in group["params"]),
            )
            logger.debug("FSDP2 Muon %s names: %s", group["algorithm"], group["param_names"])
    return [group for group in groups.values() if group["params"]]


def _parameter_mesh(groups):
    params = [p for group in groups for p in group["params"]]
    if not params:
        raise ValueError("Muon needs trainable parameters.")
    mesh = None
    sharded = isinstance(params[0], DTensor)
    for p in params:
        if p.dtype != torch.float32:
            raise ValueError("FSDP2 Muon requires FP32 stored parameters and states.")
        if not (p.is_cpu or p.is_cuda):
            raise ValueError("FSDP2 Muon supports CUDA training; NPU/MPS are unsupported.")
        if isinstance(p, DTensor) != sharded:
            raise ValueError("Muon does not support mixed Tensor and DTensor parameters.")
        if sharded:
            if p.device_mesh.ndim != 1 or p.placements != (Shard(0),):
                raise ValueError("Muon requires a single 1D FSDP Shard(0) mesh; HSDP/TP are unsupported.")
            if mesh is not None and p.device_mesh != mesh:
                raise ValueError("All Muon/AdamW parameters must use the same FSDP DeviceMesh.")
            mesh = p.device_mesh
    if not sharded and dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Distributed FSDP2 Muon requires actual DTensor parameters.")
    return mesh


def _check_gradients(optimizer, args, kwargs):
    """Collectively reject asymmetric None masks before matrix communication."""
    # Dion evaluates a closure inside step, after pre-hooks; allowing it would bypass this check.
    if (len(args) > 1 and args[1] is not None) or kwargs.get("closure") is not None:
        raise ValueError("FSDP2 Muon requires backward before step; closures are unsupported.")
    params = [p for g in optimizer.param_groups for p in g["params"]]
    mesh = _parameter_mesh(optimizer.param_groups)
    valid = []
    for p in params:
        grad = p.grad
        ok = grad is None or (not grad.is_sparse and grad.dtype == p.dtype and grad.shape == p.shape)
        if grad is not None and isinstance(p, DTensor):
            ok = ok and isinstance(grad, DTensor) and grad.device_mesh == mesh and grad.placements == p.placements
        valid.append(ok)
    if mesh is not None and mesh.size() > 1:
        flags = torch.tensor(
            [int(p.grad is not None) for p in params] + [int(all(valid))], dtype=torch.int32, device=params[0].device
        )
        low, high = flags.clone(), flags.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=mesh.get_group())
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=mesh.get_group())
        if not torch.equal(low[:-1], high[:-1]):
            raise ValueError("Muon grad=None mask differs across FSDP ranks; no parameters were updated.")
        if not low[-1].item():
            raise ValueError("Muon requires dense gradients matching parameter dtype, mesh and placements.")
    elif not all(valid):
        raise ValueError("Muon requires dense gradients matching parameter dtype, mesh and placements.")


def _validate_checkpoint(optimizer, state_dict):
    current = optimizer.param_groups
    saved = state_dict["param_groups"]
    if len(saved) != len(current):
        raise ValueError("Muon checkpoint parameter groups changed.")
    for old, new in zip(saved, current, strict=True):
        for key in ("param_names", "algorithm", "verl_muon_layout", "verl_muon_definition"):
            if old.get(key) != new.get(key):
                raise ValueError(f"Muon checkpoint mismatch: {key}; same layout/world size/definition required.")


def build_fsdp2_muon(model, config, engine_config, model_config=None):
    validate_fsdp2_muon_config(config, engine_config, model_config)
    groups = group_fsdp2_muon_parameters(model)
    if not any(g["algorithm"] == "muon" for g in groups):
        raise ValueError("No ordinary Linear weights eligible for Muon; refusing an AdamW-only fallback.")
    mesh = _parameter_mesh(groups)
    try:
        muon = importlib.import_module("dion").Muon
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "FSDP2 Muon requires the optional pinned dependency: uv pip install -r requirements-fsdp2-muon.txt"
        ) from exc
    if mesh is not None:
        from verl.utils.fsdp2_muon_optimizer import FSDP2Muon

        muon = FSDP2Muon
    for group in groups:
        group["verl_muon_layout"] = {
            "world_size": mesh.size() if mesh is not None else 1,
            "sharded": mesh is not None,
            "shapes": [tuple(p.shape) for p in group["params"]],
        }
        group["verl_muon_definition"] = {
            "dion_commit": DION_COMMIT,
            "state_dtype": "fp32",
            "polar_express": config.muon_use_polar_express,
            "matrix_evaluation": "unsharded_shape_batch_v1",
        }
    optimizer = muon(
        groups,
        distributed_mesh=mesh,
        lr=config.lr,
        mu=config.muon_momentum,
        betas=tuple(config.betas),
        weight_decay=config.weight_decay,
        epsilon=config.muon_epsilon,
        nesterov=config.muon_nesterov,
        adjust_lr=config.muon_adjust_lr,
        flatten=False,
        cautious_wd=False,
        use_triton=False,
        use_gram_newton_schulz=False,
        use_polar_express=config.muon_use_polar_express,
    )
    optimizer.register_step_pre_hook(_check_gradients)
    optimizer.register_load_state_dict_pre_hook(_validate_checkpoint)
    return optimizer
