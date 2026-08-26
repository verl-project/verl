# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Utility functions for the Automodel engine integration."""

from dataclasses import dataclass

import torch
import torch.distributed
from torch.distributed.tensor import DTensor

from verl.utils.device import get_device_id, get_torch_device


def get_dp_rank(device_mesh, include_cp=False):
    """Get data-parallel rank from device mesh."""
    if device_mesh is None:
        return 0
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh.get_local_rank("dp_cp")
    return device_mesh.get_local_rank("dp")


def get_tp_rank(device_mesh):
    """Get tensor-parallel rank from device mesh."""
    if device_mesh is None or "tp" not in device_mesh.mesh_dim_names or device_mesh["tp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("tp")


def get_pp_rank(device_mesh):
    """Get pipeline-parallel rank from device mesh."""
    if device_mesh is None or "pp" not in device_mesh.mesh_dim_names or device_mesh["pp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("pp")


def get_dp_group_size(device_mesh, include_cp=False):
    """Get data-parallel group size from device mesh."""
    if device_mesh is None:
        return torch.distributed.get_world_size()
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh["dp_cp"].size()
    if "dp" in device_mesh.mesh_dim_names:
        return device_mesh["dp"].size()
    return torch.distributed.get_world_size()


def maybe_fully_shard_optimizer(model, optimizer, distributed_config):
    """Call fully_shard_optimizer for MegatronFSDP strategy."""
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
        from megatron_fsdp.fully_shard import fully_shard_optimizer

        fully_shard_optimizer(model, optimizer)


def build_distributed_config_from_engine_config(engine_config, world_size):
    """Build a v5 ``DistributedSetup`` (with device_mesh / moe_mesh) from engine config.

    Automodel 0.5.0 collapsed the previously-separate distributed kwargs
    (``distributed_config``, ``moe_mesh``, ``moe_config``, ``activation_checkpointing``)
    into a single :class:`DistributedSetup` object accepted by
    ``NeMoAutoModelForCausalLM.from_pretrained(distributed_setup=...)``. This helper
    builds that object from the verl ``AutomodelEngineConfig`` and also exposes the
    underlying ``strategy_config`` / meshes for the rest of the engine, which still
    reads them through the pre-0.5.0 field names.

    Args:
        engine_config: AutomodelEngineConfig instance.
        world_size: Total number of processes in the job.

    Returns:
        Tuple of (distributed_setup, strategy_config, device_mesh, moe_mesh), where
        ``distributed_setup`` is the :class:`DistributedSetup` to pass to
        ``from_pretrained``, ``strategy_config`` is the FSDP2Config / MegatronFSDPConfig
        / DDPConfig (used by ``maybe_fully_shard_optimizer`` and the optimizer builder),
        and ``device_mesh`` / ``moe_mesh`` are pulled from ``distributed_setup.mesh_context``.
    """
    from nemo_automodel.components.distributed.config import (
        DDPConfig,
        DistributedSetup,
        FSDP2Config,
        MegatronFSDPConfig,
        MoEParallelizerConfig,
        _resolve_strategy_config,
    )
    from nemo_automodel.components.distributed.mesh import ParallelismSizes
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

    from verl.utils.torch_dtypes import PrecisionType

    strategy = engine_config.distributed_strategy

    mp_policy = MixedPrecisionPolicy(
        param_dtype=PrecisionType.to_dtype(engine_config.mp_param_dtype),
        reduce_dtype=PrecisionType.to_dtype(engine_config.mp_reduce_dtype),
        output_dtype=PrecisionType.to_dtype(engine_config.mp_output_dtype),
        cast_forward_inputs=True,
    )

    offload_policy = CPUOffloadPolicy() if engine_config.param_offload else None

    # Build the typed strategy config; verl only supports the subset of strategies
    # whose constructor accepts the fields below.
    if strategy == "fsdp2":
        strategy_config = FSDP2Config(
            sequence_parallel=engine_config.sequence_parallel,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=engine_config.defer_fsdp_grad_sync,
        )
    elif strategy == "megatron_fsdp":
        strategy_config = MegatronFSDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )
    elif strategy == "ddp":
        strategy_config = DDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )
    else:
        # ``_resolve_strategy_config`` gives a helpful, version-accurate error.
        strategy_config = _resolve_strategy_config(strategy)

    parallelism = ParallelismSizes(
        tp_size=engine_config.tp_size,
        pp_size=engine_config.pp_size,
        cp_size=engine_config.cp_size,
        ep_size=engine_config.ep_size,
        dp_replicate_size=engine_config.dp_replicate_size,
    )

    # MoE parallelizer config: fold in the FSDP mp_policy so EP experts share the
    # same mixed-precision policy (mirrors the pre-0.5.0 ``moe_kwargs`` logic).
    moe_parallel_config = None
    if engine_config.ep_size > 1:
        moe_kwargs = dict(engine_config.moe_config) if engine_config.moe_config else {}
        moe_kwargs.setdefault("mp_policy", mp_policy)
        moe_parallel_config = MoEParallelizerConfig(**moe_kwargs)

    distributed_setup = DistributedSetup.build(
        strategy=strategy_config,
        parallelism_sizes=parallelism,
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=engine_config.activation_checkpointing,
        world_size=world_size,
    )

    mesh_context = distributed_setup.mesh_context
    return distributed_setup, strategy_config, mesh_context.device_mesh, mesh_context.moe_mesh


def build_peft_config(model_config):
    """Build an Automodel ``PeftConfig`` from verl's flat LoRA fields, or ``None`` if disabled.

    verl gates LoRA via ``model_config.lora_rank > 0`` (FSDP-style flat fields). This maps
    those fields to Automodel 0.5.0's :class:`PeftConfig` so that
    ``NeMoAutoModelForCausalLM.from_pretrained(peft_config=...)`` injects adapters,
    freezes base weights, and flags the internal checkpointer as PEFT — all automatically.

    A ``PeftConfig`` *instance* (not a dict) is required because Automodel's
    infrastructure code mutates ``peft_config.use_triton`` and reads ``.dim``/``.alpha``
    as attributes.
    """
    if model_config.lora_rank <= 0:
        return None
    from nemo_automodel.components._peft.lora import PeftConfig

    target_modules = model_config.target_modules
    match_all_linear = False
    # verl uses the "all-linear" string sentinel; Automodel uses match_all_linear=True
    # with an empty target_modules list.
    if isinstance(target_modules, str) and target_modules == "all-linear":
        target_modules = []
        match_all_linear = True
    elif target_modules is not None:
        target_modules = list(target_modules)
    else:
        target_modules = []

    # verl stores exclude_modules as Optional[str]; PeftConfig expects a list.
    exclude_modules = []
    if model_config.exclude_modules:
        exclude_modules = [model_config.exclude_modules]

    return PeftConfig(
        dim=model_config.lora_rank,
        alpha=model_config.lora_alpha,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        match_all_linear=match_all_linear,
        # use Automodel defaults for: use_dora, dropout, dropout_position,
        # lora_A_init, lora_dtype, use_memory_efficient_lora, use_triton,
        # moe_rank_scaling
    )


def build_automodel_model(model_config, engine_config, distributed_setup, strategy_config, device_mesh, moe_mesh):
    """Build a model using NeMoAutoModelForCausalLM.from_pretrained().

    Args:
        model_config: HFModelConfig with model path and settings.
        engine_config: AutomodelEngineConfig with distributed settings.
        distributed_setup: Resolved Automodel 0.5.0 ``DistributedSetup`` (topology +
            policy) passed to ``from_pretrained``. ``strategy_config``, ``device_mesh``
            and ``moe_mesh`` are the pre-0.5.0 decomposed views kept for callers that
            still read them directly (grad clipping, checkpointer, mesh helpers).
        strategy_config: FSDP2Config / MegatronFSDPConfig / DDPConfig.
        device_mesh: Pre-created device mesh (or None for DDP).
        moe_mesh: Pre-created MoE mesh (or None).

    Returns:
        Tuple of (model, peft_config) where peft_config is an Automodel ``PeftConfig``
        instance when LoRA is enabled (``model_config.lora_rank > 0``) or ``None``.
        The caller reuses peft_config for checkpointing and rollout weight sync.
    """
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM

    kwargs = {}

    if engine_config.enable_fp8:
        from nemo_automodel.components.quantization.fp8 import FP8Config

        kwargs["fp8_config"] = FP8Config()

    if engine_config.enable_compile:
        from nemo_automodel.components.utils.compile_utils import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    # Qwen/Llama with ep_size<=1: use HF implementation.
    from transformers import AutoConfig

    _cfg = AutoConfig.from_pretrained(model_config.path, trust_remote_code=model_config.trust_remote_code)
    _arch = (getattr(_cfg, "architectures", None) or [""])[0].lower()
    if engine_config.ep_size <= 1 and ("qwen" in _arch or "llama" in _arch):
        kwargs["force_hf"] = True

    if engine_config.backend_config and not kwargs.get("force_hf", False):
        from nemo_automodel.components.models.common.utils import BackendConfig

        backend_kwargs = dict(engine_config.backend_config)
        kwargs["backend"] = BackendConfig(**backend_kwargs)

    kwargs["attn_implementation"] = engine_config.attn_implementation

    from verl.utils.torch_dtypes import PrecisionType

    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    # LoRA: pass an Automodel PeftConfig instance so from_pretrained injects adapters,
    # freezes base params, and flags the internal base-weight checkpointer as PEFT.
    peft_config = build_peft_config(model_config)
    if peft_config is not None:
        kwargs["peft_config"] = peft_config

    if getattr(model_config, "override_config", None):
        kwargs["config"] = dict(model_config.override_config)

    model = NeMoAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.path,
        distributed_setup=distributed_setup,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )

    return model, peft_config


@torch.no_grad()
def offload_automodel_model_to_cpu(model, empty_cache=True):
    """Offload an FSDP2-wrapped model to CPU (reshard, move to CPU, optional cache clear)."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_automodel_model_to_gpu(model):
    """Load model back to GPU."""
    device = get_device_id()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_automodel_optimizer(optimizer):
    """Offload optimizer state to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_automodel_optimizer(optimizer, device_id):
    """Load optimizer state back to GPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


# Automodel <-> vLLM weight sync: split packed 3D MoE params into the per-expert
# 2D keys vLLM expects. See transformer_impl.get_per_tensor_param for dispatch.


def _norm_ac_ckpt(name):
    """Strip ``._checkpoint_wrapped_module.`` so module names match state_dict keys."""
    return name.replace("._checkpoint_wrapped_module.", ".")


def collect_automodel_lora_param_maps(model):
    """One module-tree walk -> (packed_expert_prefixes, moe_lora_prefixes,
    lora_linear_prefixes, moe_lora_modules, dense_lora_modules) for the
    weight-sync generator. The first three (specs / prefix sets) drive the
    non-merge two-phase split; the last two (prefix -> module) drive the
    merge path. Each is empty when the relevant types are absent or
    nemo_automodel is missing.
    """
    packed_expert_prefixes = {}
    moe_lora_prefixes = {}
    lora_linear_prefixes = set()
    moe_lora_modules = {}
    dense_lora_modules = {}

    try:
        from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE

        expert_types = (GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE)
    except ImportError:
        expert_types = ()
    try:
        from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA

        lora_expert_types = (GroupedExpertsLoRA,)
    except ImportError:
        lora_expert_types = ()
    try:
        from nemo_automodel.components._peft.lora import LinearLoRA

        lora_linear_types = (LinearLoRA,)
    except ImportError:
        lora_linear_types = ()

    # Order matters: GroupedExpertsLoRA subclasses GroupedExperts, so check the
    # LoRA variant first; otherwise it'd be caught by the base-expert branch.
    for name, module in model.named_modules():
        if lora_expert_types and isinstance(module, lora_expert_types):
            name = _norm_ac_ckpt(name)
            cfg = module.config
            spec = _MoELoRASpec(
                prefix=name,
                moe_inter_dim=cfg.moe_inter_dim,
                is_gated=bool(getattr(module, "is_gated", True)),
            )
            for attr in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
                moe_lora_prefixes[f"{name}.{attr}"] = spec
            moe_lora_modules[name] = module
            continue
        if expert_types and isinstance(module, expert_types):
            name = _norm_ac_ckpt(name)
            if hasattr(module, "gate_and_up_projs"):
                packed_expert_prefixes[f"{name}.gate_and_up_projs"] = _PackedExpertSpec(
                    prefix=name,
                    packed_attr="gate_and_up_projs",
                    splits=(("gate_proj", "up_proj"),),
                )
            if hasattr(module, "down_projs"):
                packed_expert_prefixes[f"{name}.down_projs"] = _PackedExpertSpec(
                    prefix=name,
                    packed_attr="down_projs",
                    splits=(("down_proj",),),
                )
            continue
        if lora_linear_types and isinstance(module, lora_linear_types):
            n = _norm_ac_ckpt(name)
            lora_linear_prefixes.add(n)
            dense_lora_modules[n] = module

    return packed_expert_prefixes, moe_lora_prefixes, lora_linear_prefixes, moe_lora_modules, dense_lora_modules


@dataclass
class _PackedExpertSpec:
    """One packed MoE base param and its split rule."""

    prefix: str
    packed_attr: str
    splits: tuple


def split_packed_expert(spec, packed_tensor, expert_id):
    """Yield ``(sub_name, 2D_tensor)`` for one expert from a packed 3D base param
    (``[n_experts, in, out]``), transposed to ``[out, in]``. Fused gate+up is
    chunked gate-then-up along the last dim.
    """
    per_expert = packed_tensor[expert_id]  # [in, out]
    for sub_names in spec.splits:
        if len(sub_names) == 1:
            yield sub_names[0] + ".weight", per_expert.t().contiguous()
        else:
            for sub_name, chunk in zip(sub_names, per_expert.chunk(len(sub_names), dim=-1), strict=False):
                yield sub_name + ".weight", chunk.t().contiguous()


@dataclass
class _MoELoRASpec:
    """Per-expert lora_A/lora_B split params for one GroupedExpertsLoRA module."""

    prefix: str
    moe_inter_dim: int
    is_gated: bool


def split_moe_lora_adapter(spec, attr, packed_tensor):
    """Yield ``(vllm_key, 2D_tensor)`` per expert from one fused 3D adapter param.
    ``attr`` is ``lora_gate_and_up_A/B`` (w1+w3) or ``lora_down_A/B`` (w2); A is
    ``[in, rank]``, B is ``[rank, out]``. Non-gated MoE skips up_proj (w3).
    """
    n_experts = packed_tensor.size(0)
    moe_inter_dim = spec.moe_inter_dim
    is_gated = spec.is_gated

    if attr == "lora_gate_and_up_A":
        # [n, expert_dim, lora_dim] -> w1 A and w3 A (shared input).
        for i in range(n_experts):
            a = packed_tensor[i].contiguous()
            yield f"{spec.prefix}.{i}.gate_proj.lora_A.weight", a
            if is_gated:
                yield f"{spec.prefix}.{i}.up_proj.lora_A.weight", a.clone()
    elif attr == "lora_gate_and_up_B":
        # [n, lora_dim, 2*moe_inter_dim] -> split last dim: w1 B | w3 B.
        for i in range(n_experts):
            b = packed_tensor[i]
            yield f"{spec.prefix}.{i}.gate_proj.lora_B.weight", b[..., :moe_inter_dim].contiguous()
            if is_gated:
                yield f"{spec.prefix}.{i}.up_proj.lora_B.weight", b[..., moe_inter_dim:].contiguous()
    elif attr == "lora_down_A":
        for i in range(n_experts):
            yield f"{spec.prefix}.{i}.down_proj.lora_A.weight", packed_tensor[i].contiguous()
    elif attr == "lora_down_B":
        for i in range(n_experts):
            yield f"{spec.prefix}.{i}.down_proj.lora_B.weight", packed_tensor[i].contiguous()
    else:
        raise ValueError(f"Unknown MoE LoRA adapter attr: {attr!r}")


def to_vllm_peft_dict(peft_config):
    """Translate an Automodel ``PeftConfig`` to the HF/peft dict vLLM's
    ``PEFTHelper.from_dict`` expects (``r``/``lora_alpha``/``target_modules``).
    ``match_all_linear`` -> ``target_modules=None`` (vLLM treats ``None`` as
    "all"; ``[]`` means "nothing").
    """
    d = peft_config.to_dict()
    target_modules = d.get("target_modules") or []
    target_modules = None if d.get("match_all_linear") else list(target_modules)
    return {
        "r": d["dim"],
        "lora_alpha": d["alpha"],
        "target_modules": target_modules,
        "use_rslora": False,
        "use_dora": bool(d.get("use_dora", False)),
        "bias": "none",
    }


# --- LoRA merge (model.lora.merge=true): fold adapters into base weights -----
# On-the-fly (no in-place mutation / backup-restore): the weight-sync generator
# calls these at yield time and ``.full_tensor()`` already copies, so yielded
# tensors never alias module storage.


def _full(t):
    """Unshard a DTensor to a plain tensor; pass through plain tensors."""
    return t.full_tensor() if isinstance(t, DTensor) else t


def merged_dense_lora_weight(module):
    """Merged weight ``base + scale*(lora_B @ lora_A)`` for a ``LinearLoRA``.

    Prefers the layer's ``materialize_effective_weight``; falls back to manual
    math when it raises (DoRA / quantized / delegated — not hit on TE backends).
    """
    try:
        return module.materialize_effective_weight().detach()
    except NotImplementedError:
        w = _full(module.weight)
        a = _full(module.lora_A.weight)
        b = _full(module.lora_B.weight)
        return (w + module.scale * (b @ a)).detach()


def merged_packed_expert_base(module, attr):
    """Merged 3D base for a ``GroupedExpertsLoRA`` module.

    ``attr`` is ``"gate_and_up_projs"`` or ``"down_projs"``. Returns the full
    ``[n_experts, ...]`` base tensor with the scaled ``(A @ B)`` delta folded
    per expert (no in-place mutation). MoE forward is ``x @ A[i] @ B[i]`` =
    ``x @ (A@B)``; base is stored untransposed ``[in, out]``, so the delta is
    ``A @ B`` (``[n,in,rank] @ [n,rank,out] -> [n,in,out]``) — NOT ``B @ A``.
    """
    if attr == "gate_and_up_projs":
        a = _full(module.lora_gate_and_up_A)
        b = _full(module.lora_gate_and_up_B)
    elif attr == "down_projs":
        a = _full(module.lora_down_A)
        b = _full(module.lora_down_B)
    else:
        raise ValueError(f"Unknown packed expert base attr: {attr!r}")
    base = _full(getattr(module, attr))
    scale = module.scale
    return (base + scale * torch.bmm(a, b)).detach()
