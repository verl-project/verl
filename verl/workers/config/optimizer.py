# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = [
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "WEIGHT_DECAY_SCALE_KEY",
    "build_optimizer",
    "VeOmniOptimizerConfig",
    "TorchtitanOptimizerConfig",
    "AutomodelOptimizerConfig",
]

WEIGHT_DECAY_SCALE_KEY = "_verl_weight_decay_scale"


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
        zero_indexed_step (bool): Whether the LR schedule uses 0-indexed steps. If True (default),
            step counting starts at 0. If False, step counting starts at 1.
        weight_decay_policy (str): Parameter selection policy for weight decay. ``"standard"`` excludes
            biases and normalization parameters, while ``"all"`` preserves the legacy behavior of applying
            weight decay to every parameter.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    zero_indexed_step: bool = True
    weight_decay_policy: Literal["standard", "all"] = "standard"

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        assert self.weight_decay_policy in ["standard", "all"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
        use_precision_aware_optimizer (bool): Enable Megatron's precision-aware optimizer so the
            grad-accumulation buffer and Adam moments can be stored below fp32 (bf16 training only).
            Opt-in; default False keeps the fp32 optimizer state and prior numerics. Requires
            TransformerEngine's FusedAdam. Mirrors Megatron's ``--use-precision-aware-optimizer``.
        main_grads_dtype (str): dtype of the main-grad / grad-accumulation buffer when the
            precision-aware optimizer is enabled ("fp32" or "bf16"). Also drives the DDP grad-bucket
            dtype so the two stay consistent. Mirrors Megatron's ``--main-grads-dtype``.
        exp_avg_dtype (str): dtype of the Adam first moment (m) when the precision-aware optimizer is
            enabled ("fp32" or "bf16"). Mirrors Megatron's ``--exp-avg-dtype``.
        exp_avg_sq_dtype (str): dtype of the Adam second moment (v) when the precision-aware optimizer
            is enabled ("fp32" or "bf16"). Mirrors Megatron's ``--exp-avg-sq-dtype``.
        optimizer (str): Optimizer algorithm; "adam"/"sgd" use Megatron's classic optimizers, while
            "muon" route through Megatron-Core's emerging_optimizers path (which builds
            the tensor-parallel-aware Muon optimizer). The ``muon_*`` fields below only take effect when
            a Muon algorithm is selected and are ignored otherwise.
        use_layer_wise_distributed_optimizer (bool): Wrap the emerging (Muon) optimizer with Megatron's
            LayerWiseDistributedOptimizer. Only relevant for Muon; mirrors Megatron's
            ``--use-layer-wise-distributed-optimizer``.
        use_layer_wise_param_layout (bool): Use Megatron's padded shard-aligned DDP layout for LayerWise
            buffers (master weights in param buffer). Default None → auto True when Muon+LayerWise.
        muon_momentum (float): Momentum of the internal SGD in Muon. Mirrors Megatron's ``--muon-momentum``.
        muon_nesterov (bool): Use Nesterov-style momentum in Muon's internal SGD.
        muon_split_qkv (bool): Split fused QKV parameters before the Muon update.
        muon_scale_mode (str): Scale-factor mode for the Muon update (e.g. "spectral"/"unit_rms_norm").
        muon_coefficient_type (str): Newton-Schulz coefficient type (e.g. "quintic"); valid values are
            discovered from the installed ``emerging_optimizers`` package.
        muon_num_ns_steps (int): Number of Newton-Schulz iteration steps.
        muon_tp_mode (str): How the Newton-Schulz calculation is performed for tensor-parallel weights
            (e.g. "blockwise").
        muon_fp32_matmul_prec (str): Precision for Muon's fp32 matmul (e.g. "medium").
        muon_extra_scale_factor (float): Additional scale factor applied to the Muon update. Muon's
            effective step size is ``lr * muon_extra_scale_factor``; the Megatron-Core default of
            ``1.0`` is *not* AdamW-comparable, so reusing an AdamW learning rate unchanged gives a
            much larger effective step. Prefer ``muon_match_adamw_update_rms`` over hard-coding a
            constant here.
        muon_match_adamw_update_rms (bool): Derive ``muon_extra_scale_factor`` from
            ``sqrt((1 - betas[0]) / (1 + betas[0]))``, the closed form that analytically matches
            AdamW's update RMS norm (emerging_optimizers 0.3.0 ``get_muon_scale_factor`` docstring;
            https://kexue.fm/archives/11267; https://arxiv.org/abs/2502.16982). At the default
            ``betas[0] = 0.9`` this resolves to ~0.2294. The resolved value is logged on rank 0.
            verl-side only -- it is not forwarded to Megatron-Core. Conflicts with an explicitly set
            ``muon_extra_scale_factor`` and raises in that case.
        muon_scalar_optimizer (str): Optimizer intended for the non-matrix ("scalar") parameters
            (embeddings, biases, norms) when Muon is selected. Megatron-Core declares this field
            but no Megatron-Core code path currently reads it, so the effective (and only
            recommended) behaviour is the default, "adam"; setting any other value is a silent
            no-op today. Forwarded as-is for forward-compatibility.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: str = "fp32"
    exp_avg_dtype: str = "fp32"
    exp_avg_sq_dtype: str = "fp32"
    # Muon (emerging optimizer) options. Only consumed when `optimizer` selects a Muon algorithm;
    # each field mirrors the like-named field on Megatron-Core's OptimizerConfig and is passed through
    # by `init_megatron_optim_config`. Defaults track Megatron-Core so leaving them unset reproduces
    # Megatron's built-in Muon defaults.
    use_layer_wise_distributed_optimizer: bool = False
    use_layer_wise_param_layout: bool | None = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = False
    muon_split_qkv: bool = True
    muon_scale_mode: str = "spectral"
    muon_coefficient_type: str = "quintic"
    muon_num_ns_steps: int = 5
    muon_tp_mode: str = "blockwise"
    muon_fp32_matmul_prec: str = "medium"
    muon_extra_scale_factor: float = 1.0
    muon_scalar_optimizer: str = "adam"
    # verl-side convenience, not forwarded to Megatron: derives muon_extra_scale_factor
    # from betas[0] instead of hard-coding a constant. See
    # verl.utils.megatron.optimizer.adamw_rms_match_scale_factor.
    muon_match_adamw_update_rms: bool = False
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        allowed_dtypes = {"fp32", "float32", "32", "bf16", "bfloat16"}
        for field_name in ("main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            value = getattr(self, field_name)
            assert str(value) in allowed_dtypes, (
                f"`{field_name}` must be one of {sorted(allowed_dtypes)}, got {value!r}"
            )
        return super().__post_init__()


@dataclass
class TorchtitanOptimizerConfig(OptimizerConfig):
    """Torchtitan optimizer configuration extending base OptimizerConfig.

    Args:
        name (str): Optimizer name; default is "AdamW".
        eps (float): Epsilon value for AdamW optimizer, default 1e-8.
        decay_type (str): Weight decay type: "linear", "sqrt", or "cosine".
        min_lr_factor (float): Minimum learning rate factor.
    """

    name: str = "AdamW"
    eps: float = 1e-8
    decay_type: str = "linear"
    min_lr_factor: float = 0.0


@dataclass
class AutomodelOptimizerConfig(OptimizerConfig):
    """Automodel optimizer configuration extending base OptimizerConfig.

    Uses the same optimizer building mechanism as FSDP (dynamic import from optimizer_impl).
    LR scheduling is handled by Automodel's OptimizerParamScheduler.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim").
        lr (float): Learning rate (maps to max_lr in OptimizerParamScheduler).
        init_lr_ratio (Optional[float]): Initial LR ratio for warmup start (init_lr = lr * init_lr_ratio).
        min_lr_ratio (Optional[float]): Minimum LR ratio after decay (min_lr = lr * min_lr_ratio).
        lr_scheduler_type (str): LR decay style: "constant", "cosine", "linear", or "inverse-square-root".
        wd_incr_style (str): Weight decay increment style: "constant", "linear", or "cosine".
        num_cycles (float): Kept for backward compatibility (unused by Automodel scheduler).
        zero_indexed_step (bool): Kept for backward compatibility (unused by Automodel scheduler).
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    init_lr_ratio: Optional[float] = 0.1
    min_lr_ratio: Optional[float] = 0.01
    lr_scheduler_type: str = "cosine"
    wd_incr_style: str = "constant"
    num_cycles: float = 0.5
    zero_indexed_step: bool = True
    # Common optimizer kwargs
    eps: float = 1e-8
    master_weights: bool = False
    store_param_remainders: bool = False
    exp_avg_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    exp_avg_sq_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    master_weight_dtype: Optional[str] = None  # "fp32", "bf16", "fp16", or "torch.float32" etc.
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        assert self.lr_scheduler_type in ["constant", "cosine", "linear", "inverse-square-root"]
        return super().__post_init__()


def _get_weight_decay_parameter_groups(model, weight_decay: float):
    """Split trainable model parameters into decay and no-decay groups."""
    from torch import nn
    from transformers.trainer_pt_utils import get_parameter_names

    normalization_layer_types = [
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ]
    if hasattr(nn, "RMSNorm"):
        normalization_layer_types.append(nn.RMSNorm)

    # Keep this aligned with Hugging Face Trainer's default policy. The explicit
    # layer types cover standard torch normalization modules, while the patterns
    # catch model-specific implementations such as LlamaRMSNorm and QwenRMSNorm.
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
    ]
    decay_parameter_names = set(get_parameter_names(model, normalization_layer_types, forbidden_name_patterns))
    custom_normalization_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if module.__class__.__name__.lower().endswith("norm")
        for parameter in module.parameters(recurse=False)
    }

    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in decay_parameter_names and id(parameter) not in custom_normalization_parameter_ids:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = []
    if decay_parameters:
        parameter_groups.append(
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
                WEIGHT_DECAY_SCALE_KEY: 1.0,
            }
        )
    if no_decay_parameters:
        parameter_groups.append(
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                WEIGHT_DECAY_SCALE_KEY: 0.0,
            }
        )
    if not parameter_groups:
        raise ValueError("Cannot build an optimizer because the model has no trainable parameters.")

    return parameter_groups


def build_optimizer(model_or_parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        model_or_parameters: Model to optimize. An iterable of parameters is accepted only when
            ``weight_decay_policy="all"``.
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    import importlib

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    effective_weight_decay = float(optimizer_args["weight_decay"])
    optimizer_args["weight_decay"] = effective_weight_decay

    from torch import nn

    if isinstance(model_or_parameters, nn.Module):
        if config.weight_decay_policy == "standard":
            parameters = _get_weight_decay_parameter_groups(model_or_parameters, effective_weight_decay)
        else:
            parameters = model_or_parameters.parameters()
    else:
        if config.weight_decay_policy != "all":
            raise ValueError(
                "weight_decay_policy='standard' requires the model itself so parameters can be classified. "
                "Pass an nn.Module to build_optimizer, or use weight_decay_policy='all' for legacy behavior."
            )
        parameters = model_or_parameters

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
