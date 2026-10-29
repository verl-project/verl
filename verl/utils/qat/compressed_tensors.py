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
"""Low-precision rollout expressed as a compressed-tensors ``quantization_config``.

The rollout config either selects a compressed-tensors preset scheme per layer kind::

    quantization:
      linear: FP8_BLOCK  # dense linear layers (attention, dense MLP, shared experts)
      moe: FP8_BLOCK     # routed experts
      ignore: []         # extra "re:" patterns kept in high precision

or provides a complete ``quantization_config`` JSON via ``quantization.config_file``.

``RolloutQuantizationConfig.build_quantization_config`` (via
``build_compressed_tensors_config``) turns that into the HF ``quantization_config``
the inference engine is started with (``load_format=dummy``), and
``CompressedTensorsWeightQuantizer`` quantizes the trainer's exported HF weights
with exactly the same targets and ignore list, using the compressed-tensors
compressors, so every refit ships tensors in the layout the engine loads.
"""

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch

# compressed-tensors ships with the inference engines rather than verl, so it is
# imported lazily to keep this module importable without them.
if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

__all__ = [
    "MOE_TARGET",
    "MOE_ACTIVATIONS",
    "SUPPORTED_SCHEMES",
    "CompressedTensorsWeightQuantizer",
    "Fp8ConfigWeightQuantizer",
    "NVFP4PerTokenWeightQuantizer",
    "is_nvfp4_per_token_config",
    "is_nvfp4_per_token_scheme",
    "nvfp4_per_token_module_matcher",
    "dispatch_rollout_weight_quantizer",
    "build_compressed_tensors_config",
    "is_all_mxfp8_config",
    "qat_modes_for_config",
]

# Matched against vLLM's unfused expert names (``...mlp.experts.0.gate_proj``),
# which is how compressed-tensors resolves fused MoE layers.
MOE_TARGET = r"re:.*mlp\.experts\..*"

# Presets whose rollout refit path (weight reload + process_weights_after_loading)
# has been validated, mapped to the ModelOpt QAT mode that fake-quantizes identically.
SUPPORTED_SCHEMES: dict[str, Optional[str]] = {
    "FP8_BLOCK": "fp8",
    "MXFP8": "mxfp8",
    # There is no compressed-tensors preset for this combination.  verl's
    # MXFP4 mode deliberately means MXFP4 expert weights with dynamically
    # quantized MXFP8 expert inputs, rather than the upstream MXFP4 W4A4
    # preset.
    "MXFP4": "mxfp4",
    # NVFP4 is deliberately W4A4 and routed-expert-only.  Unlike MX formats,
    # its tensor-level activation scale is static and must come from QAT.
    "NVFP4": "nvfp4",
}

# ``moe_activation`` overrides, named as vLLM's ``quantization_config.moe.activation``,
# mapped to the ``moe`` preset each applies to. ``nvfp4_per_token`` quantizes the
# NVFP4 expert inputs per token at runtime, as Transformer Engine's row-scaled NVFP4
# training does; it pairs with ``override_transformer_config.fp4``, not QAT.
MOE_ACTIVATIONS: dict[str, str] = {
    "nvfp4_per_token": "NVFP4",
}

# Layers that stay in high precision regardless of model. Patterns must start
# with "re:": vLLM rewrites plain dotted names through the model's
# WeightsMapper, but leaves regexes alone, so these match on both sides.
DEFAULT_IGNORE = [
    r"re:.*lm_head$",
    r"re:.*embed_tokens$",
    r"re:.*visual\..*",
    r"re:^mtp\..*",
    r"re:.*mlp\.gate$",
    r"re:.*shared_expert_gate$",
]


_HIGH_PRECISION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _check_scheme(kind: str, scheme: Optional[str]) -> Optional[str]:
    if scheme is None:
        return None
    scheme = scheme.upper()
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported rollout quantization {kind}={scheme!r}, choose from {sorted(SUPPORTED_SCHEMES)}")
    return scheme


_NUMERIC_FIELDS = {
    "num_bits",
    "type",
    "symmetric",
    "group_size",
    "strategy",
    "block_structure",
    "dynamic",
    "scale_dtype",
}


def _numerics(args: Optional["QuantizationArgs"]) -> Optional[dict[str, Any]]:
    return None if args is None else args.model_dump(include=_NUMERIC_FIELDS)


def _preset_of(scheme: "QuantizationScheme") -> Optional[tuple[str, Optional[str]]]:
    """The supported ``(preset, moe_activation)`` ``scheme`` quantizes like, ignoring observers and targets."""

    variants = [(preset, None) for preset in SUPPORTED_SCHEMES] + [
        (preset, activation) for activation, preset in MOE_ACTIVATIONS.items()
    ]
    for preset, activation in variants:
        reference = _scheme_for_preset(preset, scheme.targets, activation)
        if all(
            _numerics(getattr(scheme, kind)) == _numerics(getattr(reference, kind))
            for kind in ("weights", "input_activations", "output_activations")
        ):
            return preset, activation
    return None


def _group_schemes(ct_config: Mapping[str, Any]) -> list["QuantizationScheme"]:
    from compressed_tensors.quantization import QuantizationScheme

    return [
        QuantizationScheme.model_validate({k: v for k, v in group.items() if k != "format"})
        for group in ct_config["config_groups"].values()
    ]


def qat_modes_for_config(ct_config: Mapping[str, Any]) -> set[Optional[str]]:
    """ModelOpt QAT modes whose fake quant matches the config groups of a rollout ``quantization_config``."""
    variants = {_preset_of(scheme) for scheme in _group_schemes(ct_config)}
    if variants == {("MXFP4", None), ("MXFP8", None)}:
        return {"mxfp4_mxfp8"}
    # No ModelOpt QAT mode matches an activation override.
    return {SUPPORTED_SCHEMES[preset] if activation is None else None for preset, activation in variants}


def is_all_mxfp8_config(ct_config: Mapping[str, Any]) -> bool:
    """Whether every compressed-tensors group uses the MXFP8 preset."""
    if ct_config.get("quant_method") != "compressed-tensors" or not ct_config.get("config_groups"):
        return False
    return all(_preset_of(scheme) == ("MXFP8", None) for scheme in _group_schemes(ct_config))


def _matches(name: str, targets: Iterable[str]) -> bool:
    for target in targets:
        if target.startswith("re:"):
            if re.match(target[3:], name):
                return True
        elif name == target:
            return True
    return False


def _scheme_for_preset(preset: str, targets: list[str], moe_activation: Optional[str] = None) -> "QuantizationScheme":
    """Resolve verl presets, including the MXFP4-weight/MXFP8-input combination."""
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(preset, targets)
    if moe_activation == "nvfp4_per_token":
        # Fully dynamic inputs: no static global scale, the kernel derives one per token.
        inputs = scheme.input_activations.model_copy(update={"dynamic": True, "observer": None})
        return scheme.model_copy(update={"input_activations": inputs})
    if preset == "MXFP4":
        mxfp8 = preset_name_to_scheme("MXFP8", targets)
        scheme = scheme.model_copy(update={"input_activations": mxfp8.input_activations})
    return scheme


def _is_nvfp4_args(args: Any) -> bool:
    if args is None:
        return False
    get = args.get if isinstance(args, Mapping) else lambda key: getattr(args, key, None)
    strategy, kind = get("strategy"), get("type")
    return (
        getattr(strategy, "value", strategy) == "tensor_group"
        and getattr(kind, "value", kind) == "float"
        and get("num_bits") == 4
        and get("group_size") == 16
        and bool(get("symmetric"))
    )


def is_nvfp4_per_token_scheme(weights: Any, input_activations: Any) -> bool:
    """Whether a config group is per-token NVFP4: NVFP4 weights and fully dynamic NVFP4 inputs."""
    if not (_is_nvfp4_args(weights) and _is_nvfp4_args(input_activations)):
        return False
    dynamic = (
        input_activations.get("dynamic")
        if isinstance(input_activations, Mapping)
        else getattr(input_activations, "dynamic", None)
    )
    return dynamic is True


def is_nvfp4_per_token_config(quantization_config: Any) -> bool:
    """Whether a ``quantization_config`` (dict or HF config object) has a per-token NVFP4 group."""
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, Mapping) or quantization_config.get("quant_method") != "compressed-tensors":
        return False
    groups = (quantization_config.get("config_groups") or {}).values()
    return any(is_nvfp4_per_token_scheme(group.get("weights"), group.get("input_activations")) for group in groups)


def nvfp4_per_token_module_matcher(ct_config: Mapping[str, Any]) -> Callable[[str], bool]:
    """Predicate on an unfused expert projection name (``...experts.0.gate_proj``):
    whether ``ct_config`` quantizes it in per-token NVFP4.

    Uses the exporter's own name matching, so it covers any model the export does.
    """
    module_scheme = CompressedTensorsWeightQuantizer(ct_config)._module_scheme

    def matches(module: str) -> bool:
        scheme = module_scheme(module)
        return scheme is not None and is_nvfp4_per_token_scheme(scheme.weights, scheme.input_activations)

    return matches


def _config_group(targets: list[str], preset: str, moe_activation: Optional[str] = None) -> dict[str, Any]:
    from compressed_tensors.compressors.format import infer_module_format

    scheme = _scheme_for_preset(preset, targets, moe_activation)
    group = scheme.model_dump(mode="json", exclude_none=True)
    group["format"] = infer_module_format(torch.nn.Linear, scheme).value
    return group


def _load_config_file(config_file: str) -> dict[str, Any]:
    """Read a compressed-tensors ``quantization_config`` (bare, or inside an HF ``config.json``)."""
    with open(config_file) as f:
        config = json.load(f)
    config = config.get("quantization_config", config)
    if config.get("quant_method") != "compressed-tensors" or not config.get("config_groups"):
        raise ValueError(f"{config_file} is not a compressed-tensors quantization_config with config_groups")
    for name, scheme in zip(config["config_groups"], _group_schemes(config), strict=True):
        if _preset_of(scheme) is None:
            raise ValueError(
                f"{config_file}: config group {name!r} does not match any supported scheme {sorted(SUPPORTED_SCHEMES)}"
            )
    return config


def build_compressed_tensors_config(
    *,
    linear: Optional[str],
    moe: Optional[str],
    ignore: Iterable[str],
    config_file: Optional[str],
    model_path: Optional[str] = None,
    moe_activation: Optional[str] = None,
) -> dict[str, Any]:
    """Build the HF compressed-tensors ``quantization_config`` for the rollout engine.

    Prefer ``RolloutQuantizationConfig.build_quantization_config``; the arguments mirror its fields.
    """
    ignore = list(ignore or [])
    if config_file:
        if linear or moe or moe_activation or ignore:
            raise ValueError(
                "rollout.quantization.config_file is exclusive with linear / moe / moe_activation / ignore"
            )
        return _load_config_file(config_file)

    linear = _check_scheme("linear", linear)
    moe = _check_scheme("moe", moe)
    if moe_activation is not None and MOE_ACTIVATIONS.get(moe_activation) != moe:
        supported = {activation: f"moe={preset}" for activation, preset in MOE_ACTIVATIONS.items()}
        raise ValueError(
            f"Unsupported rollout.quantization.moe_activation={moe_activation!r} with moe={moe!r}: {supported}"
        )
    if linear in ("MXFP4", "NVFP4"):
        raise ValueError(f"{linear} is supported only for routed MoE experts; use BF16 or MXFP8 dense linear layers")
    if linear is None and moe is None:
        raise ValueError("rollout quantization needs at least one of `linear` / `moe`")

    ignore = DEFAULT_IGNORE + ignore

    config_groups: dict[str, Any] = {}
    if linear == moe and moe_activation is None:
        config_groups["group_0"] = _config_group(["Linear"], linear)
    else:
        # A regex target wins over the class name, so experts always get the MoE scheme.
        if moe is not None:
            config_groups["group_moe"] = _config_group([MOE_TARGET], moe, moe_activation)
        else:
            ignore.append(MOE_TARGET)
        if linear is not None:
            config_groups["group_linear"] = _config_group(["Linear"], linear)

    formats = {g["format"] for g in config_groups.values()}
    return {
        "quant_method": "compressed-tensors",
        "format": formats.pop() if len(formats) == 1 else "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "sparsity_config": {},
        "global_compression_ratio": None,
    }


def _weight_min_max(weight: torch.Tensor, args: "QuantizationArgs") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile min/max of a 2D weight, shaped like the scale the strategy expects."""
    from compressed_tensors.quantization import QuantizationStrategy
    from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant

    if args.strategy == QuantizationStrategy.BLOCK:
        rows, cols = args.block_structure
        padded = maybe_pad_tensor_for_block_quant(weight, (rows, cols))
        tiles = padded.unflatten(1, (-1, cols)).unflatten(0, (-1, rows))
        return tiles.amin(dim=(1, 3)), tiles.amax(dim=(1, 3))
    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        groups = weight.unflatten(-1, (-1, args.group_size))
        return groups.amin(dim=-1), groups.amax(dim=-1)
    if args.strategy == QuantizationStrategy.CHANNEL:
        return weight.amin(dim=-1, keepdim=True), weight.amax(dim=-1, keepdim=True)
    if args.strategy == QuantizationStrategy.TENSOR:
        return weight.amin(), weight.amax()
    raise NotImplementedError(f"Trainer-side quantization not implemented for strategy {args.strategy}")


def _compress_weight(
    weight: torch.Tensor,
    scheme: "QuantizationScheme",
    *,
    global_scale: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Quantize one module's weight into its compressed-tensors state dict."""
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.compressors.format import infer_module_format
    from compressed_tensors.quantization import QuantizationStrategy
    from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

    args = scheme.weights
    min_vals, max_vals = (t.float() for t in _weight_min_max(weight, args))
    state_dict = {"weight": weight}
    if args.strategy == QuantizationStrategy.TENSOR_GROUP:
        if global_scale is None:
            global_scale = generate_gparam(min_vals.amin(), max_vals.amax())
        state_dict["weight_global_scale"] = global_scale
    scale, zero_point = calculate_qparams(min_vals, max_vals, args, global_scale=global_scale)
    state_dict["weight_scale"] = scale
    if not args.symmetric:
        state_dict["weight_zero_point"] = zero_point
    compressor = BaseCompressor.get_value_from_registry(infer_module_format(torch.nn.Linear, scheme).value)
    return compressor.compress(state_dict, scheme)


# HF checkpoints that pack all routed experts into one 3D tensor (e.g. Qwen3.5 MoE).
_PACKED_EXPERTS = re.compile(r"(?P<prefix>.*\.experts)\.(?P<proj>gate_up_proj|down_proj)(?:\.weight)?$")


def _unpack_experts(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """``experts.gate_up_proj`` [E, 2I, H] / ``experts.down_proj`` [E, H, I] -> per-expert 2D ``*.weight``.

    The engines load per-expert names with their scales, while their packed
    3D loaders only map the weight itself.
    """
    match = _PACKED_EXPERTS.match(name)
    prefix, proj = match["prefix"], match["proj"]
    for expert_id, weight in enumerate(tensor):
        if proj == "gate_up_proj":
            gate, up = weight.chunk(2, dim=0)
            yield f"{prefix}.{expert_id}.gate_proj.weight", gate
            yield f"{prefix}.{expert_id}.up_proj.weight", up
        else:
            yield f"{prefix}.{expert_id}.down_proj.weight", weight


class CompressedTensorsWeightQuantizer:
    """Quantize exported HF weights to match a compressed-tensors rollout config.

    A weight is quantized iff it is a 2D ``*.weight`` whose module is not
    ignored and is targeted by a config group, resolved like compressed-tensors
    does: a name / "re:" target wins over the ``Linear`` class target. Packed 3D
    experts are split per expert first. Everything else passes through unchanged.
    """

    def __init__(
        self,
        ct_config: Mapping[str, Any],
        input_global_scales: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        self.ignore = list(ct_config.get("ignore") or [])
        self.schemes = _group_schemes(ct_config)
        self.linear_scheme = next((s for s in self.schemes if "Linear" in s.targets), None)
        self.input_global_scales = dict(input_global_scales or {})

    def set_input_global_scales(self, scales: Mapping[str, torch.Tensor]) -> None:
        """Install ModelOpt-observed NVFP4 activation scales keyed by HF module name."""
        self.input_global_scales = dict(scales)

    def refresh_input_global_scales(self, model: torch.nn.Module) -> None:
        """Refresh NVFP4 activation scales from ModelOpt before an export."""
        from verl.utils.qat.modelopt import collect_nvfp4_input_global_scales

        scales = collect_nvfp4_input_global_scales(model)
        if scales:
            self.set_input_global_scales(scales)

    def _input_global_scale(self, module: str, scheme: "QuantizationScheme") -> Optional[torch.Tensor]:
        from compressed_tensors.quantization import QuantizationStrategy

        if scheme.input_activations is None or scheme.input_activations.strategy != QuantizationStrategy.TENSOR_GROUP:
            return None
        scale = self.input_global_scales.get(module)
        if scale is None:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$", module)
            if match:
                projection = "gate_up" if match.group(2) in ("gate_proj", "up_proj") else "down"
                scale = self.input_global_scales.get(f"layers.{match.group(1)}.mlp.experts.{projection}")
        if scale is None:
            # The first rollout sync precedes any actor forward, so no observer
            # value exists yet.  One is the neutral CT divisor and leaves the
            # dynamic group-16 FP8 local scales their full representable range.
            scale = torch.ones(1)
        scale = scale.detach().float().reshape(1)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"Invalid input_global_scale for {module!r}: {scale}")
        return scale

    def _compress_packed_experts(
        self,
        name: str,
        tensor: torch.Tensor,
        scheme: "QuantizationScheme",
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Compress packed experts, sharing NVFP4 gate/up tensor scales."""
        from compressed_tensors.quantization import QuantizationStrategy
        from compressed_tensors.quantization.utils import generate_gparam

        unpacked = list(_unpack_experts(name, tensor))
        shared_global_scales: dict[tuple[int, str], torch.Tensor] = {}
        if scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP:
            by_expert: dict[int, list[torch.Tensor]] = {}
            for module_name, weight in unpacked:
                match = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj)\.weight$", module_name)
                if match:
                    by_expert.setdefault(int(match.group(1)), []).append(weight)
            for expert_id, weights in by_expert.items():
                amax = torch.stack([weight.float().abs().amax() for weight in weights]).amax()
                shared_global_scales[(expert_id, "gate_up")] = generate_gparam(-amax, amax)

        for weight_name, weight in unpacked:
            module = weight_name[: -len(".weight")]
            match = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj)$", module)
            global_scale = shared_global_scales.get((int(match.group(1)), "gate_up")) if match else None
            compressed = _compress_weight(weight, scheme, global_scale=global_scale)
            for key, value in compressed.items():
                yield weight_name[: -len("weight")] + key, value
            input_scale = self._input_global_scale(module, scheme)
            if input_scale is not None:
                yield weight_name[: -len("weight")] + "input_global_scale", input_scale

    def _module_scheme(self, module: str) -> Optional["QuantizationScheme"]:
        if _matches(module, self.ignore):
            return None
        for scheme in self.schemes:
            if _matches(module, scheme.targets):
                return scheme
        return self.linear_scheme

    def _scheme_for(self, name: str, tensor: torch.Tensor) -> Optional["QuantizationScheme"]:
        if not name.endswith(".weight") or tensor.ndim != 2 or tensor.dtype not in _HIGH_PRECISION_DTYPES:
            return None
        return self._module_scheme(name[: -len(".weight")])

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            packed = _PACKED_EXPERTS.match(name) if tensor.ndim == 3 else None
            scheme = self._module_scheme(f"{packed['prefix']}.0.gate_proj") if packed else None
            if packed and scheme is not None:
                yield from self._compress_packed_experts(name, tensor, scheme)
                continue
            scheme = self._scheme_for(name, tensor)
            if scheme is None:
                yield name, tensor
                continue
            prefix = name[: -len("weight")]
            for key, value in _compress_weight(tensor, scheme).items():
                yield prefix + key, value
            input_scale = self._input_global_scale(name[: -len(".weight")], scheme)
            if input_scale is not None:
                yield prefix + "input_global_scale", input_scale


_UNPACKED_EXPERT = re.compile(r"(?P<prefix>.*\.experts\.\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.")


class NVFP4PerTokenWeightQuantizer(CompressedTensorsWeightQuantizer):
    """Export per-token NVFP4 routed experts (``moe=NVFP4``, ``moe_activation=nvfp4_per_token``).

    Each expert's ``gate_proj`` and ``up_proj`` share one global scale computed
    over both, since the rollout runs them as one fused GEMM. Weights are
    quantized with :func:`verl.utils.qat.nvfp4.quantize_nvfp4`, which matches
    the kernel vLLM's online ``nvfp4_per_token`` method uses bitwise, so the
    exported values match what that method produces from the same BF16 weights. Activations need no scale: the
    static input scale is exported as 1 and the kernel scales each token.

    With ``bf16_expert_layers`` (NVFP4 training), every exported routed-expert
    weight is checked against the training carve-out under its exported name.
    """

    def __init__(self, ct_config: Mapping[str, Any], *, bf16_expert_layers: Optional[Iterable[int]] = None):
        super().__init__(ct_config)
        self.bf16_expert_layers = None if bf16_expert_layers is None else sorted(set(bf16_expert_layers))

    def _check_carveout(self, name: str, per_token: bool) -> None:
        if self.bf16_expert_layers is None or name.startswith("mtp."):
            return
        layer = _EXPERT_LAYER.search(name)
        if layer is None or not (_UNPACKED_EXPERT.match(name) or _PACKED_EXPERTS.match(name)):
            return
        if per_token == (int(layer.group(1)) not in self.bf16_expert_layers):
            return
        layers = "|".join(str(index) for index in self.bf16_expert_layers)
        hint = f'["re:.*layers\\.({layers})\\.mlp\\.experts\\..*"]' if layers else "[]"
        raise ValueError(
            f"rollout.quantization exports {name} in {'NVFP4' if per_token else 'BF16'}, but NVFP4 training keeps "
            f"the routed experts of layers {self.bf16_expert_layers} in BF16; set rollout.quantization.ignore={hint}"
        )

    def refresh_input_global_scales(self, model: torch.nn.Module) -> None:
        """Per-token activations carry no observed scale."""

    @staticmethod
    def _quantize(weight: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from verl.utils.qat.nvfp4 import quantize_nvfp4

        return quantize_nvfp4(weight, global_scale)

    @staticmethod
    def _global_scale(*weights: torch.Tensor) -> torch.Tensor:
        # vLLM's online NVFP4: (fp4_max * fp8_max) / amax, amax clamped away from zero.
        amax = torch.stack([w.float().abs().amax() for w in weights]).amax().clamp_min(1e-8)
        return (6.0 * torch.finfo(torch.float8_e4m3fn).max / amax).reshape(1)

    def _emit(self, prefix: str, weights: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
        global_scale = self._global_scale(*weights.values())
        names = list(weights)
        packed, scale = self._quantize(torch.cat([weights[name] for name in names]), global_scale)
        rows = [weights[name].shape[0] for name in names]
        input_scale = torch.ones(1, dtype=torch.float32, device=global_scale.device)
        for name, packed_part, scale_part in zip(names, packed.split(rows), scale.split(rows), strict=True):
            module = f"{prefix}.{name}"
            yield f"{module}.weight_packed", packed_part
            yield f"{module}.weight_scale", scale_part
            yield f"{module}.weight_global_scale", global_scale
            yield f"{module}.input_global_scale", input_scale

    def _is_per_token_expert(self, name: str, tensor: torch.Tensor) -> bool:
        packed = _PACKED_EXPERTS.match(name) if tensor.ndim == 3 else None
        if packed is None and (tensor.ndim != 2 or not _UNPACKED_EXPERT.match(name)):
            return False
        if tensor.dtype not in _HIGH_PRECISION_DTYPES:
            return False
        scheme = self._module_scheme(f"{packed['prefix']}.0.gate_proj" if packed else name[: -len(".weight")])
        return scheme is not None and is_nvfp4_per_token_scheme(scheme.weights, scheme.input_activations)

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        # gate_proj waits here for its up_proj, which shares its global scale.
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            per_token = self._is_per_token_expert(name, tensor)
            self._check_carveout(name, per_token)
            if not per_token:
                yield from super().__call__([(name, tensor)])
                continue
            experts = _unpack_experts(name, tensor) if tensor.ndim == 3 else [(name, tensor)]
            for weight_name, weight in experts:
                match = _UNPACKED_EXPERT.match(weight_name)
                prefix, proj = match["prefix"], match["proj"]
                if proj == "down_proj":
                    yield from self._emit(prefix, {proj: weight})
                    continue
                halves = pending.setdefault(prefix, {})
                halves[proj] = weight
                if len(halves) == 2:
                    del pending[prefix]
                    yield from self._emit(prefix, {"gate_proj": halves["gate_proj"], "up_proj": halves["up_proj"]})
        if pending:
            unpaired = sorted(pending)
            raise RuntimeError(f"Per-token NVFP4 export is missing gate_proj/up_proj halves for {unpaired[:4]}")


class Fp8ConfigWeightQuantizer:
    """Export trainer weights in vLLM's native blockwise ``Fp8Config`` layout.

    Unlike :class:`CompressedTensorsWeightQuantizer`, this keeps the rollout's
    native ``Fp8Config`` and therefore emits ``weight_scale_inv`` rather than
    compressed-tensors' ``weight_scale``.  The quantization calculation is
    shared with the compressed-tensors FP8_BLOCK exporter.

    DeepSeek-V4 uses this dispatcher entry too, but the quantizer passes through
    its private mixed FP8/FP4 layout already exported by Megatron-Bridge.
    """

    def __init__(self, fp8_config: Mapping[str, Any], *, model_types: Iterable[Optional[str]] = ()):
        quant_method = str(fp8_config.get("quant_method", "")).lower()
        self.passthrough = bool({"deepseek_v4", "deepseek_v41"} & set(model_types))
        if self.passthrough:
            return
        activation_scheme = fp8_config.get("activation_scheme")
        block_size = fp8_config.get("weight_block_size")
        store_dtype = fp8_config.get("store_dtype")
        if "fp8" not in quant_method:
            raise ValueError(f"Expected an FP8 quantization config, got quant_method={quant_method!r}")
        if activation_scheme != "dynamic" or block_size != [128, 128] or store_dtype is not None:
            raise NotImplementedError(
                "Trainer-side native Fp8Config export currently supports only standard "
                "FP8_BLOCK (dynamic activations, 128x128 weights, no mixed expert store_dtype); "
                f"got activation_scheme={activation_scheme!r}, weight_block_size={block_size!r}, "
                f"store_dtype={store_dtype!r}"
            )
        self.ignore = list(fp8_config.get("ignored_layers") or fp8_config.get("modules_to_not_convert") or [])
        self.ignore_match_mode = fp8_config.get("ignored_layers_match_mode", "exact")
        if self.ignore_match_mode not in ("exact", "substring", "suffix"):
            raise ValueError(f"Unsupported ignored_layers_match_mode={self.ignore_match_mode!r}")
        self.scheme = _scheme_for_preset("FP8_BLOCK", ["Linear"])

    def _is_ignored(self, module: str) -> bool:
        if _matches(module, DEFAULT_IGNORE):
            return True
        if self.ignore_match_mode == "exact":
            return module in self.ignore
        if self.ignore_match_mode == "substring":
            return any(ignored in module for ignored in self.ignore)
        return any(module == ignored or module.endswith(f".{ignored}") for ignored in self.ignore)

    def _compress(self, name: str, weight: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        prefix = name[: -len("weight")]
        compressed = _compress_weight(weight, self.scheme)
        yield name, compressed["weight"]
        yield prefix + "weight_scale_inv", compressed["weight_scale"]

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        if self.passthrough:
            # Megatron-Bridge already exported DeepSeek-V4's private FP8/FP4
            # layout. vLLM allocated the expert bytes as uint8, so only
            # reinterpret the packed values and E8M0 scales before transfer.
            for name, tensor in weights:
                if ".experts." in name and tensor.dtype in (torch.int8, torch.float8_e8m0fnu):
                    tensor = tensor.view(torch.uint8)
                yield name, tensor
            return
        for name, tensor in weights:
            packed = _PACKED_EXPERTS.match(name) if tensor.ndim == 3 else None
            if packed:
                probe = f"{packed['prefix']}.0.gate_proj"
                if self._is_ignored(probe):
                    yield name, tensor
                else:
                    for weight_name, weight in _unpack_experts(name, tensor):
                        yield from self._compress(weight_name, weight)
                continue

            if (
                not name.endswith(".weight")
                or tensor.ndim != 2
                or tensor.dtype not in _HIGH_PRECISION_DTYPES
                or self._is_ignored(name[: -len(".weight")])
            ):
                yield name, tensor
                continue
            yield from self._compress(name, tensor)


def _as_quantization_config(config: Any) -> Optional[dict[str, Any]]:
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"quantization_config must be a mapping, got {type(config).__name__}")


def dispatch_rollout_weight_quantizer(
    *,
    model_config: Any,
    rollout_config: Any,
    qat_config: Any = None,
    real_nvfp4_bf16_layers: Optional[tuple[int, int]] = None,
) -> Optional[Any]:
    """Select the trainer-side exporter for native or overridden rollout quantization.

    The rollout engine keeps its native ``quantization_config`` unless
    ``rollout.quantization`` explicitly overrides it. Both native and override
    compressed-tensors configs use :class:`CompressedTensorsWeightQuantizer`;
    native ``Fp8Config`` uses :class:`Fp8ConfigWeightQuantizer`. DeepSeek-V4
    still gets an FP8 quantizer object, but it passes through the private mixed
    FP8/FP4 layout already produced by Megatron-Bridge.
    Per-token NVFP4 routed experts use :class:`NVFP4PerTokenWeightQuantizer`; NVFP4 training
    (``real_nvfp4_bf16_layers`` set) requires them and the same BF16 layers.
    """
    rollout_quantization = getattr(rollout_config, "quantization", None)
    linear = getattr(rollout_quantization, "linear", None)
    moe = getattr(rollout_quantization, "moe", None)
    moe_activation = getattr(rollout_quantization, "moe_activation", None)
    ignore = getattr(rollout_quantization, "ignore", None) or []
    config_file = getattr(rollout_quantization, "config_file", None)
    has_override = bool(linear or moe or config_file)

    hf_config = model_config.hf_config
    if has_override:
        quantization_config = build_compressed_tensors_config(
            linear=linear,
            moe=moe,
            ignore=ignore,
            config_file=config_file,
            model_path=model_config.local_path,
            moe_activation=moe_activation,
        )
    else:
        native = getattr(hf_config, "quantization_config", None)
        if native is None:
            native = getattr(getattr(hf_config, "text_config", None), "quantization_config", None)
        quantization_config = _as_quantization_config(native)

    qat_mode = qat_config.mode if qat_config is not None and qat_config.enable else None
    real_nvfp4 = real_nvfp4_bf16_layers is not None
    if quantization_config is None:
        if real_nvfp4:
            raise ValueError(
                "override_transformer_config.fp4 needs rollout.quantization.moe=NVFP4 "
                "with moe_activation=nvfp4_per_token"
            )
        if qat_mode is not None:
            raise ValueError(
                "QAT needs a quantized rollout to sync to: use a model with a native "
                "quantization_config or set rollout.quantization.linear / moe / config_file"
            )
        return None

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    per_token = quant_method == "compressed-tensors" and any(
        is_nvfp4_per_token_scheme(scheme.weights, scheme.input_activations)
        for scheme in _group_schemes(quantization_config)
    )
    if per_token and getattr(rollout_config, "name", None) != "vllm":
        raise ValueError("rollout.quantization.moe_activation=nvfp4_per_token is supported only with the vLLM rollout")
    if per_token and qat_mode is not None:
        raise ValueError(
            "rollout.quantization.moe_activation=nvfp4_per_token pairs with override_transformer_config.fp4, not QAT"
        )
    if real_nvfp4 and not per_token:
        raise ValueError(
            "override_transformer_config.fp4 trains routed experts in per-token NVFP4; "
            "set rollout.quantization.moe=NVFP4 with moe_activation=nvfp4_per_token"
        )
    if quant_method == "compressed-tensors":
        qat_modes = qat_modes_for_config(quantization_config) if qat_mode is not None else set()
        if per_token:
            bf16_expert_layers = None
            if real_nvfp4:
                from verl.utils.megatron.low_precision import bf16_layer_indices, num_decoder_layers

                bf16_expert_layers = bf16_layer_indices(num_decoder_layers(hf_config), *real_nvfp4_bf16_layers)
            quantizer = NVFP4PerTokenWeightQuantizer(quantization_config, bf16_expert_layers=bf16_expert_layers)
        elif getattr(rollout_config, "name", None) == "sglang" and is_all_mxfp8_config(quantization_config):
            from verl.utils.sglang.sglang_quant_utils import SGLangMxfp8WeightQuantizer

            quantizer = SGLangMxfp8WeightQuantizer(quantization_config)
        else:
            quantizer = CompressedTensorsWeightQuantizer(quantization_config)
    elif "fp8" in quant_method:
        qat_modes = {"fp8"}
        model_types = {
            getattr(hf_config, "model_type", None),
            getattr(getattr(hf_config, "text_config", None), "model_type", None),
        }
        quantizer = Fp8ConfigWeightQuantizer(quantization_config, model_types=model_types)
    else:
        raise NotImplementedError(
            f"Trainer-side rollout weight export does not support native quant_method={quant_method!r}"
        )

    if qat_mode is not None and qat_modes != {qat_mode}:
        raise ValueError(f"QAT mode {qat_mode!r} does not match rollout quantization (QAT modes {qat_modes})")
    return quantizer
