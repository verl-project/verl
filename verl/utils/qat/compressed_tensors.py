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
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch

# compressed-tensors ships with the inference engines rather than verl, so it is
# imported lazily to keep this module importable without them.
if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

__all__ = [
    "MOE_TARGET",
    "SUPPORTED_SCHEMES",
    "CompressedTensorsWeightQuantizer",
    "Fp8ConfigWeightQuantizer",
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
# Trainers export these checkpoints under their own names, in their mixed FP8/FP4 layout.
_DEEPSEEK_V4_MODEL_TYPES = frozenset({"deepseek_v4", "deepseek_v41"})


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


def _preset_of(scheme: "QuantizationScheme") -> Optional[str]:
    """The supported preset ``scheme`` quantizes like, ignoring observers and targets."""

    for preset in SUPPORTED_SCHEMES:
        reference = _scheme_for_preset(preset, scheme.targets)
        if all(
            _numerics(getattr(scheme, kind)) == _numerics(getattr(reference, kind))
            for kind in ("weights", "input_activations", "output_activations")
        ):
            return preset
    return None


def _group_schemes(ct_config: Mapping[str, Any]) -> list["QuantizationScheme"]:
    from compressed_tensors.quantization import QuantizationScheme

    return [
        QuantizationScheme.model_validate({k: v for k, v in group.items() if k != "format"})
        for group in ct_config["config_groups"].values()
    ]


def qat_modes_for_config(ct_config: Mapping[str, Any]) -> set[Optional[str]]:
    """ModelOpt QAT modes whose fake quant matches the config groups of a rollout ``quantization_config``."""
    presets = {_preset_of(scheme) for scheme in _group_schemes(ct_config)}
    if presets == {"MXFP4", "MXFP8"}:
        return {"mxfp4_mxfp8"}
    return {SUPPORTED_SCHEMES[preset] for preset in presets}


def is_all_mxfp8_config(ct_config: Mapping[str, Any]) -> bool:
    """Whether every compressed-tensors group uses the MXFP8 preset."""
    if ct_config.get("quant_method") != "compressed-tensors" or not ct_config.get("config_groups"):
        return False
    return all(_preset_of(scheme) == "MXFP8" for scheme in _group_schemes(ct_config))


def _module_name_for_quant_match(module: str) -> str:
    """Match exported LoRA base weights against unwrapped HF FP8 exclusions."""
    return module.removesuffix(".base_layer").removeprefix("model.")


def _matches(name: str, targets: Iterable[str]) -> bool:
    for target in targets:
        if target.startswith("re:"):
            if re.match(target[3:], name):
                return True
        elif name == target:
            return True
    return False


def _scheme_for_preset(preset: str, targets: list[str]) -> "QuantizationScheme":
    """Resolve verl presets, including the MXFP4-weight/MXFP8-input combination."""
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(preset, targets)
    if preset == "MXFP4":
        mxfp8 = preset_name_to_scheme("MXFP8", targets)
        scheme = scheme.model_copy(update={"input_activations": mxfp8.input_activations})
    return scheme


def _config_group(targets: list[str], preset: str) -> dict[str, Any]:
    from compressed_tensors.compressors.format import infer_module_format

    scheme = _scheme_for_preset(preset, targets)
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
) -> dict[str, Any]:
    """Build the HF compressed-tensors ``quantization_config`` for the rollout engine.

    Prefer ``RolloutQuantizationConfig.build_quantization_config``; the arguments mirror its fields.
    """
    ignore = list(ignore or [])
    if config_file:
        if linear or moe or ignore:
            raise ValueError("rollout.quantization.config_file is exclusive with linear / moe / ignore")
        return _load_config_file(config_file)

    linear = _check_scheme("linear", linear)
    moe = _check_scheme("moe", moe)
    if linear in ("MXFP4", "NVFP4"):
        raise ValueError(f"{linear} is supported only for routed MoE experts; use BF16 or MXFP8 dense linear layers")
    if linear is None and moe is None:
        raise ValueError("rollout quantization needs at least one of `linear` / `moe`")

    ignore = DEFAULT_IGNORE + ignore

    config_groups: dict[str, Any] = {}
    if linear == moe:
        config_groups["group_0"] = _config_group(["Linear"], linear)
    else:
        # A regex target wins over the class name, so experts always get the MoE scheme.
        if moe is not None:
            config_groups["group_moe"] = _config_group([MOE_TARGET], moe)
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
        self.passthrough = bool(_DEEPSEEK_V4_MODEL_TYPES & set(model_types))
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
        self.ignore = [
            _module_name_for_quant_match(module)
            for module in (fp8_config.get("ignored_layers") or fp8_config.get("modules_to_not_convert") or [])
        ]
        self.ignore_match_mode = fp8_config.get("ignored_layers_match_mode", "exact")
        if self.ignore_match_mode not in ("exact", "substring", "suffix"):
            raise ValueError(f"Unsupported ignored_layers_match_mode={self.ignore_match_mode!r}")
        self.scheme = _scheme_for_preset("FP8_BLOCK", ["Linear"])

    def _is_ignored(self, module: str) -> bool:
        module = _module_name_for_quant_match(module)
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


def _vllm_hf_overrides_quantization_config(rollout_config: Any) -> tuple[bool, Any]:
    """``(present, value)`` of a ``quantization_config`` set in ``rollout.engine_kwargs.vllm.hf_overrides``.

    vLLM applies these overrides on top of the checkpoint config, so the rollout
    runs whatever they set. A top-level ``null`` still allows a text-level
    quantization config; without one it selects a BF16 rollout.
    """
    if getattr(rollout_config, "name", None) != "vllm":
        return False, None
    engine_kwargs = getattr(rollout_config, "engine_kwargs", None) or {}
    hf_overrides = (engine_kwargs.get("vllm") or {}).get("hf_overrides") or {}
    if isinstance(hf_overrides, str):
        hf_overrides = json.loads(hf_overrides)
    if "quantization_config" not in hf_overrides:
        return False, None
    return True, hf_overrides["quantization_config"]


def dispatch_rollout_weight_quantizer(
    *,
    model_config: Any,
    rollout_config: Any,
    qat_config: Any = None,
) -> Optional[Any]:
    """Select the trainer-side exporter for native or overridden rollout quantization.

    The rollout engine keeps its native ``quantization_config`` unless
    ``rollout.quantization`` explicitly overrides it; a ``quantization_config``
    in vLLM's ``hf_overrides`` replaces the native one. Both native and override
    compressed-tensors configs use :class:`CompressedTensorsWeightQuantizer`;
    native ``Fp8Config`` uses :class:`Fp8ConfigWeightQuantizer`. DeepSeek-V4
    still gets an FP8 quantizer object, but it passes through the private mixed
    FP8/FP4 layout already produced by Megatron-Bridge.
    """
    rollout_quantization = getattr(rollout_config, "quantization", None)
    linear = getattr(rollout_quantization, "linear", None)
    moe = getattr(rollout_quantization, "moe", None)
    ignore = getattr(rollout_quantization, "ignore", None) or []
    config_file = getattr(rollout_quantization, "config_file", None)
    has_override = bool(linear or moe or config_file)

    hf_config = model_config.hf_config
    model_types = {
        getattr(hf_config, "model_type", None),
        getattr(getattr(hf_config, "text_config", None), "model_type", None),
    }
    if has_override:
        if model_types & _DEEPSEEK_V4_MODEL_TYPES:
            raise NotImplementedError(
                "rollout.quantization overrides are not supported for DeepSeek-V4: the trainer exports its "
                "checkpoint names (e.g. ffn.experts) and its own FP8/FP4 layout, which the override's "
                "compressed-tensors targets do not match. Drop the override to sync the native quantization."
            )
        quantization_config = build_compressed_tensors_config(
            linear=linear,
            moe=moe,
            ignore=ignore,
            config_file=config_file,
            model_path=model_config.local_path,
        )
    else:
        has_hf_override, native = _vllm_hf_overrides_quantization_config(rollout_config)
        if not has_hf_override:
            native = getattr(hf_config, "quantization_config", None)
        if native is None:
            native = getattr(getattr(hf_config, "text_config", None), "quantization_config", None)
        quantization_config = _as_quantization_config(native)
        if (
            has_hf_override
            and model_types & _DEEPSEEK_V4_MODEL_TYPES
            and quantization_config is not None
            and str(quantization_config.get("quant_method", "")).lower() == "compressed-tensors"
        ):
            raise NotImplementedError(
                "hf_overrides compressed-tensors quantization overrides are not supported for DeepSeek-V4: "
                "the trainer exports its native FP8/FP4 layout. Drop the override to sync the native quantization."
            )

    qat_mode = qat_config.mode if qat_config is not None and qat_config.enable else None
    if quantization_config is None:
        if qat_mode is not None:
            raise ValueError(
                "QAT needs a quantized rollout to sync to: use a model with a native "
                "quantization_config or set rollout.quantization.linear / moe / config_file"
            )
        return None
    if getattr(getattr(rollout_config, "checkpoint_engine", None), "backend", None) == "delta_sharded":
        raise NotImplementedError(
            "checkpoint_engine.backend='delta_sharded' does not support quantized rollouts: its steady-state "
            "deltas bypass the trainer-side rollout weight quantizer"
        )

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method == "compressed-tensors":
        qat_modes = qat_modes_for_config(quantization_config) if qat_mode is not None else set()
        if getattr(rollout_config, "name", None) == "sglang" and is_all_mxfp8_config(quantization_config):
            from verl.utils.sglang.sglang_quant_utils import SGLangMxfp8WeightQuantizer

            quantizer = SGLangMxfp8WeightQuantizer(quantization_config)
        else:
            quantizer = CompressedTensorsWeightQuantizer(quantization_config)
    elif "fp8" in quant_method:
        qat_modes = {"fp8"}
        quantizer = Fp8ConfigWeightQuantizer(quantization_config, model_types=model_types)
    else:
        raise NotImplementedError(
            f"Trainer-side rollout weight export does not support native quant_method={quant_method!r}"
        )

    if qat_mode is not None and qat_modes != {qat_mode}:
        raise ValueError(f"QAT mode {qat_mode!r} does not match rollout quantization (QAT modes {qat_modes})")
    return quantizer
