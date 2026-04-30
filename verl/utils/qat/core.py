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

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    weight_block_size: Optional[list[int]] = None
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None


FP8_QAT_MODES = {"fp8", "w8a8", "w8a16"}
FP4_QAT_MODES = {"w4a4", "w4a16"}
DEFAULT_FP8_WEIGHT_BLOCK_SIZE = [128, 128]


def normalize_qat_mode(mode: str) -> str:
    """Normalize user-facing QAT mode aliases."""
    mode = mode.lower()
    if mode == "fp8":
        return "w8a8"
    return mode


def is_fp8_qat_mode(mode: str) -> bool:
    """Return True when *mode* is an FP8 QAT mode."""
    return mode.lower() in FP8_QAT_MODES


def is_fp4_qat_mode(mode: str) -> bool:
    """Return True when *mode* is an NVFP4 QAT mode."""
    return mode.lower() in FP4_QAT_MODES


def get_fp8_weight_block_size(weight_block_size: Optional[list[int]] = None) -> list[int]:
    """Return the 2D FP8 block size used by vLLM blockwise FP8."""
    if weight_block_size is None:
        return list(DEFAULT_FP8_WEIGHT_BLOCK_SIZE)
    if len(weight_block_size) != 2:
        raise ValueError(f"FP8 weight_block_size must have two entries, got: {weight_block_size}")
    return [int(weight_block_size[0]), int(weight_block_size[1])]


def build_fp8_quantization_config(weight_block_size: Optional[list[int]] = None) -> dict[str, Any]:
    """Build the vLLM quantization config for FP8 QAT rollout."""
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": get_fp8_weight_block_size(weight_block_size),
    }


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        if is_fp8_qat_mode(qat_config.mode):
            return build_fp8_quantization_config(qat_config.weight_block_size)
        raise ValueError("quantization_config_path is required when NVFP4 QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if quant_config.get("quant_method") == "fp8":
        quant_config["weight_block_size"] = get_fp8_weight_block_size(
            quant_config.get("weight_block_size") or qat_config.weight_block_size
        )

    if qat_config.ignore_patterns and quant_config.get("quant_method") in {"compressed-tensors", "modelopt"}:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    if "lora_" in name or ".lora_" in name:
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    if is_fp4_qat_mode(config.mode) and module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model by replacing nn.Linear with QATLinear."""
    from verl.utils.qat.linear import QATLinear, QATMode

    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    mode = QATMode(normalize_qat_mode(config.mode))
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    converted_count = 0
    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            weight_block_size=config.weight_block_size,
            activation_observer=config.activation_observer,
        )

        _set_module(model, name, fake_quant_module)
        converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")

    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
