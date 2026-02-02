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

import torch
import torch.nn as nn

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
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

    if module.in_features % config.group_size != 0:
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

    if isinstance(config, dict):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    mode = QATMode(config.mode.lower())
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
        )

        _set_module(model, name, fake_quant_module)
        logger.debug(f"Converted {name} to QATLinear")

    model._qat_config = config

    converted_count = sum(1 for name, m in model.named_modules() if isinstance(m, QATLinear))
    logger.info(f"Successfully applied QAT to {converted_count} layers")

    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _create_decoder_layer_fuse_hook():
    """Create a forward pre-hook for DecoderLayer to compute fused scales."""
    from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA, FP8_E4M3_DATA

    from verl.utils.qat.linear import QATLinear, QATMode

    def _apply_fused_input_scale(modules: list, force: bool = False):
        """Apply fused input scale to a group of W4A4 modules."""
        w4a4_modules = []
        amaxes = []
        for m in modules:
            if m.mode != QATMode.W4A4:
                continue
            if not hasattr(m, "input_amax"):
                continue
            if m.input_amax.item() == m._UNINITIALIZED_SCALE:
                continue
            w4a4_modules.append(m)
            amaxes.append(m.input_amax)

        if len(w4a4_modules) < 2:
            return

        if not force:
            first_amax = amaxes[0].item()
            all_equal = all(abs(a.item() - first_amax) < 1e-6 for a in amaxes)
            if all_equal:
                return

        fused_amax = torch.max(torch.stack(amaxes))
        scale_factor = FP8_E4M3_DATA.max * FP4_E2M1_DATA.max
        fused_scale = (scale_factor / (fused_amax + 1e-12)).float().view(1)

        for m in w4a4_modules:
            m.input_amax.copy_(fused_amax.view(1).to(m.input_amax.device))
            m.input_global_scale.copy_(fused_scale.to(m.input_global_scale.device))

    def fuse_hook(module, inputs):
        """Fuse INPUT scales for W4A4 mode only."""
        with torch.no_grad():
            all_qat_modules = {}
            for name, submodule in module.named_modules():
                if isinstance(submodule, QATLinear):
                    if submodule.weight.device == torch.device("meta"):
                        continue
                    all_qat_modules[name] = submodule

            if not all_qat_modules:
                return None

            # QKV W4A4 input fusion
            qkv_patterns = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
            qkv_modules = []
            for name, m in all_qat_modules.items():
                if any(p in name or name == p.split(".")[-1] for p in qkv_patterns):
                    qkv_modules.append(m)
            _apply_fused_input_scale(qkv_modules, force=module.training)

            # Gate/Up W4A4 input fusion
            gate_up_groups = {}
            for name, m in all_qat_modules.items():
                if name.endswith("gate_proj") or name.endswith("up_proj"):
                    parent = name.rsplit(".", 1)[0]
                    proj_type = name.rsplit(".", 1)[1]
                    if parent not in gate_up_groups:
                        gate_up_groups[parent] = {}
                    gate_up_groups[parent][proj_type] = m

            for parent, projs in gate_up_groups.items():
                if "gate_proj" in projs and "up_proj" in projs:
                    _apply_fused_input_scale([projs["gate_proj"], projs["up_proj"]], force=module.training)

        return None

    return fuse_hook


def register_fused_scale_hooks(model: nn.Module):
    """Register forward pre-hooks on DecoderLayers to compute fused scales."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    decoder_layer_classes = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "DecoderLayer" in class_name or "TransformerBlock" in class_name:
            decoder_layer_classes.add(type(module))

    if not decoder_layer_classes:
        logger.warning("[QAT Fuse] No DecoderLayer found in model")
        return []

    logger.info(f"[QAT Fuse] Found DecoderLayer classes: {[c.__name__ for c in decoder_layer_classes]}")

    hooks = []
    for name, module in model.named_modules():
        actual_module = module
        if isinstance(module, FSDP):
            actual_module = module._fsdp_wrapped_module if hasattr(module, "_fsdp_wrapped_module") else module

        if type(actual_module) in decoder_layer_classes:
            hook_fn = _create_decoder_layer_fuse_hook()
            if hook_fn:
                handle = actual_module.register_forward_pre_hook(hook_fn)
                hooks.append(handle)
                logger.debug(f"[QAT Fuse] Registered fuse hook on {name}")

    logger.info(f"[QAT Fuse] Registered {len(hooks)} fused scale hooks on DecoderLayers")

    model._qat_fuse_hooks = hooks

    return hooks


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            qat_modules[name] = module

    # Setup QKV fusion siblings
    qkv_groups = {}
    for name, module in qat_modules.items():
        for proj in ["q_proj", "k_proj", "v_proj"]:
            if name.endswith(proj):
                parent = name.rsplit(".", 1)[0]
                if parent not in qkv_groups:
                    qkv_groups[parent] = {}
                qkv_groups[parent][proj] = module

    qkv_count = 0
    for parent, projs in qkv_groups.items():
        if len(projs) >= 2:
            modules = list(projs.values())
            for i, m in enumerate(modules):
                siblings = [modules[j] for j in range(len(modules)) if j != i]
                m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
            qkv_count += 1

    # Setup GateUp fusion siblings
    gate_up_groups = {}
    for name, module in qat_modules.items():
        if name.endswith("gate_proj") or name.endswith("up_proj"):
            parent = name.rsplit(".", 1)[0]
            proj_type = name.rsplit(".", 1)[1]
            if parent not in gate_up_groups:
                gate_up_groups[parent] = {}
            gate_up_groups[parent][proj_type] = module

    gate_up_count = 0
    for parent, projs in gate_up_groups.items():
        if "gate_proj" in projs and "up_proj" in projs:
            gate = projs["gate_proj"]
            up = projs["up_proj"]
            gate._fusion_siblings_ref = [weakref.ref(up)]
            up._fusion_siblings_ref = [weakref.ref(gate)]
            gate_up_count += 1

    logger.info(f"[QAT Fuse] Setup fusion siblings: {qkv_count} QKV groups, {gate_up_count} GateUp pairs")

    return qkv_count, gate_up_count


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings and registers hooks."""
    setup_fusion_siblings(model)
    register_fused_scale_hooks(model)
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
