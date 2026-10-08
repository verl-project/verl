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

"""Real FP8 training utilities for verl FSDP training.

Not to be confused with verl.utils.qat (fake-quantization for a future
low-bit export) or verl.utils.fp8_utils.FP8QuantizerHelper (rollout-engine
weight quantization for serving). This module swaps nn.Linear layers for
verl.utils.fp8_training.linear.FP8Linear, which performs real float8_e4m3fn
compute during training itself.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from verl.base_config import BaseConfig
from verl.utils.fp8_training.linear import FP8Linear, FP8Mode
from verl.utils.qat.core import _set_module

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Same include/exclude rule set already battle-tested on the rollout-side
# quantizer (verl/utils/fp8_utils.py:FP8QuantizerHelper) -- ported here rather
# than re-derived, since it's the same model families (q/k/v/o/gate/up/down
# projections), just applied to the training-time Linear instead of a
# serving-engine weight.
_DEFAULT_EXCLUDE_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "layernorm",
    "norm",
    "ln_",
    "embeddings",
    "mlp.gate",  # MoE router -- never quantize
]
_DEFAULT_INCLUDE_PATTERNS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
]


@dataclass
class FP8TrainingConfig(BaseConfig):
    """Configuration for real FP8 weight-storage training (not QAT)."""

    enable: bool = False
    mode: str = "rowwise"  # "rowwise" or "blockwise"
    block_size: list[int] = field(default_factory=lambda: [128, 128])
    ignore_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_EXCLUDE_PATTERNS))
    include_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_INCLUDE_PATTERNS))
    min_features: int = 256  # skip tiny Linears -- quantize overhead can dominate


def _should_quantize_fp8(name: str, module: nn.Module, config: FP8TrainingConfig) -> bool:
    if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
        return False

    name_lower = name.lower()
    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], name):
                return False
        elif pattern in name_lower:
            return False

    if not any(pattern in name_lower for pattern in config.include_patterns):
        return False

    if module.in_features < config.min_features or module.out_features < config.min_features:
        logger.debug(f"Skipping {name}: below min_features={config.min_features}")
        return False

    return True


def apply_fp8_training(model: nn.Module, config: FP8TrainingConfig | dict[str, Any]) -> nn.Module:
    """Replace eligible nn.Linear layers with FP8Linear. Call after the model
    is built and after any dtype cast, before FSDP wrapping -- see
    FSDPEngine._build_model_optimizer, which calls this right alongside
    _apply_qat for exactly that reason."""
    if not isinstance(config, FP8TrainingConfig):
        config = FP8TrainingConfig(**config)

    if not config.enable:
        logger.info("FP8 training is disabled, returning original model")
        return model

    mode = FP8Mode(config.mode.lower())
    block_size = tuple(config.block_size)
    logger.info(f"Applying real FP8 training with mode={mode.value}, block_size={block_size}")

    targets = [(name, module) for name, module in model.named_modules() if _should_quantize_fp8(name, module, config)]
    logger.info(f"Found {len(targets)} Linear layers to convert to FP8Linear")

    for name, module in targets:
        fp8_module = FP8Linear.from_linear(module, mode=mode, block_size=block_size)
        _set_module(model, name, fp8_module)

    logger.info(f"Successfully applied real FP8 to {len(targets)} layers")
    return model
