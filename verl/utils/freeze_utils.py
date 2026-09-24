# Copyright 2025 Individual Contributor: Wu Zehuan
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
"""Freeze module parameters by regex pattern.

Used by both FSDP and Megatron engines to selectively freeze model
parameters during training (e.g. freeze vision encoder in VLMs).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_freeze_to_module(
    module: nn.Module,
    freeze_pattern: str,
) -> int:
    """Freeze parameters whose fully-qualified name matches the regex pattern.

    Sets ``requires_grad = False`` on matching parameters before FSDP
    wrapping or Megatron/VeOmni/TorchTitan optimizer construction.

    Args:
        module: The loaded model (before FSDP/DDP wrapping).
        freeze_pattern: Regex pattern to match against parameter names.
            e.g. ``"model\\.visual"`` to match all visual model params.

    Returns:
        Number of parameters frozen.

    Raises:
        ValueError: If the regex pattern is invalid.
    """
    if not freeze_pattern:
        return 0

    try:
        pattern = re.compile(freeze_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid freeze_module_pattern regex: '{freeze_pattern}' — {exc}") from exc

    frozen = 0
    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = False
            frozen += 1

    if frozen == 0:
        logger.warning(
            "freeze_module_pattern '%s' matched 0 parameters. Check your regex.  Example parameter names: %s",
            freeze_pattern,
            ", ".join(n for i, (n, _) in enumerate(module.named_parameters()) if i < 5),
        )
    else:
        logger.info(
            "Freeze applied: %d parameters frozen (pattern: '%s')",
            frozen,
            freeze_pattern,
        )

    return frozen
