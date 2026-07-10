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

"""Safe persistence helpers for mutable PPO controller state."""

import logging
import math
import numbers
import os
import tempfile
from pathlib import Path

import torch

from verl.trainer.ppo.core_algos import AdaptiveKLController

logger = logging.getLogger(__name__)

KL_CONTROLLER_STATE_FILENAME = "kl_ctrl.pt"
KL_CONTROLLER_STATE_VERSION = 1


def _state_path(checkpoint_dir: str | os.PathLike) -> Path:
    return Path(checkpoint_dir) / KL_CONTROLLER_STATE_FILENAME


def save_adaptive_kl_controller_state(controller: AdaptiveKLController | None, checkpoint_dir: str | os.PathLike):
    """Atomically save the evolving adaptive KL coefficient.

    The payload contains tensors and integers only, so it can be loaded with
    ``weights_only=True`` without executing checkpoint-provided Python code.
    Fixed controllers and disabled KL-in-reward paths have no mutable state to
    persist and are intentionally no-ops.
    """
    if not isinstance(controller, AdaptiveKLController):
        return

    value = float(controller.value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Adaptive KL controller value must be finite and non-negative, got {value}")

    target = _state_path(checkpoint_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(
            {
                "version": torch.tensor(KL_CONTROLLER_STATE_VERSION, dtype=torch.int64),
                "value": torch.tensor(value, dtype=torch.float64),
            },
            temporary,
        )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_adaptive_kl_controller_state(
    controller: AdaptiveKLController | None, checkpoint_dir: str | os.PathLike
) -> bool:
    """Restore adaptive KL state, returning ``False`` for old checkpoints.

    Missing state is expected for checkpoints written before this fix and keeps
    the configured initial coefficient. Present but malformed state fails
    loudly so a corrupted checkpoint cannot silently alter training.
    """
    if not isinstance(controller, AdaptiveKLController):
        return False

    path = _state_path(checkpoint_dir)
    if not path.exists():
        logger.warning("No adaptive KL controller state found at %s; using initial value", path)
        return False

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Adaptive KL controller state must be a dict, got {type(state).__name__}")

    version = state.get("version", torch.tensor(0))
    if not isinstance(version, torch.Tensor) or version.numel() != 1:
        raise ValueError(f"Unsupported adaptive KL controller state version: {version!r}")
    if version.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"Unsupported adaptive KL controller state version: {version!r}")
    version_value = int(version.item())
    if version_value not in (0, 1):
        raise ValueError(f"Unsupported adaptive KL controller state version: {version!r}")

    value = state.get("value")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Adaptive KL controller state value must be a scalar tensor")
        value = value.item()
    elif version_value != 0 or not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise ValueError("Adaptive KL controller state value must be a scalar tensor")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Adaptive KL controller state value must be finite and non-negative, got {value}")

    controller.value = value
    return True
