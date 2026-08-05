# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Backend-specific extensions of :class:`verl.trainer.config.CheckpointConfig`.

The base :class:`CheckpointConfig` lives in ``verl/trainer/config/config.py`` and
carries only fields that every backend understands (``save_contents``,
``load_contents``, ``async_save``). Anything that is meaningful only to one
training backend (e.g. mbridge options for Megatron) goes into a subclass here,
mirroring how ``ActorConfig`` / ``McoreActorConfig`` are split between
``verl/trainer/config`` and ``verl/workers/config``.
"""

from dataclasses import dataclass, field
from typing import Any

from verl.trainer.config import CheckpointConfig

__all__ = ["McoreCheckpointConfig", "MindSpeedCheckpointConfig"]


@dataclass
class McoreCheckpointConfig(CheckpointConfig):
    """Checkpoint config for the Megatron-Core backend.

    Adds the mbridge-specific knobs consumed by
    :class:`verl.utils.checkpoint.megatron_checkpoint_manager.MegatronCheckpointManager`
    when it forwards kwargs to ``bridge.save_weights()``.

    Args:
        mbridge_config (dict[str, Any]): Extra kwargs forwarded to
            ``bridge.save_weights``. Typical keys include
            ``distributed_filesystem`` and ``memory_efficient`` for the
            ``vanilla_mbridge`` path. Keys that are not accepted by the active
            bridge's ``save_weights`` signature are silently ignored.
    """

    mbridge_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MindSpeedCheckpointConfig(McoreCheckpointConfig):
    """Checkpoint config for the MindSpeed backend.

    MindSpeed reuses the Megatron checkpoint manager and therefore inherits the
    mbridge knob from :class:`McoreCheckpointConfig`. A dedicated class is kept
    so MindSpeed-only fields (should any appear in the future) have a natural
    home and so the ``_target_`` in MindSpeed yamls mirrors the backend name.
    """
