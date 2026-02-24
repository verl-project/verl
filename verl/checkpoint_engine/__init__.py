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

from .base import (
    CheckpointEngine,
    CheckpointEngineManager,
    CheckpointEngineRegistry,
    CheckpointEngineWorker,
    ColocatedCheckpointEngine,
    TensorMeta,
)

__all__ = [
    "CheckpointEngine",
    "CheckpointEngineRegistry",
    "TensorMeta",
    "ColocatedCheckpointEngine",
    "CheckpointEngineManager",
    "CheckpointEngineWorker",
]

try:
    from .nccl_checkpoint_engine import NCCLCheckpointEngine

    __all__ += ["NCCLCheckpointEngine"]
except ImportError:
    NCCLCheckpointEngine = None

try:
    from .hccl_checkpoint_engine import HCCLCheckpointEngine

    __all__ += ["HCCLCheckpointEngine"]
except ImportError:
    HCCLCheckpointEngine = None


try:
    from .nixl_checkpoint_engine import NIXLCheckpointEngine

    __all__ += ["NIXLCheckpointEngine"]
except ImportError:
    NIXLCheckpointEngine = None
