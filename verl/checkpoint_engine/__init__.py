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

# Eager: no optional deps and registration is a side-effect of import.
from .remote_backend import RemoteBackendCheckpointEngine  # noqa: F401

__all__ = [
    "CheckpointEngine",
    "CheckpointEngineRegistry",
    "TensorMeta",
    "ColocatedCheckpointEngine",
    "CheckpointEngineManager",
    "CheckpointEngineWorker",
    "RemoteBackendCheckpointEngine",
]

# Every engine below is optional: it carries its own transport dependency, and
# the CUDA / NPU engines are mutually exclusive by construction (hccl registers
# the "nccl" backend name on Ascend). Failures are recorded so that asking for a
# backend whose module did not import reports the missing dependency instead of
# just "not registered".
try:
    from .nccl_checkpoint_engine import NCCLCheckpointEngine

    __all__ += ["NCCLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("nccl_checkpoint_engine", e)
    NCCLCheckpointEngine = None

try:
    from .hccl_checkpoint_engine import HCCLCheckpointEngine

    __all__ += ["HCCLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("hccl_checkpoint_engine", e)
    HCCLCheckpointEngine = None

try:
    from .nixl_checkpoint_engine import NIXLCheckpointEngine

    __all__ += ["NIXLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("nixl_checkpoint_engine", e)
    NIXLCheckpointEngine = None

try:
    from .kimi_checkpoint_engine import KIMICheckpointEngine

    __all__ += ["KIMICheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("kimi_checkpoint_engine", e)
    KIMICheckpointEngine = None

try:
    from .mooncake_checkpoint_engine import MooncakeCheckpointEngine

    __all__ += ["MooncakeCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("mooncake_checkpoint_engine", e)
    MooncakeCheckpointEngine = None

try:
    from .delta_checkpoint_engine import DeltaShardedCheckpointEngine

    __all__ += ["DeltaShardedCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("delta_checkpoint_engine", e)
    DeltaShardedCheckpointEngine = None
