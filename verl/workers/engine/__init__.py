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
import warnings

from .base import BaseEngine, EngineRegistry
from .fsdp import FSDPEngine, FSDPEngineWithLMHead, FSDPTurboEngineWithLMHead

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
    "FSDPTurboEngineWithLMHead",
]

try:
    from .torchtitan import TorchTitanEngine, TorchTitanEngineWithLMHead

    __all__ += ["TorchTitanEngine", "TorchTitanEngineWithLMHead"]
except ImportError as e:
    warnings.warn(f"torchtitan engine is not available: {e!r}", stacklevel=1)
    TorchTitanEngine = None
    TorchTitanEngineWithLMHead = None

try:
    from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

    __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
except ImportError as e:
    warnings.warn(f"veomni engine is not available: {e!r}", stacklevel=1)
    VeOmniEngine = None
    VeOmniEngineWithLMHead = None

try:
    from .automodel import AutomodelEngine, AutomodelEngineWithLMHead

    __all__ += ["AutomodelEngine", "AutomodelEngineWithLMHead"]
except ImportError as e:
    warnings.warn(f"automodel engine is not available: {e!r}", stacklevel=1)
    AutomodelEngine = None
    AutomodelEngineWithLMHead = None

# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead, MindspeedEngineWithValueHead

    __all__ += ["MindspeedEngineWithLMHead", "MindspeedEngineWithValueHead"]
except ImportError as e:
    warnings.warn(f"mindspeed engine is not available: {e!r}", stacklevel=1)
    MindspeedEngineWithLMHead = None
    MindspeedEngineWithValueHead = None

try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead, MegatronEngineWithValueHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead", "MegatronEngineWithValueHead"]
except ImportError as e:
    warnings.warn(f"megatron engine is not available: {e!r}", stacklevel=1)
    MegatronEngine = None
    MegatronEngineWithLMHead = None
