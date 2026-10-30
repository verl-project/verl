# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# Bootstrap the vendored nemo_automodel before the top-level import below:
# transformer_impl imports nemo_automodel at module import time, and this
# package's __init__ is the only place guaranteed to run first regardless of
# which submodule gets imported.
from .utils import ensure_vendored_nemo_automodel

ensure_vendored_nemo_automodel()

from .transformer_impl import AutomodelEngine, AutomodelEngineWithLMHead  # noqa: E402

__all__ = [
    "AutomodelEngine",
    "AutomodelEngineWithLMHead",
]
