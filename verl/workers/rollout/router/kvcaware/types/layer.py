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

"""Cache layer constants — backend-agnostic canonical layer names.

Backend-specific medium strings (e.g. vLLM's ``"GPU"``/``"cpu"``) are mapped to
these constants at each backend's decoder boundary. Downstream store and
strategy layers reference cache layers via these constants — never raw backend
strings.
"""

from __future__ import annotations

from enum import Enum


class Layer(str, Enum):
    """Canonical cache-layer names (backend-agnostic).

    Inherits ``str`` so members interoperate with plain strings: YAML-loaded
    ``layer_weights`` keys (``"gpu"``) index the same dict slot as ``Layer.GPU``,
    and ``Layer.GPU == "gpu"`` holds for set/validation comparisons.
    """

    GPU = "gpu"  # GPU — local reverse index
    CPU = "cpu"  # CPU (e.g. mooncake L2)
    SSD = "ssd"  # SSD
