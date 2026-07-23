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

"""Strategy slow-path (fallback) scoring mode.

The mode selected when the sticky short-circuit misses. Canonical YAML strings
(``prefix-load-aware`` / ``least-inflight``) are mapped to these constants at
the config boundary (``KVCAwareStrategyConfig``), so downstream strategy code
references members — never raw strings.
"""

from __future__ import annotations

from enum import Enum


class SlowCut(str, Enum):
    """Fallback scoring mode (used after the sticky short-circuit misses).

    Inherits ``str`` so members interoperate with plain strings: a YAML-loaded
    ``slow_cut: prefix-load-aware`` compares equal to ``SlowCut.PREFIX_LOAD_AWARE``
    and is accepted by ``SlowCut("prefix-load-aware")`` at the config boundary.
    """

    PREFIX_LOAD_AWARE = "prefix-load-aware"  # S = α·S_cache + (1-α)·S_load
    LEAST_INFLIGHT = "least-inflight"  # -INFLIGHT_COUNT (verl GlobalRequestLoadBalancer-style)
    CAPACITY_TOKEN_AWARE = "capacity-token-aware"  # token capacity gate + prefill increment (discrete)
