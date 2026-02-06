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
"""
Distributed backend selection module.

This module allows users to choose between Ray and YuanRong (ray_adapter) backends
via the DISTRIBUTED_BACKEND environment variable.

Usage:
    Set DISTRIBUTED_BACKEND=yr or DISTRIBUTED_BACKEND=yuanrong to use ray_adapter
    Set DISTRIBUTED_BACKEND=ray or leave unset to use ray (default)

    Import this module at the very beginning of entry points:
        import verl.utils.distributed_backend  # Must be before any other import ray
"""

import os
import sys

_BACKEND = os.getenv("DISTRIBUTED_BACKEND", "ray").lower()

if _BACKEND in ("yr", "yuanrong"):
    try:
        import ray_adapter as _ray_module
    except ImportError:
        raise ImportError(
            f"DISTRIBUTED_BACKEND is set to '{_BACKEND}' but ray_adapter is not installed. "
            "Please install ray_adapter or set DISTRIBUTED_BACKEND=ray to use the default Ray backend."
        ) from None
    # Inject the selected module into sys.modules so that all subsequent
    # 'import ray' statements will use the selected backend
    sys.modules["ray"] = _ray_module
else:
    pass
