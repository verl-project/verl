# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Explicit early runtime bootstrap for Megatron-based engines."""

from __future__ import annotations

import os

USE_MEGATRON_ADAPTOR = os.getenv("VERL_USE_MEGATRON_ADAPTOR", "").lower() in {"1", "true", "enabled"}


def bootstrap_megatron_runtime() -> None:
    """Register MegatronAdaptor before any Megatron engine imports."""
    if not USE_MEGATRON_ADAPTOR:
        return

    try:
        from mindspeed_bridge.runtime import state
        from mindspeed_bridge.runtime.auto_register import register_mindspeed_adaptor
    except ImportError as exc:
        raise RuntimeError("VERL_USE_MEGATRON_ADAPTOR=enabled requires MS-Bridge and MegatronAdaptor") from exc
    if not state.MEGATRON_ADAPTOR_REGISTERED_BY_RUNTIME:
        register_mindspeed_adaptor(strict=True)
