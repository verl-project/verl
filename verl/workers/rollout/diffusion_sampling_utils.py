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
from typing import Any


def build_diffusion_backend_sampling_params(
    sampling_params: dict[str, Any],
    *,
    model_extra_configs: dict[str, Any] | None,
    direct_param_names: set[str],
) -> dict[str, Any]:
    """Translate generic diffusion request params into backend sampling kwargs.

    Model-specific fields from *model_extra_configs* are merged first, then
    per-request *sampling_params* are applied on top (request-level overrides
    win).  Keys that appear in *direct_param_names* are promoted to top-level
    backend params; everything else goes into ``extra_args``.
    """
    backend_sampling_params: dict[str, Any] = {}
    extra_args: dict[str, Any] = {}

    # 1. model_extra_configs: model-level defaults (lower priority)
    for key, value in (model_extra_configs or {}).items():
        if value is None:
            continue
        if key in direct_param_names:
            backend_sampling_params[key] = value
        else:
            extra_args[key] = value

    # 2. sampling_params: per-request overrides (higher priority)
    for key, value in sampling_params.items():
        if value is None:
            continue

        if key in direct_param_names:
            backend_sampling_params[key] = value
        else:
            extra_args[key] = value

    if extra_args:
        backend_sampling_params["extra_args"] = extra_args

    return backend_sampling_params
