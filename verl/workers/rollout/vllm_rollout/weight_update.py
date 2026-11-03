# Copyright 2026 Individual Contributor: yeyunu
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

"""Select an opt-in weight update mode without importing accelerator packages."""

from collections.abc import Mapping


def _config_value(config, key):
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def resolve_weight_update_mode(config) -> str | None:
    """Return the requested native mode, or None to preserve the legacy path.

    Selecting a mode does not validate device availability. The native adapter
    must check its platform, dependencies, and supported topology before starting
    either side of a transfer. Other vLLM backends keep their existing behavior.
    """
    vllm_config = _config_value(_config_value(config, "engine_kwargs"), "vllm")
    transfer_config = _config_value(vllm_config, "weight_transfer_config")
    backend = _config_value(transfer_config, "backend")
    return backend if backend in ("npu_ipc",) else None
