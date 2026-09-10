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

"""Root RuntimeConfig mapping and selected-backend section helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

RuntimeConfig: TypeAlias = Mapping[str, object]

_ALLOWED_ROOT_KEYS = frozenset({"backend", "env_vars", "ray", "monarch", "topology"})
_BUILTIN_BACKENDS = frozenset({"ray", "monarch"})


def parse_env_vars(value: object | None) -> dict[str, str]:
    """Parse root-level ``env_vars`` into a str-to-str mapping.

    Args:
        value: Root ``RuntimeConfig["env_vars"]`` value, or None when absent.

    Returns:
        Validated environment mapping (empty when ``value`` is None).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f'RuntimeConfig["env_vars"] must be a mapping, got {type(value)!r}')
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f'RuntimeConfig["env_vars"] keys must be str, got {type(key)!r}')
        if not isinstance(item, str):
            raise TypeError(f'RuntimeConfig["env_vars"][{key!r}] must be a str, got {type(item)!r}')
        result[key] = item
    return result


def select_backend(config: RuntimeConfig) -> str:
    """Return the selected backend name after validating the root mapping.

    Args:
        config: Root composition mapping.

    Returns:
        Selected built-in backend name.

    Raises:
        TypeError: Root config is not a mapping or backend is not a string.
        ValueError: Root keys are unknown or backend is unsupported/missing.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"RuntimeConfig must be a mapping, got {type(config)!r}")
    unknown = set(config) - _ALLOWED_ROOT_KEYS
    if unknown:
        unknown_list = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"unknown RuntimeConfig root key(s): {unknown_list}")
    if "backend" not in config:
        raise ValueError('RuntimeConfig missing required key "backend"')
    backend = config["backend"]
    if not isinstance(backend, str):
        raise TypeError(f'RuntimeConfig "backend" must be a str, got {type(backend)!r}')
    if backend not in _BUILTIN_BACKENDS:
        raise ValueError(f"unsupported RuntimeConfig backend {backend!r}; expected one of {sorted(_BUILTIN_BACKENDS)}")
    return backend


def materialize_backend_section(config: RuntimeConfig, backend: str | None = None) -> dict[str, object]:
    """Copy the selected backend section and inject root ``env_vars``.

    ``env_vars`` is a root-level RuntimeConfig field applied to every backend.
    Backend sections must not declare it; this helper injects the parsed root
    value before private parsers run.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"RuntimeConfig must be a mapping, got {type(config)!r}")
    name = backend if backend is not None else select_backend(config)
    raw = config.get(name)
    if raw is None:
        section: dict[str, object] = {}
    else:
        if not isinstance(raw, Mapping):
            raise TypeError(f'RuntimeConfig["{name}"] must be a mapping, got {type(raw)!r}')
        section = dict(raw)
    if "env_vars" in section:
        raise ValueError(f'env_vars belongs at RuntimeConfig root, not in RuntimeConfig["{name}"]')
    section["env_vars"] = parse_env_vars(config.get("env_vars"))
    return section
