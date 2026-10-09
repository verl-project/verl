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

"""Private Monarch backend configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from verl.runtime.config import parse_env_vars


@dataclass(frozen=True, slots=True)
class _TorchStoreConfig:
    store_name_prefix: str
    timeout_s: float
    strategy: Literal["host", "local_rank"]
    local_cache_bytes: int


@dataclass(frozen=True, slots=True)
class MonarchRuntimeConfig:
    """Store validated Monarch backend configuration.

    Attributes:
        job_mode: ``current`` loads the job submitted by ``monarch apply``;
            ``process`` creates and owns a local ``ProcessJob``.
        env_vars: Environment variables from root RuntimeConfig (injected).
        worker_ready_timeout_s: Maximum seconds allowed for Monarch host, process,
            actor, and Worker readiness.
        shutdown_timeout_s: Maximum seconds allowed for Runtime-owned worker shutdown.
        object_store: Process-global TorchStore configuration.
    """

    job_mode: Literal["current", "process"]
    env_vars: Mapping[str, str]
    worker_ready_timeout_s: float
    shutdown_timeout_s: float
    object_store: _TorchStoreConfig


def parse_monarch_runtime_config(section: Mapping[str, object] | None) -> MonarchRuntimeConfig:
    """Convert the selected monarch section into private typed config.

    Args:
        section: Selected ``RuntimeConfig["monarch"]`` mapping after root
            ``env_vars`` injection, or None for defaults.

    Returns:
        Validated private Monarch configuration.
    """
    mapping: Mapping[str, object] = {} if section is None else section
    if not isinstance(mapping, Mapping):
        raise TypeError(f'RuntimeConfig["monarch"] must be a mapping, got {type(mapping)!r}')

    if "mesh_name" in mapping:
        raise ValueError("Monarch root mesh name is fixed as 'hosts'; remove mesh_name from RuntimeConfig")

    raw_job_mode = mapping.get("job_mode", "current")
    if not isinstance(raw_job_mode, str):
        raise TypeError(f"job_mode must be a str, got {type(raw_job_mode)!r}")
    if raw_job_mode == "current":
        job_mode: Literal["current", "process"] = "current"
    elif raw_job_mode == "process":
        job_mode = "process"
    else:
        raise ValueError(f"unsupported Monarch job_mode {raw_job_mode!r}; expected 'current' or 'process'")

    worker_ready_timeout_s = _positive_timeout(mapping, "worker_ready_timeout_s")
    shutdown_timeout_s = _positive_timeout(mapping, "shutdown_timeout_s")
    return MonarchRuntimeConfig(
        job_mode=job_mode,
        env_vars=parse_env_vars(mapping.get("env_vars")),
        worker_ready_timeout_s=worker_ready_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        object_store=_parse_object_store_config(mapping.get("object_store")),
    )


def _parse_object_store_config(value: object | None) -> _TorchStoreConfig:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError('RuntimeConfig["monarch"]["object_store"] must be a mapping')
    store_name_prefix = value.get("store_name_prefix", "verl")
    if not isinstance(store_name_prefix, str) or not store_name_prefix:
        raise TypeError(f"store_name_prefix must be a non-empty str, got {store_name_prefix!r}")
    timeout_s = _positive_timeout(value, "timeout_s")
    strategy = value.get("strategy", "host")
    if strategy not in ("host", "local_rank"):
        raise ValueError(f"strategy must be 'host' or 'local_rank', got {strategy!r}")
    local_cache_bytes = value.get("local_cache_bytes", 0)
    if isinstance(local_cache_bytes, bool) or not isinstance(local_cache_bytes, int) or local_cache_bytes < 0:
        raise ValueError("local_cache_bytes must be a nonnegative integer")
    return _TorchStoreConfig(
        store_name_prefix=store_name_prefix,
        timeout_s=timeout_s,
        strategy=strategy,
        local_cache_bytes=local_cache_bytes,
    )


def _positive_timeout(mapping: Mapping[str, object], name: str) -> float:
    value = mapping.get(name, 300.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite positive number, got {type(value)!r}")
    value = float(value)
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value
