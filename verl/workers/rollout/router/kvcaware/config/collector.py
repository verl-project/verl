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

"""Collector config: connection-type tuning parameters shared by all collectors.

Individual collectors are referenced by name (via ``collector_names`` on
strategies); this config carries the shared tuning knobs grouped by
connection type, not per-collector definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ConfigError, _multiline_repr

_DEFAULT_HTTP_POLLING: dict[str, float] = {
    "polling_interval": 5.0,
    "http_timeout": 10.0,
}

_DEFAULT_LONG_CONNECTION: dict[str, float | int] = {
    "base_retry_delay": 1.0,
    "max_retry_delay": 30.0,
    "max_retry_attempts": 5,
    "retry_backoff_factor": 2.0,
}


def _validate_http_polling(params: dict[str, float]) -> None:
    """Validate http_polling dict parameters.

    Only validates keys that are present; absent keys are left to defaults
    at runtime.  Future HTTP polling parameters can be added without
    changing this signature.
    """
    pi = params.get("polling_interval", _DEFAULT_HTTP_POLLING["polling_interval"])
    ht = params.get("http_timeout", _DEFAULT_HTTP_POLLING["http_timeout"])
    if "polling_interval" in params and pi <= 0:
        raise ConfigError(f"polling_interval must be > 0, got {pi}")
    if "http_timeout" in params and ht <= 0:
        raise ConfigError(f"http_timeout must be > 0, got {ht}")


def _validate_long_connection(params: dict[str, float | int]) -> None:
    """Validate long_connection dict parameters.

    Only validates keys that are present; absent keys are left to defaults
    at runtime.  Future long-connection parameters can be added without
    changing this signature.
    """
    brd = params.get("base_retry_delay", 1.0)
    mrd = params.get("max_retry_delay", 30.0)
    mra = params.get("max_retry_attempts", 5)
    rbf = params.get("retry_backoff_factor", 2.0)
    if "base_retry_delay" in params and brd <= 0:
        raise ConfigError(f"base_retry_delay must be > 0, got {brd}")
    if "max_retry_delay" in params and mrd < brd:
        raise ConfigError(
            f"max_retry_delay must be >= base_retry_delay, got max_retry_delay={mrd} < base_retry_delay={brd}"
        )
    if "max_retry_attempts" in params and mra < 1:
        raise ConfigError(f"max_retry_attempts must be >= 1, got {mra}")
    if "retry_backoff_factor" in params and rbf <= 0:
        raise ConfigError(f"retry_backoff_factor must be > 0, got {rbf}")


@dataclass(repr=False)
class CollectorConfig:
    """Config for the collectors module: connection-type tuning parameters.

    Tuning parameters are grouped by connection type as plain dicts:
    - ``http_polling``: HTTP polling-related parameters (e.g. polling_interval,
      http_timeout).  More HTTP parameters may be added in the future.
    - ``long_connection``: Long-connection-related parameters (e.g. base_retry_delay,
      max_retry_delay, max_retry_attempts, retry_backoff_factor).  More
      long-connection parameters may be added in the future.

    Attributes:
        http_polling: HTTP polling tuning parameters dict.
        long_connection: Long-connection tuning parameters dict.
    """

    http_polling: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_HTTP_POLLING))
    long_connection: dict[str, float | int] = field(default_factory=lambda: dict(_DEFAULT_LONG_CONNECTION))

    def __post_init__(self) -> None:
        _validate_http_polling(self.http_polling)
        _validate_long_connection(self.long_connection)

    __repr__ = _multiline_repr
