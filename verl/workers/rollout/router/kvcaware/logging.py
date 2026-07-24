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

"""Centralised loguru logging for the kvcaware router.

Router components run inside a Ray actor process with no pre-configured root
handler — INFO messages would be swallowed. This module ensures loguru has a
stdout sink (so routing decisions reach Ray's captured log stream) and exposes
``get_router_logger()`` for per-component bound loggers.
"""

from __future__ import annotations

import sys

import loguru
from loguru import logger

# ── Ensure a stdout sink exists for the Ray actor process ────────────────
# Equivalent to the old per-file ``if not logger.handlers: … StreamHandler
# + propagate=False`` block, but done once centrally.
_stream_sink_id: int | None = None


def _ensure_stdout_sink() -> None:
    """Add a loguru stdout sink if none exists yet.

    Called at module import so that any ``get_router_logger()`` call
    immediately produces visible output, even inside a bare Ray actor
    process where loguru's default handler was removed by
    ``async_logging.py``.
    """
    global _stream_sink_id
    if _stream_sink_id is not None:
        return
    # Check whether any StreamSink (stdout/stderr) handler already exists
    # from async_logging's DEBUG_MODE setup — don't add a duplicate.
    for h in logger._core.handlers.values():
        sink = h._sink
        if hasattr(sink, "_stream") and sink._stream in (sys.stdout, sys.stderr):
            return
    _stream_sink_id = logger.add(
        sys.stdout,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {extra[name]: <20} | {level: <8} | {message}",
    )


_ensure_stdout_sink()


# ── Per-component logger factory ─────────────────────────────────────────


def get_router_logger(name: str) -> loguru.Logger:
    """Return a loguru bound logger for an llm_router component.

    Unlike the full ``async_logging.get_logger(name, run_id)`` which binds
    a ``run_id``, llm_router components operate inside a Ray actor without
    per-run context. We bind only ``name`` for readable log output.

    Args:
        name: Human-readable component label (e.g. ``"balancer"``).
              Appears in the ``{extra[name]}`` column of the format string.

    Returns:
        A loguru ``BoundLogger`` that can be used as ``logger.info(…)`` etc.
    """
    return logger.bind(name=name)
