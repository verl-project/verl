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
"""Bounded controller-side wait for checkpoint-engine process-group init.

The first ``ray.util.collective`` rendezvous during checkpoint-engine weight
sync can hang indefinitely on some environments -- a timing race in the
Ray/NCCL layer reported in verl issue #6967. When it hangs, both ranks sit at
0% util with no traceback and the whole job is stuck forever; only the first
sync is affected (the group is reused afterward).

``CheckpointEngineManager.build_process_group()`` dispatches every worker's
``init_process_group`` concurrently and blocks on ``ray.get(...)``. We bound
that controller-side wait so a hung first sync fails fast with a clear,
actionable error instead of hanging forever. A stalled NCCL/collective call in
a worker cannot be safely interrupted, so we deliberately do not retry: we
surface the error and let the launcher tear the job down (Ray reclaims the
actors and GPUs when the driver exits).
"""

import logging
import math
import os
from typing import Any

import ray

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Generous default: long enough never to trip a slow-but-healthy first sync
# (large models can take tens of seconds), short enough to abort a genuine hang
# instead of waiting forever. Override via env if needed.
INIT_TIMEOUT_ENV = "VERL_CKPT_ENGINE_INIT_TIMEOUT_S"
DEFAULT_INIT_TIMEOUT_S = 600.0


class CheckpointEngineInitError(RuntimeError):
    """Raised when checkpoint-engine process-group init exceeds its timeout."""


def _resolve_timeout_s(timeout_s: float | None) -> float:
    """Return the init timeout in seconds, rejecting values that would not bound the wait.

    ``ray.get`` treats ``timeout=-1`` as "wait forever" and cannot convert inf/nan,
    so an unchecked value could silently bring back the hang this guards against.
    """
    source = "timeout_s"
    if timeout_s is None:
        source = INIT_TIMEOUT_ENV
        timeout_s = os.environ.get(INIT_TIMEOUT_ENV, DEFAULT_INIT_TIMEOUT_S)
    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{source} must be a positive, finite number of seconds, got {timeout_s!r}")
    return value


def wait_for_group_init(refs: list, *, timeout_s: float | None = None) -> Any:
    """``ray.get`` the process-group-init object refs under a bounded timeout.

    Args:
        refs: Ray object refs returned by dispatching ``init_process_group`` to
            the actor and rollout worker groups.
        timeout_s: Seconds to wait. Defaults to the ``VERL_CKPT_ENGINE_INIT_TIMEOUT_S``
            env var, else ``DEFAULT_INIT_TIMEOUT_S``.

    Returns:
        The ``ray.get`` results.

    Raises:
        ValueError: If the timeout is not a positive, finite number of seconds.
        CheckpointEngineInitError: If the init does not complete within the timeout.
    """
    timeout_s = _resolve_timeout_s(timeout_s)

    try:
        return ray.get(refs, timeout=timeout_s)
    except ray.exceptions.GetTimeoutError as e:
        raise CheckpointEngineInitError(
            f"checkpoint-engine process-group init did not complete within "
            f"{timeout_s:.0f}s; the first weight sync appears hung (verl issue "
            f"#6967). Aborting instead of hanging indefinitely. If this was a "
            f"slow-but-healthy init, raise {INIT_TIMEOUT_ENV}."
        ) from e
