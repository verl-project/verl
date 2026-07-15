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
"""Bounded first-rendezvous init for the NCCL checkpoint engine.

The first ``ray.util.collective`` rendezvous (``init_collective_group`` +
``barrier``) can hang indefinitely on some environments -- a timing race in the
Ray/NCCL layer reported in verl issue #6967. When it hangs, both ranks sit at
0% util with no traceback and the whole job is stuck forever.

This wraps that first init in a bounded timeout so a stalled rendezvous fails
fast with a clear, actionable error instead of hanging silently. A stalled
NCCL/collective call cannot be safely interrupted, so we deliberately do NOT
retry in place: we surface the error and let the launcher tear the job down.
"""

import logging
import os
import threading
from typing import Callable

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Generous default: long enough never to trip a slow-but-healthy first sync
# (large models can take seconds to tens of seconds), short enough to abort a
# genuine hang instead of waiting forever. Override via env if needed.
INIT_TIMEOUT_ENV = "VERL_CKPT_ENGINE_INIT_TIMEOUT_S"
DEFAULT_INIT_TIMEOUT_S = 600.0


class CheckpointEngineInitError(RuntimeError):
    """Raised when checkpoint-engine process-group init exceeds its timeout."""


def run_group_init_with_timeout(
    init_fn: Callable[[], None],
    *,
    group_name: str,
    timeout_s: float | None = None,
) -> None:
    """Run the blocking group registration + first barrier under a timeout.

    Args:
        init_fn: Zero-arg callable performing ``init_collective_group`` and the
            first ``barrier`` (and any per-rank setup between them).
        group_name: Collective group name, used only for error messages.
        timeout_s: Seconds to wait. Defaults to the ``VERL_CKPT_ENGINE_INIT_TIMEOUT_S``
            env var, else ``DEFAULT_INIT_TIMEOUT_S``.

    Raises:
        CheckpointEngineInitError: If ``init_fn`` does not finish within the timeout.
        BaseException: Re-raises whatever ``init_fn`` itself raised.
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get(INIT_TIMEOUT_ENV, DEFAULT_INIT_TIMEOUT_S))

    done = threading.Event()
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            init_fn()
        except BaseException as e:  # surfaced to the caller via `error`
            error["exc"] = e
        finally:
            done.set()

    worker = threading.Thread(target=_target, name=f"ckpt-group-init-{group_name}", daemon=True)
    worker.start()
    if not done.wait(timeout_s):
        # The worker thread is still blocked in NCCL/collective and cannot be
        # safely interrupted; we abort the caller and let the launcher tear down.
        raise CheckpointEngineInitError(
            f"checkpoint-engine NCCL group {group_name!r} init did not complete "
            f"within {timeout_s:.0f}s; the first weight sync appears hung "
            f"(verl issue #6967). Aborting instead of hanging indefinitely. If "
            f"this was a slow-but-healthy init, raise {INIT_TIMEOUT_ENV}."
        )
    if "exc" in error:
        raise error["exc"]
