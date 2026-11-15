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

import asyncio

import pytest

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutErrorSignal,
    first_unexpected_asyncio_exception,
    format_rollout_error_signal,
    wait_for_task_failure_or_completion,
)


def test_rollout_error_signal_from_exception_preserves_error_details():
    try:
        raise RuntimeError("injected rollout failure")
    except RuntimeError as exc:
        signal = RolloutErrorSignal.from_exception(exc)

    assert signal.error_type == "RuntimeError"
    assert signal.message == "injected rollout failure"
    assert "RuntimeError: injected rollout failure" in signal.traceback


def test_rollout_error_signal_formats_error_details():
    signal = RolloutErrorSignal(
        error_type="RuntimeError",
        message="injected rollout failure",
        traceback="Traceback details",
    )

    message = format_rollout_error_signal(signal)

    assert message == "Rollout generation failed: RuntimeError: injected rollout failure\nTraceback details"
    assert format_rollout_error_signal(object()) is None


def test_asyncio_exception_filter_ignores_cancellations():
    cancellation = asyncio.CancelledError()
    rollout_error = RuntimeError("rollout failed")

    assert first_unexpected_asyncio_exception([cancellation]) is None
    assert first_unexpected_asyncio_exception([None, cancellation, rollout_error]) is rollout_error


@pytest.mark.asyncio
async def test_wait_for_task_failure_returns_before_pending_task_completes():
    gate = asyncio.Event()

    async def fail_generation():
        await asyncio.sleep(0)
        raise RuntimeError("rollout failed")

    async def monitor_loop():
        await gate.wait()

    generation_task = asyncio.create_task(fail_generation())
    monitor_task = asyncio.create_task(monitor_loop())

    try:
        error = await asyncio.wait_for(
            wait_for_task_failure_or_completion([generation_task, monitor_task]), timeout=1.0
        )

        assert isinstance(error, RuntimeError)
        assert str(error) == "rollout failed"
        assert not monitor_task.done()
    finally:
        monitor_task.cancel()
        await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
