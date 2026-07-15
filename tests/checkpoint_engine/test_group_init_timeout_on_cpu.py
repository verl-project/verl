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
"""CPU unit tests for the checkpoint-engine first-init timeout guardrail (#6967).

These exercise the timeout mechanism only (no NCCL/GPU): the guardrail must turn
a hung first rendezvous into a fast, clear error, must not interfere with a
normal init, and must faithfully surface an init error.
"""

import time

import pytest

from verl.checkpoint_engine._group_init import (
    CheckpointEngineInitError,
    run_group_init_with_timeout,
)


def test_completes_within_timeout():
    calls = []
    run_group_init_with_timeout(lambda: calls.append(1), group_name="g", timeout_s=5.0)
    assert calls == [1]


def test_raises_fast_on_timeout_instead_of_hanging():
    # The #6967 symptom: init blocks far longer than the bound. The guardrail
    # must raise promptly (~timeout), not wait for the (here 10s) hang to end.
    start = time.monotonic()
    with pytest.raises(CheckpointEngineInitError):
        run_group_init_with_timeout(lambda: time.sleep(10.0), group_name="g", timeout_s=0.3)
    assert time.monotonic() - start < 3.0


def test_reraises_init_error():
    def boom():
        raise ValueError("nccl boom")

    with pytest.raises(ValueError, match="nccl boom"):
        run_group_init_with_timeout(boom, group_name="g", timeout_s=5.0)


def test_env_var_sets_default_timeout(monkeypatch):
    monkeypatch.setenv("VERL_CKPT_ENGINE_INIT_TIMEOUT_S", "0.2")
    start = time.monotonic()
    with pytest.raises(CheckpointEngineInitError):
        run_group_init_with_timeout(lambda: time.sleep(10.0), group_name="g")
    assert time.monotonic() - start < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
