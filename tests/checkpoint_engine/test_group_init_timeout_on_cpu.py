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

These exercise ``wait_for_group_init`` -- the controller-side bound that
``CheckpointEngineManager.build_process_group()`` wraps around its
``ray.get`` of the workers' ``init_process_group`` futures. ``ray.get`` is
mocked so the tests run on CPU with no Ray cluster: a hung init must become a
fast, clear ``CheckpointEngineInitError``; a normal init must pass through; a
real init error must propagate unchanged; and the env var must set the bound.
"""

from unittest.mock import patch

import pytest
import ray

from verl.checkpoint_engine._group_init import (
    CheckpointEngineInitError,
    wait_for_group_init,
)


def test_returns_results_when_init_completes():
    with patch("verl.checkpoint_engine._group_init.ray.get", return_value=["ok0", "ok1"]) as mock_get:
        assert wait_for_group_init(["ref0", "ref1"], timeout_s=5.0) == ["ok0", "ok1"]
        mock_get.assert_called_once()


def test_raises_clear_error_on_timeout():
    with patch(
        "verl.checkpoint_engine._group_init.ray.get",
        side_effect=ray.exceptions.GetTimeoutError("timed out"),
    ):
        with pytest.raises(CheckpointEngineInitError, match="#6967"):
            wait_for_group_init(["ref"], timeout_s=0.1)


def test_reraises_non_timeout_errors():
    with patch(
        "verl.checkpoint_engine._group_init.ray.get",
        side_effect=RuntimeError("nccl boom"),
    ):
        with pytest.raises(RuntimeError, match="nccl boom"):
            wait_for_group_init(["ref"], timeout_s=5.0)


def test_env_var_sets_default_timeout(monkeypatch):
    monkeypatch.setenv("VERL_CKPT_ENGINE_INIT_TIMEOUT_S", "12.5")
    captured = {}

    def fake_get(refs, timeout=None):
        captured["timeout"] = timeout
        return []

    with patch("verl.checkpoint_engine._group_init.ray.get", side_effect=fake_get):
        wait_for_group_init(["ref"])
    assert captured["timeout"] == 12.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
