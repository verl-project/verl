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

"""``nccl_timeout`` must reach the worker's torch process group.

``actor_rollout_ref.nccl_timeout`` and ``critic.nccl_timeout`` are documented public config keys
for the torch process-group timeout. The FSDP/Megatron workers read them; when those workers were
replaced by the engine workers every reader disappeared, so ``TrainingWorker.__init__`` created the
process group with ``timeout_second=None`` (torch's own default) and any configured value was
silently ignored.

These tests drive the worker constructor with the process-group init mocked out, so no GPU, no ray
cluster and no real distributed init are needed.
"""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch

from verl.utils.distributed import initialize_global_process_group_ray
from verl.workers import engine_workers
from verl.workers.config import TrainingWorkerConfig


class _StopInit(Exception):
    """Aborts ``TrainingWorker.__init__`` right after the process group is initialized."""


def _timeout_second_forwarded_by(config: TrainingWorkerConfig):
    """Run ``TrainingWorker.__init__`` up to the process-group init and return its timeout.

    Everything after that point needs a real model/engine, so the recorder raises to stop the
    constructor as soon as the value under test has been observed. ``Worker.__init__`` is stubbed
    out because it expects the ray-injected worker environment.
    """
    recorded = {}

    def _record(timeout_second=None, backend=None):
        recorded["timeout_second"] = timeout_second
        raise _StopInit

    with (
        patch.object(engine_workers.Worker, "__init__", lambda self: None),
        patch.object(engine_workers, "initialize_global_process_group_ray", _record),
        patch.object(engine_workers, "set_numa_affinity", lambda: None),
    ):
        with pytest.raises(_StopInit):
            engine_workers.TrainingWorker(config=config)

    assert "timeout_second" in recorded, "TrainingWorker never initialized a process group"
    return recorded["timeout_second"]


@pytest.mark.parametrize("nccl_timeout", [600, 1800])
def test_training_worker_forwards_configured_nccl_timeout(nccl_timeout):
    """The value the trainer copies out of ``actor_rollout_ref``/``critic`` must be used verbatim."""
    config = TrainingWorkerConfig(model_type="language_model", nccl_timeout=nccl_timeout)

    assert _timeout_second_forwarded_by(config) == nccl_timeout


def test_training_worker_without_nccl_timeout_keeps_torch_default():
    """Callers that configure nothing (e.g. SFT) must keep torch's own default timeout."""
    config = TrainingWorkerConfig(model_type="language_model")

    assert config.nccl_timeout is None
    assert _timeout_second_forwarded_by(config) is None


@pytest.mark.parametrize(
    ("timeout_second", "expected_timeout"),
    [(600, timedelta(seconds=600)), (1800, timedelta(seconds=1800)), (None, None)],
)
def test_initialize_global_process_group_ray_forwards_timeout(monkeypatch, timeout_second, expected_timeout):
    """The seconds the worker passes down land on ``torch.distributed.init_process_group``."""
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs))

    # An explicit backend keeps this off the accelerator-detection path.
    initialize_global_process_group_ray(timeout_second=timeout_second, backend="gloo")

    assert len(calls) == 1
    assert calls[0]["timeout"] == expected_timeout
