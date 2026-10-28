# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Disaggregated senders consume the global stream, independent of rollout EP."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.mark.parametrize("backend", ["hccl", "kimi", "delta_sharded"])
def test_separate_async_does_not_request_colocated_expert_layout(backend):
    stream = iter([("expert.weight", object())])
    engine = SimpleNamespace(get_per_tensor_param=Mock(return_value=(stream, None)), is_param_offload_enabled=False)
    sender = SimpleNamespace(send_weights=AsyncMock(return_value={"version": 2}))
    worker = SimpleNamespace(
        actor=SimpleNamespace(engine=engine),
        checkpoint_engine=sender,
        config=OmegaConf.create({"rollout": {"checkpoint_engine": {"backend": backend}, "expert_parallel_size": 8}}),
    )
    result = asyncio.run(inspect.unwrap(ActorRolloutRefWorker.update_weights)(worker, global_steps=2))
    assert result == {"version": 2}
    if backend == "delta_sharded":
        engine.get_per_tensor_param.assert_not_called()
        sender.send_weights.assert_awaited_once_with(engine, global_steps=2)
    else:
        engine.get_per_tensor_param.assert_called_once_with()
        sender.send_weights.assert_awaited_once_with(stream, global_steps=2)
