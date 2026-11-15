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
"""DAS agent-loop worker and manager for the v1 TransferQueue trainer (``main_ppo`` default)."""

from __future__ import annotations

import ray
from tensordict import TensorDict

from verl.experimental.das.agent_loop import DASIterationDriver, das_config_from
from verl.experimental.das.client import push_gate, wrap_llm_client
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.tensordict_utils import get_non_tensor_data

# ``AgentLoopWorkerTQ`` is a Ray actor class; subclass the underlying Python class. The subclass
# stays a plain class (the manager wraps it with ``ray.remote``) so Ray pickles it by reference.
_AgentLoopWorkerTQ = AgentLoopWorkerTQ.__ray_actor_class__


class DASAgentLoopWorkerTQ(_AgentLoopWorkerTQ):
    """``AgentLoopWorkerTQ`` whose server client carries DAS problem ids and pushes trajectories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_client = wrap_llm_client(self.llm_client, self.rollout_config.das)

    async def generate_sequences(self, batch: TensorDict) -> None:
        # Read the flag before the stock worker pops it; the ContextVar is inherited by
        # every trajectory task spawned below, so a later validation batch cannot gate
        # the training batch's still-running trajectories.
        validate = bool(get_non_tensor_data(batch, "validate", False))
        with push_gate(not validate):
            return await super().generate_sequences(batch)


class DASAgentLoopManagerTQ(AgentLoopManagerTQ):
    """``AgentLoopManagerTQ`` that owns the DAS store and advances it once per training step."""

    def __init__(self, config, llm_client, *args, **kwargs):
        super().__init__(config, llm_client, *args, **kwargs)
        # Workers are created in ``create()``, after ``__init__``: override the stock class here.
        self.agent_loop_workers_class = ray.remote(DASAgentLoopWorkerTQ)
        self.das = DASIterationDriver(das_config_from(config))

    def generate_sequences(self, prompts: TensorDict) -> None:
        self.das.begin_iteration(
            get_non_tensor_data(prompts, "global_steps", None),
            bool(get_non_tensor_data(prompts, "validate", False)),
        )
        return super().generate_sequences(prompts)
