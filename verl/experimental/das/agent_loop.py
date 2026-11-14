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
"""DAS agent-loop worker and manager for the DataProto path (fully-async rollouter).

The TransferQueue (v1) variants live in :mod:`verl.experimental.das.agent_loop_tq`
so this module never imports ``transfer_queue``.
"""

from __future__ import annotations

import logging
import os

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.das.client import push_gate, wrap_llm_client
from verl.experimental.das.service import get_or_create_suffix_service
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncAgentLoopManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_BEGIN_ITERATION_TIMEOUT_S = 30.0


def das_config_from(config) -> DASConfig:
    """``rollout.das`` of the whole trainer config as a ``DASConfig``."""
    das = config.actor_rollout_ref.rollout.das
    if isinstance(das, DASConfig):
        return das
    return omega_conf_to_dataclass(das, DASConfig)


class DASAgentLoopWorker(AgentLoopWorker):
    """``AgentLoopWorker`` whose server client carries DAS problem ids and pushes trajectories."""

    _das_track_weight_version = False

    def __init__(self, config, llm_client, *args, **kwargs):
        super().__init__(config, llm_client, *args, **kwargs)
        self.llm_client = wrap_llm_client(
            self.llm_client, self.rollout_config.das, track_weight_version=self._das_track_weight_version
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        validate = bool(batch.meta_info.get("validate", False))
        with push_gate(not validate):
            return await super().generate_sequences(batch)


class DASFullyAsyncAgentLoopWorker(DASAgentLoopWorker):
    """Fully-async variant: the step boundary comes from the weight version stamped on responses."""

    _das_track_weight_version = True


class DASIterationDriver:
    """Manager-side owner of the store; advances the iteration once per training step."""

    def __init__(self, cfg: DASConfig) -> None:
        self.cfg = cfg
        self.service = get_or_create_suffix_service(cfg)
        self._last_step: int | None = None

    def begin_iteration(self, step, validate: bool = False) -> None:
        """Open iteration ``step`` (skipped for validation batches and already-seen steps)."""
        if validate or step is None:
            return
        step = int(step)
        if self._last_step is not None and step <= self._last_step:
            return
        try:
            info = ray.get(self.service.begin_iteration.remote(step), timeout=_BEGIN_ITERATION_TIMEOUT_S)
            self._last_step = step
            logger.info(
                "DAS: store iteration %s -> problems=%s total_tokens=%s log_version=%s",
                info.get("iteration"),
                info.get("problems"),
                info.get("total_tokens"),
                info.get("version"),
            )
        except Exception as e:
            logger.warning("DAS: begin_iteration(%d) failed (%s); engines keep the previous iteration", step, e)


class DASFullyAsyncAgentLoopManager(FullyAsyncAgentLoopManager):
    """Fully-async rollouter: owns the store; workers signal boundaries from weight versions."""

    def __init__(self, config, llm_client, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DASFullyAsyncAgentLoopWorker)
        super().__init__(config, llm_client, *args, **kwargs)
        self.das = DASIterationDriver(das_config_from(config))
