# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import os

from omegaconf import DictConfig

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
)
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker"]


class DetachActorWorker(ActorRolloutRefWorker):
    """
    A worker class that extends ActorRolloutRefWorker to support detaching and restoring the actor model.

    This worker facilitates saving the model state to CPU and restoring it, enabling efficient
    resource management and checkpointing in distributed training. It currently supports
    FSDP, FSDP2, and Megatron strategies.
    """

    def __init__(self, config: DictConfig, role: str):
        """
        Initialize the DetachActorWorker.

        Args:
            config: Configuration dictionary.
            role: The role of the worker (e.g., 'actor', 'rollout', 'ref').
        """
        ActorRolloutRefWorker.__init__(self, config, role)
        self._strategy_handlers = None
        self.copy_handler, self.restore_handler = self._get_strategy_handlers()

    def _get_strategy_handlers(self):
        """
        Get the strategy-specific handlers for saving and restoring the model.

        Returns:
            tuple: A tuple containing (save_handler, restore_handler).

        Raises:
            NotImplementedError: If the strategy is not supported.
        """
        if self._strategy_handlers is not None:
            return self._strategy_handlers

        strategy = self.config.actor.strategy

        if strategy in ["fsdp", "fsdp2"]:
            from verl.experimental.fully_async_policy.fsdp2_utils import (
                fsdp2_sharded_load_from_cpu,
                fsdp2_sharded_save_to_cpu,
            )

            self._strategy_handlers = (fsdp2_sharded_save_to_cpu, fsdp2_sharded_load_from_cpu)
        elif strategy == "megatron":
            from verl.experimental.fully_async_policy.megatron_utils import (
                copy_megatron_model_to_cpu,
                restore_megatron_model_from_cpu,
            )

            self._strategy_handlers = (copy_megatron_model_to_cpu, restore_megatron_model_from_cpu)
        else:
            raise NotImplementedError(f"Unsupported strategy: {strategy}")

        return self._strategy_handlers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        """
        Save the current model state to CPU memory.

        Args:
            n: Identifier/Key for the saved model state.
        """
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}

        self.cpu_saved_models[n] = self.copy_handler(self.actor.engine.module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        """
        Restore the model state from CPU memory.

        Args:
            n: Identifier/Key for the saved model state to restore.
        """
        if n in self.cpu_saved_models:
            strategy = self.config.actor.strategy

            if strategy in ["fsdp", "fsdp2"]:
                cpu_sharded_state, global_spec = self.cpu_saved_models[n]
                self.restore_handler(self.actor.engine.module, cpu_sharded_state, global_spec)
            else:
                self.restore_handler(self.actor.engine.module, self.cpu_saved_models[n])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        """
        Clear the saved model state from CPU memory.

        Args:
            n: Identifier/Key for the saved model state to remove.
        """
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]
