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

import logging
import os
import sys

import torch.distributed

from .utils import (
    add_config,
    get_base_mcore_config_from_engine_config,
    get_base_mcore_config_from_model_config,
    get_base_mcore_config_from_optim_config,
)

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, MindSpeedLLMEngineConfig, MindSpeedLLMOptimizerConfig

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_patch(model_config, engine_config, optimizer_config):
    print("parse args to apply llm adapter")
    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    model_config = get_base_mcore_config_from_model_config(model_config)
    optimizer_config = get_base_mcore_config_from_optim_config(optimizer_config)
    engine_config = get_base_mcore_config_from_engine_config(engine_config)
    add_config(model_config)
    add_config(optimizer_config)
    add_config(engine_config)
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    args = megatron.training.arguments.parse_args(ignore_unknown_args=True)
    sys.argv = origin_sys_argv
    from megatron.training.arguments import validate_args
    from megatron.training.global_vars import set_global_variables

    validate_args(args)
    try:
        set_global_variables(args)
    except:
        print("megatron args already set")


@EngineRegistry.register(model_type="language_model", backend="mindspeedllm", device="npu")
class MindSpeedLLMEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MindSpeedLLMEngineConfig,
        optimizer_config: MindSpeedLLMOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        apply_patch(self.model_config, self.engine_config, self.optimizer_config)

    def _build_megatron_module(self):
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )

        self.is_value_model = is_value_model

        from megatron.core.enums import ModelType
        from megatron.training.training import get_model

        from .utils import gpt_model_provider
        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
            return module

        module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
        if self.vanilla_bridge:
            self.bridge.load_weights(module, self.model_config.local_path)
        else:
            raise ValueError(f"vanilla_bridge should be True, but got {self.vanilla_bridge}")

        if torch.distributed.get_rank() == 0:
            from verl.utils.model import print_model_size
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

