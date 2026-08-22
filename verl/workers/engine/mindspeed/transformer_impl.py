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

import logging
import os

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name
from verl.workers.config import (
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead
from .utils import (
    reset_fp8_reuse_quantized_weight,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# TODO: Remove this guard after adding NPU-aware staging and validating storage
# release/restore for MindSpeed parameter, gradient-buffer, and optimizer state.
def _reject_disk_offload(engine_config: McoreEngineConfig) -> None:
    disk_components = [
        component
        for component in ("param", "grad", "optimizer")
        if engine_config.get_offload_target(component) == "disk"
    ]
    assert not disk_components, (
        f"MindSpeed/NPU does not support disk offload yet; configured components: {', '.join(disk_components)}"
    )


def _reset_mindspeed_offload_state(engine, device: str, *, model: bool, optimizer: bool, grad: bool) -> None:
    """Preserve MindSpeed's transfer hook around the offload lifecycle API."""

    model = model and engine.is_param_offload_enabled
    optimizer = optimizer and engine.is_optimizer_offload_enabled
    grad = grad and engine.is_grad_offload_enabled
    if model or optimizer or grad:
        reset_fp8_reuse_quantized_weight(engine, device, model, optimizer, grad)


def _mindspeed_repatch(engine_config):
    if repatch is not None:
        from verl.utils.megatron_utils import mapping_string_to_attn_backend

        repatch_config = mapping_string_to_attn_backend(dict(engine_config.get("override_transformer_config", {})))
        # flash-attn-npu batch-invariant replaces DotProductAttention.forward; fusion attention
        # registers the same patch when use_flash_attn=True and causes "the patch of forward exist".
        if repatch_config.get("use_flash_attn_npu_batch_invariant"):
            repatch_config["use_flash_attn"] = False
        else:
            repatch_config.setdefault("use_flash_attn", True)
        if engine_config.context_parallel_size > 1:
            repatch_config["context_parallel_size"] = engine_config.context_parallel_size
        repatch(repatch_config)


@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        _reject_disk_offload(engine_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        reset_fp8_reuse_quantized_weight(self, device, model, optimizer, grad)
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

    def offload(
        self,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
        preserve_grad: bool = True,
    ) -> None:
        _reset_mindspeed_offload_state(self, "cpu", model=model, optimizer=optimizer, grad=grad)
        super().offload(model=model, optimizer=optimizer, grad=grad, preserve_grad=preserve_grad)

    def onload(self, *, model: bool = True, optimizer: bool = True, grad: bool = True, **kwargs) -> None:
        _reset_mindspeed_offload_state(
            self,
            get_device_name(),
            model=model,
            optimizer=optimizer,
            grad=grad,
        )
        super().onload(model=model, optimizer=optimizer, grad=grad, **kwargs)


@EngineRegistry.register(model_type="value_model", backend="megatron", device="npu")
class MindspeedEngineWithValueHead(MegatronEngineWithValueHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        _reject_disk_offload(engine_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()
