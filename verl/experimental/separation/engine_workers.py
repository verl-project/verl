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
from typing import Callable, Optional

from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
)
from verl.workers.config import HFModelConfig, MtpConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, DistillationConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker"]


class DetachActorWorker(ActorRolloutRefWorker):
    """
    A worker class that extends ActorRolloutRefWorker to support detaching and restoring the actor model.

    This worker facilitates saving the model state to CPU and restoring it, enabling efficient
    resource management and checkpointing in distributed training. It currently supports
    FSDP, FSDP2, VeOmni, and Megatron strategies.
    """

    FUSED_TEACHER_STATE = 1
    FUSED_OLD_STUDENT_STATE = 2
    FUSED_CURRENT_STUDENT_STATE = 3

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        """
        Initialize the DetachActorWorker.

        Args:
            config: Configuration dictionary.
            role: The role of the worker (e.g., 'actor', 'rollout', 'ref').
            distillation_config: Optional distillation configuration for OPD support.
            **kwargs: Additional arguments passed to ActorRolloutRefWorker.
        """
        ActorRolloutRefWorker.__init__(self, config, role, distillation_config=distillation_config, **kwargs)
        self._strategy_handlers = None
        self._fused_teacher_enabled = bool(
            self.distillation_enabled
            and distillation_config is not None
            and distillation_config.get("teacher_execution", "rollout") == "trainer"
        )
        self._active_model = "actor"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._fused_teacher_enabled:
            self._init_fused_teacher()

    def _init_fused_teacher(self):
        if self.config.actor.strategy != "megatron":
            raise NotImplementedError("Trainer-colocated teachers currently require actor.strategy=megatron.")
        if self.role not in {"actor", "actor_rollout", "actor_rollout_ref"}:
            raise ValueError(f"Trainer-colocated teacher requires an actor role, got {self.role!r}.")

        distillation_config = omega_conf_to_dataclass(self.distillation_config)
        if len(distillation_config.teacher_models) != 1:
            raise NotImplementedError("Trainer-colocated distillation currently supports exactly one teacher.")
        teacher_config = next(iter(distillation_config.teacher_models.values()))

        teacher_model_dict = {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": teacher_config.model_path,
            "hf_config_path": teacher_config.model_path,
            # Token-level OPD requires a shared token-id space. Load the
            # student's tokenizer while loading the teacher architecture.
            "tokenizer_path": self.config.model.get("tokenizer_path") or self.config.model.path,
            "load_tokenizer": True,
            "use_shm": self.config.model.get("use_shm", False),
            "trust_remote_code": self.config.model.get("trust_remote_code", False),
            "external_lib": self.config.model.get("external_lib", None),
            "enable_gradient_checkpointing": False,
            "use_remove_padding": self.config.model.get("use_remove_padding", True),
            "mtp": OmegaConf.to_container(
                OmegaConf.structured(MtpConfig(enable=False, enable_train=False, enable_rollout=False)),
                resolve=True,
            ),
        }
        teacher_model_config: HFModelConfig = omega_conf_to_dataclass(
            teacher_model_dict, dataclass_type=HFModelConfig
        )

        student_model_config = self.actor.model_config
        if student_model_config.mtp.enable:
            raise NotImplementedError("Single-engine fused OPD currently requires actor MTP to be disabled.")
        if student_model_config.architectures != teacher_model_config.architectures:
            raise ValueError(
                "Single-engine fused OPD requires identical model architectures, "
                f"got student={student_model_config.architectures} and "
                f"teacher={teacher_model_config.architectures}."
            )

        structural_fields = (
            "model_type",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
        )
        student_hf_config = getattr(student_model_config.hf_config, "text_config", student_model_config.hf_config)
        teacher_hf_config = getattr(teacher_model_config.hf_config, "text_config", teacher_model_config.hf_config)
        mismatches = {
            field: (getattr(student_hf_config, field, None), getattr(teacher_hf_config, field, None))
            for field in structural_fields
            if getattr(student_hf_config, field, None) != getattr(teacher_hf_config, field, None)
        }
        if mismatches:
            raise ValueError(f"Single-engine fused OPD requires isomorphic teacher/student models: {mismatches}")

        # Build the static teacher snapshot in the initialized student module.
        # Optimizer/main-parameter state remains untouched and resident.
        self.actor.to(device="device", model=True, optimizer=False, grad=False)
        self.save_model_to_cpu(self.FUSED_CURRENT_STUDENT_STATE)
        try:
            engine = self.actor.engine
            if engine.vanilla_bridge:
                engine.bridge.load_weights(engine.module, teacher_model_config.local_path)
            else:
                engine.bridge.load_hf_weights(engine.module, teacher_model_config.local_path)
            self.save_model_to_cpu(self.FUSED_TEACHER_STATE)
        finally:
            self.restore_model_from_cpu(self.FUSED_CURRENT_STUDENT_STATE)
            self.clear_cpu_model(self.FUSED_CURRENT_STUDENT_STATE)
        self._active_model = "actor"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def activate_teacher(self):
        if not self._fused_teacher_enabled:
            return
        if self._active_model == "teacher":
            return
        self.save_model_to_cpu(self.FUSED_CURRENT_STUDENT_STATE)
        self.restore_model_from_cpu(self.FUSED_TEACHER_STATE)
        self._active_model = "teacher"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def activate_actor(self):
        if not self._fused_teacher_enabled:
            return
        if self._active_model == "actor":
            return
        self.restore_model_from_cpu(self.FUSED_CURRENT_STUDENT_STATE)
        self.clear_cpu_model(self.FUSED_CURRENT_STUDENT_STATE)
        self._active_model = "actor"

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_teacher_log_prob(self, data):
        if not self._fused_teacher_enabled:
            raise RuntimeError("Trainer-colocated teacher is not initialized.")
        if self._active_model != "teacher":
            raise RuntimeError(f"Teacher scoring requires active_model='teacher', got {self._active_model!r}.")
        tu.assign_non_tensor(data, disable_auto_offload=True, calculate_entropy=False, compute_loss=False)
        output = self.actor.infer_batch(data)
        return output.cpu() if output is not None else None

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

        # NOTE: VeOmni internally uses FSDP2 for data parallelism (VeOmniEngine inherits from
        # FSDPEngine and sets data_parallel_mode="fsdp2"), so its model parameters are DTensors
        # that are compatible with FSDP2's sharded save/load utilities.
        #
        # CAVEAT: When VeOmni's param_offload=True, parameters may reside on CPU at the time of
        # save/restore. The current fsdp2_sharded_save_to_cpu / fsdp2_sharded_load_from_cpu
        # assume parameters are on GPU. Callers should ensure the model is loaded back to GPU
        # before calling save_model_to_cpu / restore_model_from_cpu in offload scenarios.
        if strategy in ["fsdp", "fsdp2", "veomni"]:
            from verl.utils.fsdp_utils import (
                fsdp2_sharded_load_from_cpu,
                fsdp2_sharded_save_to_cpu,
            )

            self._strategy_handlers = (fsdp2_sharded_save_to_cpu, fsdp2_sharded_load_from_cpu)
        elif strategy == "megatron":
            from verl.utils.megatron_utils import (
                copy_megatron_model_to_cpu,
                restore_megatron_model_from_cpu,
            )

            self._strategy_handlers = (copy_megatron_model_to_cpu, restore_megatron_model_from_cpu)
        else:
            raise NotImplementedError(f"Unsupported strategy: {strategy}")

        return self._strategy_handlers

    @property
    def copy_handler(self) -> Callable:
        """Get the copy handler for the strategy."""
        return self._get_strategy_handlers()[0]

    @property
    def restore_handler(self) -> Callable:
        """Get the restore handler for the strategy."""
        return self._get_strategy_handlers()[1]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        """
        Save the current model state to CPU memory.

        For FSDP/FSDP2/VeOmni strategies, this uses fsdp2_sharded_save_to_cpu which
        expects model parameters to be on GPU (as DTensors). If VeOmni param_offload
        is enabled, ensure the model has been reloaded to GPU before calling this method.

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

        For FSDP/FSDP2/VeOmni strategies, the saved state is a tuple of
        (cpu_sharded_state, global_spec) produced by fsdp2_sharded_save_to_cpu.
        For Megatron, the saved state is passed directly to the restore handler.

        Args:
            n: Identifier/Key for the saved model state to restore.
        """
        if n in self.cpu_saved_models:
            strategy = self.config.actor.strategy

            if strategy in ["fsdp", "fsdp2", "veomni"]:
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
