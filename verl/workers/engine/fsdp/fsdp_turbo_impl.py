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

import copy
import logging
import math
from contextlib import contextmanager

import torch

from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

from ..base import EngineRegistry
from .transformer_impl import FSDPEngineWithLMHead

logger = logging.getLogger(__file__)


@EngineRegistry.register(model_type="language_model", backend="fsdp_turbo", device=["cuda", "npu"])
class FSDPTurboEngineWithLMHead(FSDPEngineWithLMHead):
    def _init_device_mesh(self):
        super()._init_device_mesh()
        self._init_parallel_state()

    def _init_parallel_state(self):
        from fsdp_turbo.distributed.parallel_state import get_parallel_state, init_parallel_state
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass

        turbo_config = copy.deepcopy(self.engine_config.turbo_config)
        fsdp_plan = turbo_config.setdefault("distributed", {}).setdefault("fsdp_plan", {})
        cpu_offload = bool(self.engine_config.offload_policy or self.engine_config.forward_only)
        fsdp_plan["cpu_offload"] = cpu_offload
        if cpu_offload and get_device_name() == "npu":
            fsdp_plan["pin_memory"] = False
        reshard_after_forward = self.engine_config.reshard_after_forward
        fsdp_plan["reshard_after_forward"] = True if reshard_after_forward is None else reshard_after_forward
        self.fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, turbo_config)
        attn_implementation = getattr(self.fsdp_turbo_config.model, "attn_implementation", "eager")
        for config in (
            self.model_config.hf_config,
            getattr(self.model_config.hf_config, "text_config", None),
        ):
            if config is not None:
                config._attn_implementation = attn_implementation
        vision_config = getattr(self.model_config.hf_config, "vision_config", None)
        if vision_config is not None:
            vision_config._attn_implementation = attn_implementation
        init_parallel_state(self.fsdp_turbo_config)
        self._parallel_state = get_parallel_state()
        if self._is_ulysses_enabled():
            self._process_ulysses_config()

    def _build_module(self, load_pretrained=None, force_meta=False):
        # Do not let verl's Qwen VLM monkey patch slice the text model before
        # FSDP-Turbo's post-fusion model patch does the CP split.
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return super()._build_module(load_pretrained=load_pretrained, force_meta=force_meta)
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        load_after_shard = getattr(self.model_config.hf_config, "load_checkpoint_after_shard", False)
        full_state = None if load_after_shard else module.state_dict()
        module = FSDPTurbo(self.fsdp_turbo_config, module).model
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True
        if full_state is not None:
            fsdp2_load_full_state_dict(module, full_state, None, offload_policy)

        return module

    def _is_ulysses_enabled(self):
        return self._parallel_state.is_group_enable("ulysses")

    def _process_ulysses_config(self):
        if self.ulysses_sequence_parallel_size > 1:
            raise ValueError(
                "Do not enable both FSDP-Turbo CP and verl Ulysses SP. "
                "Use fsdp_kwargs.distributed.ulysses_parallel_size for Turbo CP "
                "and set ulysses_sequence_parallel_size=1."
            )

        self.model_config.hf_config._attn_implementation = "eager"
        self.ulysses_sequence_parallel_size = self._parallel_state.get_ulysses_group_size()
        self.ulysses_parallel_group = self._parallel_state.get_ulysses_group()
        self.use_ulysses_sp = True

    def get_data_parallel_rank(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_rank()
        return self._parallel_state.get_data_rank()

    def get_data_parallel_size(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_size()
        return self._parallel_state.get_data_group_size()

    def get_data_parallel_group(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_group()
        return self._parallel_state.get_data_group()

    def is_mp_src_rank_with_outputs(self):
        if not hasattr(self, "_parallel_state"):
            return super().is_mp_src_rank_with_outputs()
        is_collect = self._parallel_state.get_tp_rank() == 0
        if self._is_ulysses_enabled():
            is_collect = is_collect and self._parallel_state.get_ulysses_rank() == 0
        return is_collect

    @contextmanager
    def _gradient_sync_context(self, *, is_last_micro_batch: bool):
        # FSDP-Turbo owns gradient synchronization.
        yield

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__.
        scaler = getattr(self, "scaler", None)

        # Unscale gradients before clip so the clip threshold is applied to true gradient
        # magnitudes, not scaled ones. scaler.step() will skip the update if any grad is inf/nan.
        if scaler is not None:
            scaler.unscale_(self.optimizer)

        from fsdp_turbo.training.clip_grads import clip_grad_norm

        grad_norm = clip_grad_norm(model=self.module, max_norm=self.optimizer_config.clip_grad)
        if not math.isfinite(grad_norm):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            marker = f"FSDP-Turbo NONFINITE GRADIENT: rank={rank} grad_norm={grad_norm} optimizer_step=False"
            logger.error(marker)
            self.optimizer.zero_grad()
            raise FloatingPointError(marker)
        elif scaler is not None:
            # scaler handles inf/nan skipping internally via _check_inf_per_device.
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        return grad_norm
