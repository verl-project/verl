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
from torch.distributed.nn.functional import all_reduce

import verl.utils.tensordict_utils as tu
import verl.utils.torch_functional as verl_F
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import fsdp2_load_full_state_dict
from verl.utils.model_adapters import kimi_k3_logits_to_input_indices
from verl.utils.torch_functional import logprobs_from_logits

from ..base import EngineRegistry
from .transformer_impl import FSDPEngineWithLMHead, _scale_logits_by_temperature

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
            vision_config._attn_implementation = (
                "flash_attention_2" if self.model_config.hf_config.model_type == "kimi_k3" else attn_implementation
            )
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
            if self.model_config.hf_config.model_type == "kimi_k3":
                load_pretrained = False
                force_meta = True
            return super()._build_module(load_pretrained=load_pretrained, force_meta=force_meta)
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        if self.model_config.hf_config.model_type == "kimi_k3":
            # Wrap the meta model first, then materialize only final local shards.
            module = FSDPTurbo(self.fsdp_turbo_config, module).model
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                self._uses_fsdp2_cpu_offload_policy = True

            from .streaming_loader import load_kimi_k3_checkpoint_to_local_shards

            materialize_device = (
                "cpu" if self.engine_config.offload_policy or self.engine_config.forward_only else get_device_id()
            )
            load_kimi_k3_checkpoint_to_local_shards(
                module,
                self.model_config.local_path,
                materialize_device=materialize_device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return module

        full_state = module.state_dict()
        module = FSDPTurbo(self.fsdp_turbo_config, module).model
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True
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
        if self._is_ulysses_enabled():
            is_collect = self._parallel_state.get_ulysses_rank() == 0
        else:
            is_collect = True
        return is_collect

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if self._is_ulysses_enabled() and "routed_experts" in model_inputs:
            from fsdp_turbo.distributed.context_parallel.batch import pad_tensor_to_group, split_tensor_to_group
            from fsdp_turbo.models.kimi.kimi_r3_utils import align_kimi_full_model_routes

            # The CP forward receives routes already aligned to its local fused
            # rows. Preserve image rows and left padding before slicing all three
            # replay tensors on exactly the same axis as the hidden states.
            raw_mask = model_inputs["attention_mask"]
            raw_lengths = raw_mask.sum(-1)
            indices = output_args.get("logits_to_input_indices")
            if indices is None:
                expanded_mask = raw_mask
            else:
                unpadded_indices = kimi_k3_logits_to_input_indices(
                    model_inputs["input_ids"], model_inputs, self.model_config.hf_config, padding_mask=raw_mask
                )
                last_raw = (raw_lengths - 1).unsqueeze(-1)
                ends = indices.gather(1, last_raw) + 1
                lengths = unpadded_indices.gather(1, last_raw) + 1
                positions = torch.arange(int(indices[:, -1].max()) + 1, device=indices.device)[None, :]
                expanded_mask = (positions >= ends - lengths) & (positions < ends)
            keys = ("routed_experts", "routed_expert_weights", "router_replay_mask")
            aligned = align_kimi_full_model_routes(*(model_inputs[key] for key in keys), expanded_mask)
            for key, value in zip(keys, aligned, strict=True):
                model_inputs[key] = split_tensor_to_group(
                    pad_tensor_to_group(value, self.ulysses_parallel_group, dim=1),
                    self.ulysses_parallel_group,
                    dim=1,
                )
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        if (
            not self._is_ulysses_enabled()
            or self.model_config.hf_config.model_type != "kimi_k3"
            or tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True)
        ):
            return super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)

        # Turbo shards the image-expanded sequence. Select only raw-token
        # positions owned by this rank and exchange scalar outputs, not logits.
        logits = output.logits
        input_ids = micro_batch["input_ids"]
        offsets = input_ids.offsets()
        lengths = offsets.diff()
        indices = output_args.get("logits_to_input_indices")
        if indices is None:
            indices = torch.arange(int(lengths.max()), device=logits.device).expand(len(lengths), -1)
        valid = torch.arange(indices.shape[1], device=logits.device)[None, :] < lengths[:, None]
        start = self._parallel_state.get_ulysses_rank() * logits.shape[1]
        owned = valid & (indices >= start) & (indices < start + logits.shape[1])
        batch_indices, token_indices = owned.nonzero(as_tuple=True)
        local_logits = logits[batch_indices, indices[owned] - start]
        local_logits = _scale_logits_by_temperature(
            local_logits,
            output_args["temperature"][batch_indices, None],
            is_unit_temperature=output_args["temperature_is_one"],
        )
        flat_indices = offsets[batch_indices] + token_indices
        labels = output_args["input_ids_rmpad_rolled"][flat_indices]
        values = {
            "log_probs": logprobs_from_logits(
                logits=local_logits, labels=labels, inplace_backward=False, force_fp32=logits_processor_func is None
            )
            if local_logits.shape[0]
            else local_logits.sum(dim=-1)
        }
        if tu.get_non_tensor_data(micro_batch, "calculate_entropy", default=False):
            if self.engine_config.entropy_checkpointing:
                values["entropy"] = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, local_logits)
            else:
                values["entropy"] = verl_F.entropy_from_logits(local_logits)
        if tu.get_non_tensor_data(micro_batch, "calculate_sum_pi_squared", default=False):
            values["sum_pi_squared"] = verl_F.calculate_sum_pi_squared_from_logits(local_logits)

        local_values = torch.stack(list(values.values()), dim=-1)
        dense_values = local_values.new_zeros((*indices.shape, len(values)))
        dense_values = dense_values.index_put((batch_indices, token_indices), local_values)
        # Backward sums the replicated loss gradients across CP; FSDP averages
        # them with the other model replicas, matching the CP=1 normalization.
        dense_values = all_reduce(dense_values, group=self.ulysses_parallel_group)
        return {
            name: torch.nested.nested_tensor_from_jagged(dense_values[..., column][valid], offsets)
            for column, name in enumerate(values)
        }

    @contextmanager
    def _gradient_sync_context(self, *, is_last_micro_batch: bool):
        # FSDP-Turbo owns gradient synchronization at its collective boundary.
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

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales

            invalidate_all_scales(self.module)

        return grad_norm
