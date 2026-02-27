# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Megatron-Core / megatron-bridge monkey patches for QAT workflows.

Patches SwiGLU sharded state-dict, EP gather, extract_sort_key, local-to-global
name mapping, build_conversion_tasks, and parallelism type detection to support
SequentialMLP and quantised wrappers.
"""

import gc
import logging
import re
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)


def apply_swiglu_sharded_factory_patch():
    """Patch ``apply_swiglu_sharded_factory`` to support ``singleton_local_shards``."""
    import megatron.core.transformer.mlp as mlp_module
    from megatron.core.dist_checkpointing import ShardedTensor
    from megatron.core.dist_checkpointing.mapping import (
        ReplicaId,
        ShardedTensorFactory,
    )

    if getattr(mlp_module, "_swiglu_patched", False):
        return
    mlp_module._swiglu_patched = True
    mlp_module._original_apply_swiglu_sharded_factory = mlp_module.apply_swiglu_sharded_factory

    def patched_apply_swiglu_sharded_factory(original_sh_ten, sharded_offsets, singleton_local_shards: bool = False):
        swiglu_shard_axis = 0
        prepend_axis_num = len(sharded_offsets)
        original_shape = original_sh_ten.local_shape
        local_axis_size = original_shape[swiglu_shard_axis]
        assert original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
        rank_offset = original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
        axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
        ):
            if singleton_local_shards:
                offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
                offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
                w_key = f"{key}_w"
                v_key = f"{key}_v"
            else:
                offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
                offset_v = (
                    swiglu_shard_axis + prepend_axis_num,
                    rank_offset + axis_frag,
                    axis_frag * 2,
                )
                w_key = key
                v_key = key

            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    w_key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    v_key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]

        def sh_ten_merge_fn(sub_state_dict):
            with torch.no_grad():
                try:
                    return torch.cat(sub_state_dict)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    logger.warning(
                        "CUDA OOM during tensor merge – falling back to CPU. (Error: %s)",
                        e,
                    )
                    merged = torch.cat([t.cpu() for t in sub_state_dict])
                    gc.collect()
                    torch.cuda.empty_cache()
                    return merged

        return ShardedTensorFactory(
            original_sh_ten.key,
            original_sh_ten.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            original_sh_ten.replica_id,
            flattened_range=original_sh_ten.flattened_range,
        )

    mlp_module.apply_swiglu_sharded_factory = patched_apply_swiglu_sharded_factory
    logger.info("Applied QAT patch: apply_swiglu_sharded_factory now supports singleton_local_shards.")


def revert_swiglu_sharded_factory_patch():
    """Revert :func:`apply_swiglu_sharded_factory_patch`."""
    import megatron.core.transformer.mlp as mlp_module

    if not getattr(mlp_module, "_swiglu_patched", False):
        return
    mlp_module.apply_swiglu_sharded_factory = mlp_module._original_apply_swiglu_sharded_factory
    mlp_module._swiglu_patched = False
    logger.info("Reverted QAT patch: apply_swiglu_sharded_factory.")


def apply_ep_gather_patch():
    """Patch ``gather_from_ep_ranks`` to support SequentialMLP and TEGroupedMLP naming."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    if getattr(MegatronParamMapping, "_ep_gather_patched", False):
        return
    MegatronParamMapping._ep_gather_patched = True
    MegatronParamMapping._original_gather_from_ep_ranks = MegatronParamMapping.gather_from_ep_ranks

    def _patched_gather_from_ep_ranks(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module,  # Optional[MegatronModule]
        hf_param_name: Optional[str],
    ) -> dict[str, torch.Tensor]:
        if megatron_module is None:
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
        else:
            model_config = self._get_config(megatron_module)
            num_experts = model_config.num_moe_experts
            num_experts_per_rank = num_experts // self.ep_size
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(num_experts_per_rank, "num_experts_per_rank")

        local_expert_number = None

        # SequentialMLP pattern: local_experts.<N>
        local_experts_match = re.search(r"local_experts\.(\d+)", self.megatron_param)
        if local_experts_match:
            global_expert_number = int(local_experts_match.group(1))
            local_expert_number = global_expert_number % num_experts_per_rank
        else:
            # TEGroupedMLP pattern: weight<N> or bias<N>
            for key in (".weight", ".bias"):
                if key in self.megatron_param:
                    suffix = self.megatron_param.split(key)[-1]
                    if suffix:  # only if there is actually a number after the suffix
                        global_expert_number = int(suffix)
                        local_expert_number = global_expert_number % num_experts_per_rank
                        break

        if local_expert_number is None:
            raise ValueError(
                f"Cannot extract expert number from: {self.megatron_param}. "
                f"Expected TEGroupedMLP (weight<N>/bias<N>) or SequentialMLP (local_experts.<N>)."
            )

        gathered_expert_param_names = [
            re.sub(
                r"experts\.(\d+)",
                f"experts.{int(local_expert_number) + num_experts_per_rank * i}",
                str(hf_param_name),
            )
            for i in range(self.ep_size)
        ]
        assert str(hf_param_name) in gathered_expert_param_names, (
            f"hf_param_name {hf_param_name} not in {gathered_expert_param_names}"
        )

        gathered_weights = [torch.empty_like(megatron_weights) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_weights, megatron_weights, group=self.ep_group)

        weights_dict: dict[str, torch.Tensor] = {}
        for i, param_name in enumerate(gathered_expert_param_names):
            if param_name in weights_dict:
                weights_dict[param_name] = torch.cat(
                    [weights_dict[param_name], gathered_weights[i].unsqueeze(0)], dim=0
                )
            else:
                weights_dict[param_name] = gathered_weights[i].unsqueeze(0)
        for param_name in weights_dict:
            weights_dict[param_name] = weights_dict[param_name].squeeze()

        return weights_dict

    MegatronParamMapping.gather_from_ep_ranks = _patched_gather_from_ep_ranks
    logger.info("Applied QAT patch: MegatronParamMapping.gather_from_ep_ranks now supports SequentialMLP pattern.")


def revert_ep_gather_patch():
    """Revert :func:`apply_ep_gather_patch`."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    if not getattr(MegatronParamMapping, "_ep_gather_patched", False):
        return
    MegatronParamMapping.gather_from_ep_ranks = MegatronParamMapping._original_gather_from_ep_ranks
    MegatronParamMapping._ep_gather_patched = False
    logger.info("Reverted QAT patch: MegatronParamMapping.gather_from_ep_ranks.")


def apply_extract_sort_key_patch():
    """Patch ``extract_sort_key`` to support SequentialMLP naming pattern."""
    import megatron.bridge.models.conversion.model_bridge as bridge_module
    import megatron.bridge.models.conversion.utils as utils_module

    if getattr(utils_module, "_sort_key_patched", False):
        return
    utils_module._sort_key_patched = True
    bridge_module._sort_key_patched = True
    utils_module._original_extract_sort_key = utils_module.extract_sort_key
    bridge_module._original_extract_sort_key = bridge_module.extract_sort_key

    def _patched_extract_sort_key(param_name: str):
        numbers = []
        layer_match = re.search(r"layers\.(\d+)", param_name)
        if layer_match:
            numbers.append(int(layer_match.group(1)))

        expert_number = None

        # TEGroupedMLP: weight<N>, bias<N>
        expert_match = re.search(r"(?:bias|weight)(\d+)", param_name)
        if expert_match:
            expert_number = int(expert_match.group(1))

        # SequentialMLP: local_experts.<N>
        if expert_number is None:
            local_experts_match = re.search(r"local_experts\.(\d+)", param_name)
            if local_experts_match:
                expert_number = int(local_experts_match.group(1))

        if expert_number is not None:
            numbers.append(expert_number)

        while len(numbers) < 2:
            numbers.append(-1)
        numbers = numbers[:2]
        return numbers, param_name

    utils_module.extract_sort_key = _patched_extract_sort_key
    bridge_module.extract_sort_key = _patched_extract_sort_key
    logger.info("Applied QAT patch: extract_sort_key now supports SequentialMLP pattern.")


def revert_extract_sort_key_patch():
    """Revert :func:`apply_extract_sort_key_patch`."""
    import megatron.bridge.models.conversion.model_bridge as bridge_module
    import megatron.bridge.models.conversion.utils as utils_module

    if not getattr(utils_module, "_sort_key_patched", False):
        return
    utils_module.extract_sort_key = utils_module._original_extract_sort_key
    bridge_module.extract_sort_key = bridge_module._original_extract_sort_key
    utils_module._sort_key_patched = False
    bridge_module._sort_key_patched = False
    logger.info("Reverted QAT patch: extract_sort_key.")


def apply_local_name_to_global_patch():
    """Patch ``_megatron_local_name_to_global`` to support SequentialMLP
    local-to-global expert number conversion under EP."""
    import megatron.bridge.models.conversion.model_bridge as bridge_module
    from megatron.core import parallel_state
    from megatron.core.utils import get_pg_size

    if getattr(bridge_module, "_local_name_to_global_patched", False):
        return
    bridge_module._local_name_to_global_patched = True
    bridge_module._original_megatron_local_name_to_global = bridge_module._megatron_local_name_to_global

    _orig_fn = bridge_module._megatron_local_name_to_global

    def _patched_megatron_local_name_to_global(models, config, param_name, vp_stage=None):
        param_name = _orig_fn(models, config, param_name, vp_stage)

        ep_group = parallel_state.get_expert_model_parallel_group()
        if ".mlp.experts.local_experts." in param_name and get_pg_size(ep_group) > 1 and ".adapter." not in param_name:
            num_experts = config.num_moe_experts
            num_experts_per_rank = num_experts // ep_group.size()
            local_experts_match = re.search(r"\.local_experts\.(\d+)\.", param_name)
            if local_experts_match:
                local_expert_number = int(local_experts_match.group(1))
                global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
                param_name = param_name.replace(
                    f".local_experts.{local_expert_number}.",
                    f".local_experts.{global_expert_number}.",
                )

        return param_name

    bridge_module._megatron_local_name_to_global = _patched_megatron_local_name_to_global
    logger.info("Applied QAT patch: _megatron_local_name_to_global now supports SequentialMLP pattern.")


def revert_local_name_to_global_patch():
    """Revert :func:`apply_local_name_to_global_patch`."""
    import megatron.bridge.models.conversion.model_bridge as bridge_module

    if not getattr(bridge_module, "_local_name_to_global_patched", False):
        return
    bridge_module._megatron_local_name_to_global = bridge_module._original_megatron_local_name_to_global
    bridge_module._local_name_to_global_patched = False
    logger.info("Reverted QAT patch: _megatron_local_name_to_global.")


def apply_build_conversion_tasks_patch():
    """Patch ``build_conversion_tasks`` to filter out ``None`` entries."""
    import itertools

    import megatron.bridge.models.conversion.model_bridge as bridge_module
    from megatron.bridge.models.conversion.model_bridge import (
        MegatronModelBridge,
        WeightConversionTask,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
        persistent_buffers,
    )
    from megatron.bridge.utils.common_utils import print_rank_0
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    if getattr(MegatronModelBridge, "_build_tasks_patched", False):
        return
    MegatronModelBridge._build_tasks_patched = True
    MegatronModelBridge._original_build_conversion_tasks = MegatronModelBridge.build_conversion_tasks

    def _patched_build_conversion_tasks(self, hf_pretrained, megatron_model):
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()

        mapping_registry = self.mapping_registry()
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config, unwrapped_model)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(megatron_model)

        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name for name in sorted_global_param_names_all_pp_ranks if "output_layer" not in name
            ]

        global_names_index_dict = {name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)}

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)
        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_name or self._is_adapter_param_name(local_name):
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = bridge_module._megatron_local_name_to_global(
                    megatron_model, model_config, local_name, vp_stage
                )
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))

                if not mapping:
                    logger.warning(f"WARNING: No mapping found for megatron_param: {global_name}")
                    continue

                if not mapping.allow_hf_name_mismatch:
                    if isinstance(mapping.hf_param, str):
                        if mapping.hf_param not in hf_keys:
                            logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_keys")
                            continue
                    else:
                        missing_params = [hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_keys]
                        if missing_params:
                            logger.warning(
                                f"WARNING: Can't find the following HF parameters in hf_keys: {missing_params}"
                            )
                            continue

                local_module, local_weights = get_module_and_param_from_name(megatron_model, local_name, vp_stage)
                if local_module is not None and not hasattr(local_module, "config"):
                    local_module.config = model_config

                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            if tasks[idx] is None:
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))
                if mapping is None:
                    continue
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    global_param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        tasks = [task for task in tasks if task is not None]
        return tasks

    MegatronModelBridge.build_conversion_tasks = _patched_build_conversion_tasks
    logger.info("Applied QAT patch: MegatronModelBridge.build_conversion_tasks now filters out None entries.")


def revert_build_conversion_tasks_patch():
    """Revert :func:`apply_build_conversion_tasks_patch`."""
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    if not getattr(MegatronModelBridge, "_build_tasks_patched", False):
        return
    MegatronModelBridge.build_conversion_tasks = MegatronModelBridge._original_build_conversion_tasks
    MegatronModelBridge._build_tasks_patched = False
    logger.info("Reverted QAT patch: MegatronModelBridge.build_conversion_tasks.")


def apply_detect_parallelism_type_patch():
    """Patch ``_detect_parallelism_type`` to recognise quantised
    ``LayerNormColumnParallelLinear`` variants via substring matching."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    if getattr(AutoMapping, "_detect_parallelism_patched", False):
        return
    AutoMapping._detect_parallelism_patched = True
    AutoMapping._original_detect_parallelism_type = AutoMapping._detect_parallelism_type

    def _patched_detect_parallelism_type(self, module):
        module_type = type(module).__name__
        if "LayerNormColumnParallelLinear" in module_type:
            if self.megatron_param and (
                self.megatron_param.endswith("layer_norm_weight") or self.megatron_param.endswith("layer_norm_bias")
            ):
                return "replicated"
            return "column"
        return AutoMapping._original_detect_parallelism_type(self, module)

    AutoMapping._detect_parallelism_type = _patched_detect_parallelism_type
    logger.info(
        "Applied QAT patch: AutoMapping._detect_parallelism_type "
        "now supports quantised LayerNormColumnParallelLinear variants."
    )


def revert_detect_parallelism_type_patch():
    """Revert :func:`apply_detect_parallelism_type_patch`."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    if not getattr(AutoMapping, "_detect_parallelism_patched", False):
        return
    AutoMapping._detect_parallelism_type = AutoMapping._original_detect_parallelism_type
    AutoMapping._detect_parallelism_patched = False
    logger.info("Reverted QAT patch: AutoMapping._detect_parallelism_type.")


def apply_qat_patch():
    """Apply all QAT-related patches."""
    apply_swiglu_sharded_factory_patch()
    apply_ep_gather_patch()
    apply_extract_sort_key_patch()
    apply_local_name_to_global_patch()
    apply_build_conversion_tasks_patch()
    apply_detect_parallelism_type_patch()


def revert_qat_patch():
    """Revert all QAT-related patches."""
    revert_swiglu_sharded_factory_patch()
    revert_ep_gather_patch()
    revert_extract_sort_key_patch()
    revert_local_name_to_global_patch()
    revert_build_conversion_tasks_patch()
    revert_detect_parallelism_type_patch()
