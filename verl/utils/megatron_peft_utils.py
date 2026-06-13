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
"""Utilities for PEFT (Parameter-Efficient Fine-Tuning) of Megatron in VERL."""

import os
import re
from typing import Iterator

import torch

# Map megatron lora target modules to HF-style module names for vLLM
MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "router": ["gate"],
    # Canonical LoRA mappings
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_up": ["up_proj"],
    "linear_fc1_gate": ["gate_proj"],
    # MLA mappings
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_q_proj": ["q_proj"],
    # DSA indexer mappings
    "linear_wq_b": ["wq_b"],
    "linear_wk": ["wk"],
    "linear_weights_proj": ["weights_proj"],
}

# Modules with stacked parameters that need .base_layer suffix in vLLM
STACKED_PARAMS = [
    ".q_proj.weight",
    ".q_proj.bias",
    ".k_proj.weight",
    ".k_proj.bias",
    ".v_proj.weight",
    ".v_proj.bias",
    ".o_proj.weight",
    ".o_proj.bias",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".mlp.gate.weight",
    ".mlp.gate.bias",
    ".mlp.gate.e_score_correction_bias",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".wq_b.weight",
    ".wk.weight",
    ".weights_proj.weight",
]

_MOE_EXPERT_LORA_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts\.)(?P<expert_id>\d+)"
    r"(?P<suffix>\.(?:gate_proj|up_proj|down_proj)\.lora_[AB]\.weight)$"
)
_MOE_EXPERT_LORA_3D_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts\.)(?P<expert_id>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.lora_(?P<side>[AB])\.weight$"
)
_QWEN3_OMNI_3D_MOE_MODEL_TYPES = {"qwen3_omni_moe"}


def count_adapter_parameters(model):
    """Count the number of trainable adapter parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (adapter_params, total_params, percentage)
    """
    from verl.utils.megatron_utils import unwrap_model

    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0]

    adapter_params = 0
    total_params = 0

    for name, param in unwrapped.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower() or "adapter" in name.lower():
            if param.requires_grad:
                adapter_params += param.numel()

    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    return adapter_params, total_params, percentage


def print_adapter_info(model):
    """Print information about adapter parameters in the model."""
    adapter_params, total_params, percentage = count_adapter_parameters(model)

    print(f"\n{'=' * 60}")
    print("PEFT Adapter Information:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Adapter parameters:   {adapter_params:,}")
    print(f"  Trainable percentage: {percentage:.2f}%")
    print(f"{'=' * 60}\n")


def convert_megatron_to_hf_target_modules(megatron_modules: list[str]) -> list[str]:
    """Convert megatron lora target modules to HF-style module names.

    Args:
        megatron_modules: List of megatron-style module names.

    Returns:
        List of HF-style module names with duplicates removed.
    """
    hf_target_modules = []
    for module in megatron_modules:
        if module in MEGATRON_TO_HF_MODULES:
            hf_target_modules.extend(MEGATRON_TO_HF_MODULES[module])
        else:
            hf_target_modules.append(module)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(hf_target_modules))


def build_peft_config_for_vllm(lora_config: dict) -> dict:
    """Build a peft_config dict compatible with vLLM's PEFTHelper from megatron lora config.

    Args:
        lora_config: Megatron lora configuration dictionary.

    Returns:
        A dictionary compatible with vLLM's PEFTHelper.from_dict().
    """
    from peft import TaskType

    target_modules = lora_config.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])
    exclude_modules = lora_config.get("exclude_modules", [])
    hf_target_modules = convert_megatron_to_hf_target_modules(target_modules)
    hf_exclude_modules = convert_megatron_to_hf_target_modules(exclude_modules)

    return {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_config.get("rank", 0),
        "lora_alpha": lora_config.get("alpha", 32),
        "target_modules": hf_target_modules,
        "exclude_modules": hf_exclude_modules,
        "bias": "none",
        "lora_dropout": lora_config.get("dropout", 0.0),
    }


# vLLM needs to target all-linear no matter about specific LoRA config
def add_base_layer_suffix(
    params: Iterator[tuple[str, torch.Tensor]],
    model_type: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield param pairs with a base-layer suffix added to the param name.

    Args:
        params: Iterator of (param_name, tensor)
        model_type: The type of the model (e.g., "llama").
    """
    stacked_params = STACKED_PARAMS
    # TODO: other models may have more special treatment, or integrate this into Megatron-Bridge
    if model_type == "llama":
        stacked_params = [".embed_tokens.weight", *STACKED_PARAMS]
    for name, param in params:
        ending_suffix = ""
        for suffix in stacked_params:
            if name.endswith(suffix):
                ending_suffix = suffix
                break
        if ending_suffix:
            suffix = ending_suffix.rsplit(".", 1)[-1]
            name = f"{name[: -len(suffix)]}base_layer.{suffix}"
        yield name, param


def _get_expert_parallel_info() -> tuple[int, int, object | None]:
    try:
        from megatron.core import parallel_state
    except Exception:
        return 1, 0, None

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1, 0, None

    try:
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        ep_group = parallel_state.get_expert_model_parallel_group()
    except Exception:
        return 1, 0, None

    if ep_size is None or ep_size <= 1 or ep_group is None:
        return 1, 0, None
    return ep_size, ep_rank, ep_group


def _all_gather_tensor_for_ep(tensor: torch.Tensor, ep_size: int, ep_group) -> list[torch.Tensor]:
    src = tensor.detach().contiguous()
    original_device = src.device
    gather_src = src
    if not gather_src.is_cuda and torch.cuda.is_available():
        gather_src = gather_src.to(torch.cuda.current_device(), non_blocking=True)

    gathered = [torch.empty_like(gather_src) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, gather_src, group=ep_group)

    if original_device.type == "cpu":
        gathered = [item.cpu() for item in gathered]
    return gathered


def gather_ep_lora_adapter_weights_for_vllm(
    params: Iterator[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Gather EP-local MoE LoRA tensors and rename them to global expert ids."""
    ep_size, _ep_rank, ep_group = _get_expert_parallel_info()
    materialized = list(params)
    if ep_size <= 1:
        yield from materialized
        return

    expert_ids = []
    for name, _param in materialized:
        match = _MOE_EXPERT_LORA_RE.match(name)
        if match:
            expert_ids.append(int(match.group("expert_id")))

    if not expert_ids:
        yield from materialized
        return

    enabled = os.getenv("VERL_MEGATRON_LORA_GATHER_EP_ADAPTER_WEIGHTS", "1").lower()
    if enabled in {"0", "false", "no", "off"}:
        yield from materialized
        return

    local_expert_count = max(expert_ids) + 1
    debug = os.getenv("VERL_MEGATRON_LORA_GATHER_EP_ADAPTER_DEBUG", "0").lower() in {"1", "true", "yes"}
    gathered_expert_tensors = 0
    passthrough_tensors = 0

    for name, param in materialized:
        match = _MOE_EXPERT_LORA_RE.match(name)
        if match is None:
            passthrough_tensors += 1
            yield name, param
            continue

        local_expert_id = int(match.group("expert_id"))
        gathered = _all_gather_tensor_for_ep(param, ep_size, ep_group)
        for gathered_ep_rank, gathered_param in enumerate(gathered):
            global_expert_id = gathered_ep_rank * local_expert_count + local_expert_id
            yield f"{match.group('prefix')}{global_expert_id}{match.group('suffix')}", gathered_param
            gathered_expert_tensors += 1

    if debug:
        print(
            "[verl.megatron_lora] gathered EP adapter tensors for vLLM: "
            f"ep_size={ep_size} local_experts={local_expert_count} "
            f"expert_tensors_out={gathered_expert_tensors} passthrough={passthrough_tensors}",
            flush=True,
        )


def pack_3d_moe_lora_adapter_weights_for_vllm(
    params: Iterator[tuple[str, torch.Tensor]],
    model_type: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pack Qwen3-Omni per-expert LoRA tensors into vLLM's 3D MoE layout."""
    materialized = list(params)
    enabled = os.getenv("VERL_MEGATRON_LORA_PACK_3D_MOE_ADAPTER_WEIGHTS", "1").lower()
    if model_type not in _QWEN3_OMNI_3D_MOE_MODEL_TYPES or enabled in {"0", "false", "no", "off"}:
        yield from materialized
        return

    grouped: dict[str, dict[int, dict[tuple[str, str], tuple[int, torch.Tensor]]]] = {}
    for index, (name, tensor) in enumerate(materialized):
        match = _MOE_EXPERT_LORA_3D_RE.match(name)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {}).setdefault(int(match.group("expert_id")), {})[
            (match.group("proj"), match.group("side"))
        ] = (index, tensor)

    required = {
        ("gate_proj", "A"),
        ("gate_proj", "B"),
        ("up_proj", "A"),
        ("up_proj", "B"),
        ("down_proj", "A"),
        ("down_proj", "B"),
    }
    replacements: dict[int, list[tuple[str, torch.Tensor]]] = {}
    consumed_indices: set[int] = set()
    packed_groups = 0
    packed_tensors = 0

    for prefix, experts in grouped.items():
        if not experts or any(set(tensors) != required for tensors in experts.values()):
            continue

        expert_ids = sorted(experts)
        first_index = min(index for tensors in experts.values() for index, _tensor in tensors.values())
        gate_up_a = []
        gate_up_b = []
        down_a = []
        down_b = []
        for expert_id in expert_ids:
            tensors = experts[expert_id]
            gate_a = tensors[("gate_proj", "A")][1]
            up_a = tensors[("up_proj", "A")][1]
            if gate_a.shape != up_a.shape or not torch.equal(gate_a, up_a):
                raise ValueError(
                    "Cannot pack Qwen3-Omni 3D MoE LoRA tensors for vLLM because "
                    f"expert {expert_id} has different gate_proj and up_proj lora_A weights."
                )
            gate_up_a.append(gate_a)
            gate_up_b.append(torch.cat([tensors[("gate_proj", "B")][1], tensors[("up_proj", "B")][1]], dim=0))
            down_a.append(tensors[("down_proj", "A")][1])
            down_b.append(tensors[("down_proj", "B")][1])
            consumed_indices.update(index for index, _tensor in tensors.values())

        base_name = prefix.rstrip(".")
        replacements[first_index] = [
            (f"{base_name}.base_layer.lora_A.weight", torch.cat(gate_up_a, dim=0)),
            (f"{base_name}.base_layer.lora_B.weight", torch.cat(gate_up_b, dim=1)),
            (f"{base_name}.lora_A.weight", torch.cat(down_a, dim=0)),
            (f"{base_name}.lora_B.weight", torch.cat(down_b, dim=1)),
        ]
        packed_groups += 1
        packed_tensors += len(replacements[first_index])

    for index, item in enumerate(materialized):
        if index in replacements:
            yield from replacements[index]
        if index not in consumed_indices:
            yield item

    debug = os.getenv("VERL_MEGATRON_LORA_PACK_3D_MOE_ADAPTER_DEBUG", "0").lower() in {"1", "true", "yes"}
    if debug:
        print(
            "[verl.megatron_lora] packed 3D MoE adapter tensors for vLLM: "
            f"model_type={model_type} groups={packed_groups} "
            f"expert_tensors_in={len(consumed_indices)} packed_tensors_out={packed_tensors}",
            flush=True,
        )


__all__ = [
    "count_adapter_parameters",
    "print_adapter_info",
    "convert_megatron_to_hf_target_modules",
    "build_peft_config_for_vllm",
    "add_base_layer_suffix",
    "gather_ep_lora_adapter_weights_for_vllm",
    "pack_3d_moe_lora_adapter_weights_for_vllm",
]
