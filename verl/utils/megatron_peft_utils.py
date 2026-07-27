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

from typing import Callable

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
    # GDN mappings
    "in_proj": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
    "out_proj": ["out_proj"],
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
def add_base_layer_to_name(name: str) -> str:
    """Insert ``.base_layer`` before the leaf field of a parameter name."""
    if ".base_layer." in name or "." not in name:
        return name

    prefix, leaf = name.rsplit(".", 1)
    return f"{prefix}.base_layer.{leaf}"


def remove_base_layer_from_name(name: str) -> str:
    """Remove the first ``.base_layer`` component from a parameter name."""
    return name.replace(".base_layer.", ".", 1)


def resolve_base_layer_name(
    name: str,
    *,
    exists: Callable[[str], bool],
) -> str:
    """Resolve a parameter name against a target namespace.

    The resolver keeps the original name when it already exists. Otherwise it
    tries the single alternative form by either adding or removing one
    ``.base_layer`` component.
    """
    if exists(name):
        return name

    if ".base_layer." in name:
        candidate = remove_base_layer_from_name(name)
    else:
        candidate = add_base_layer_to_name(name)

    if candidate != name and exists(candidate):
        return candidate

    return name


__all__ = [
    "count_adapter_parameters",
    "print_adapter_info",
    "convert_megatron_to_hf_target_modules",
    "add_base_layer_to_name",
    "build_peft_config_for_vllm",
    "remove_base_layer_from_name",
    "resolve_base_layer_name",
]
