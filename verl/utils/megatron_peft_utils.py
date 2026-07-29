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


__all__ = [
    "count_adapter_parameters",
    "print_adapter_info",
    "convert_megatron_to_hf_target_modules",
    "build_peft_config_for_vllm",
    "add_base_layer_suffix",
]
