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


def build_peft_config_for_vllm(lora_config: dict) -> dict:
    """Build the ``peft_config`` every rollout backend receives, from megatron's LoRA config.

    Args:
        lora_config: Megatron lora configuration dictionary.

    Returns:
        A dict accepted by both vLLM's PEFTHelper.from_dict() and SGLang's adapter loader.
    """
    from peft import PeftType, TaskType

    return {
        "task_type": TaskType.CAUSAL_LM,
        "peft_type": PeftType.LORA,
        "r": lora_config.get("rank", 0),
        "lora_alpha": lora_config.get("alpha", 32),
        # vLLM doesn't really use target_modules to determine which modules
        # to apply LoRA to, so we set "all-linear" as a placeholder.
        "target_modules": "all-linear",
        "bias": "none",
        "lora_dropout": lora_config.get("dropout", 0.0),
    }


__all__ = [
    "count_adapter_parameters",
    "print_adapter_info",
    "build_peft_config_for_vllm",
]
