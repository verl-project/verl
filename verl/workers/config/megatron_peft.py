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
"""PEFT configuration of Megatron for VERL."""


def get_peft_cls(model_config, bridge, provider, dtype=None):
    """Get PEFT class from model config.

    Args:
        model_config: Model configuration object.
        bridge: Megatron-Bridge AutoBridge instance.
        provider: Provider instance.

    Returns:
        PEFT configuration object (LoRAConfig, CanonicalLoRAConfig, DoRAConfig) or None.
    """

    peft_cls = None
    if not hasattr(model_config, "lora"):
        return peft_cls

    lora_cfg = model_config.lora
    # Only enable if rank > 0
    if lora_cfg.get("rank", 0) <= 0:
        return peft_cls

    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"

    from verl.models.mcore.bridge import CanonicalLoRA, DoRA, LoRA, VLMLoRA

    lora_dtype = lora_cfg.get("dtype", dtype)
    if lora_dtype is not None:
        from verl.utils.torch_dtypes import PrecisionType

        lora_dtype = PrecisionType.to_dtype(lora_dtype)

    lora_type = lora_cfg.get("type", "lora")
    if lora_type == "lora":
        peft_cls = LoRA(
            target_modules=lora_cfg.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            a2a_experimental=lora_cfg.get("a2a_experimental", False),
            lora_dtype=lora_dtype,
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    if lora_type == "vlm_lora":
        peft_cls = VLMLoRA(
            target_modules=lora_cfg.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            a2a_experimental=lora_cfg.get("a2a_experimental", False),
            lora_dtype=lora_dtype,
            freeze_vision_model=lora_cfg.get("freeze_vision_model", True),
            freeze_vision_projection=lora_cfg.get("freeze_vision_projection", True),
            freeze_language_model=lora_cfg.get("freeze_language_model", True),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    elif lora_type == "canonical_lora":
        peft_cls = CanonicalLoRA(
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "linear_q",
                    "linear_k",
                    "linear_v",
                    "linear_proj",
                    "linear_fc1_up",
                    "linear_fc1_gate",
                    "linear_fc2",
                ],
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    elif lora_type == "dora":
        peft_cls = DoRA(
            target_modules=lora_cfg.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )

    print(
        f"Enabling {lora_type.upper()} with rank={lora_cfg.get('rank')}, "
        f"alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout')}"
    )
    return peft_cls


__all__ = [
    "get_peft_cls",
]
