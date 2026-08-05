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
"""PEFT configuration of Megatron for verl."""


def get_peft_cls(model_config, bridge, provider, dtype=None):
    """Create a Megatron-Bridge PEFT object from ``model_config.lora``."""
    if not hasattr(model_config, "lora"):
        return None

    lora_cfg = model_config.lora
    if lora_cfg.get("rank", 0) <= 0:
        return None

    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"

    from megatron.bridge.peft.utils import create_peft

    peft_cls = create_peft(lora_cfg, dtype=dtype)
    print(
        f"Enabling {lora_cfg.get('type', 'lora').upper()} with rank={lora_cfg.get('rank')}, "
        f"alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout')}"
    )
    return peft_cls


__all__ = [
    "get_peft_cls",
]
