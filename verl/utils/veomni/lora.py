# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Non-mutating merged export for VeOmni-native dense LoRA actors."""

import torch
from torch.distributed.tensor import DTensor


def merged_dense_lora_weights(model, device):
    """Yield full HF weights, merging one dense adapter into temporary storage.

    Every FSDP rank must consume this iterator in the same order. Collectives
    gather one base parameter and (for a wrapped weight) its A/B tensors. No
    training parameter, adapter state or optimizer storage is modified.
    """
    from veomni.lora import VeOmniLoraModel
    from veomni.lora.layers import LoraLinear

    if not isinstance(model, VeOmniLoraModel):
        raise TypeError("Expected a VeOmni-native LoRA model.")
    if model.get_lora_config().has_moe:
        raise NotImplementedError("Merged VeOmni rollout export currently supports dense LoRA only.")
    wrapped = {name: layer for name, layer in model.named_modules() if isinstance(layer, LoraLinear)}
    if any(layer.merged for layer in wrapped.values()):
        raise ValueError("Rollout export requires unmerged training adapters.")

    def full(tensor):
        tensor = tensor.detach().to(device, non_blocking=True)
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.clone()

    for name, param in model.state_dict().items():
        if any(part in {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"} for part in name.split(".")):
            continue
        with torch.no_grad():
            tensor = full(param)
            layer_name = name.removesuffix(".base_layer.weight")
            if layer_name in wrapped:
                layer = wrapped[layer_name]
                adapter = layer.active_adapter
                a = full(layer.lora_A[adapter].weight)
                b = full(layer.lora_B[adapter].weight)
                # FP32 accumulation also supports BF16 adapters on CPU; cast
                # only the temporary result back to the base storage dtype.
                tensor = torch.addmm(tensor.float(), b.float(), a.float(), beta=1, alpha=layer.scaling[adapter]).to(
                    tensor.dtype
                )
            hf_name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        yield hf_name, tensor
