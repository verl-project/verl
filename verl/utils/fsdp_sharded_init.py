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
"""Build an FSDP2 model on the meta device and load each rank's shard straight from safetensors."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor, Shard

ENABLED = os.environ.get("VERL_FSDP2_SHARDED_INIT") == "1"

# NVFP4 dequantization as in vLLM's reference (tests/kernels/quantization/nvfp4_utils.py, Apache-2.0)
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequantize_nvfp4(packed, block_scale, global_scale, block_size=16):
    """uint8 [M, K/2] (low nibble first), FP8 block scales [M, K/16], FP32 global scale -> fp32 [M, K]."""
    m = packed.shape[0]
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(m, -1)
    values = _E2M1.to(packed.device)[(nibbles & 0x07).long()] * torch.where((nibbles & 0x08).bool(), -1.0, 1.0)
    k = values.shape[1]
    scale = block_scale.view(torch.float8_e4m3fn)[:m, : k // block_size].float() / (1.0 / global_scale.float())
    return (values.view(m, -1, block_size) * scale.unsqueeze(-1)).view(m, k)


def _dequantize_experts(packed, block_scale, global_scale):
    """[E, rows, K/2] experts with one global scale per expert or per row partition (e.g. [w1 | w3])."""
    experts = []
    for q, s, g in zip(packed, block_scale, global_scale, strict=True):
        g, r = g.reshape(-1), q.shape[0] // g.numel()
        experts.append(torch.cat([dequantize_nvfp4(q[i * r : (i + 1) * r], s[i * r :], g[i]) for i in range(len(g))]))
    return torch.stack(experts)


def init_lora_on_cpu(module: torch.nn.Module, dtype: torch.dtype, seed: int = 0) -> dict[str, torch.Tensor]:
    """Initialize the LoRA weights of a meta model on CPU, identically on every rank."""
    from peft.tuners.lora import LoraLayer

    torch.manual_seed(seed)
    for layer in module.modules():
        if isinstance(layer, LoraLayer):
            for adapter in layer.active_adapters:
                layer.lora_A[adapter].to_empty(device="cpu")
                layer.lora_B[adapter].to_empty(device="cpu")
                layer.reset_lora_parameters(adapter, True)
                layer.lora_A[adapter].to(dtype)
                layer.lora_B[adapter].to(dtype)
        elif hasattr(layer, "reset_lora_parameters"):  # model-native LoRA
            lora = [child for name, child in layer.named_children() if name.startswith("lora_")]
            for child in lora:
                child.to_empty(device="cpu")
            layer.reset_lora_parameters()
            for child in lora:
                child.to(dtype)
    return {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}


def _checkpoint_key(param_name: str) -> str:
    return param_name.removeprefix("base_model.model.").replace(".base_layer", "")


def load_sharded(module: torch.nn.Module, model_path: str, lora_init: dict[str, torch.Tensor], device) -> None:
    """Fill each rank's Shard(0) rows of an FSDP2-wrapped meta model; NVFP4 experts are dequantized."""
    module.to_empty(device=device)
    weight_map = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def read(key, lo, hi):
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(str(Path(model_path) / shard), "pt")
        return handles[shard].get_slice(key)[lo:hi]

    with torch.no_grad():
        for name, param in module.named_parameters():
            local = param.to_local() if isinstance(param, DTensor) else param.data
            lo, hi = 0, param.shape[0]
            if isinstance(param, DTensor):
                assert param.placements == (Shard(0),), (name, param.placements)
                rank, size = param.device_mesh.get_local_rank(), param.device_mesh.size()
                chunk = -(-param.shape[0] // size)
                lo, hi = min(rank * chunk, param.shape[0]), min((rank + 1) * chunk, param.shape[0])
            assert local.shape[0] == hi - lo, (name, local.shape, lo, hi)
            if name in lora_init:
                rows = lora_init[name][lo:hi]
            else:
                key = _checkpoint_key(name)
                rows = read(key, lo, hi).to(local.device)
                if f"{key}_weight_scale_2" in weight_map:
                    scales = (read(f"{key}_weight_scale{s}", lo, hi).to(local.device) for s in ("", "_2"))
                    rows = _dequantize_experts(rows, *scales)
            local.copy_(rows.to(local.dtype))
