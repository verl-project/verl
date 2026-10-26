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
"""Sharded loading of the released MiniMax M3 checkpoint into an FSDP2 model.

Every rank reads only the rows of each tensor that its FSDP2 shard owns and
dequantizes them on the GPU, so no rank ever materializes the full model.

Checkpoint formats (``quantization_config`` of the release):

* attention / dense MLP / shared expert: MXFP8, ``weight`` F8_E4M3 and
  ``weight_scale_inv`` U8 (E8M0, one per 32 input columns). Dequantization
  ``fp8 * 2**(e8m0 - 127)`` is exact in BF16.
* routed experts: NVFP4 compressed-tensors, ``weight_packed`` U8 (two E2M1
  values, low nibble first), ``weight_scale`` F8_E4M3 (one per 16 input
  columns), ``weight_global_scale`` F32. ``e2m1 * weight_scale`` is exact in
  BF16; the global scale stays a separate FP32 buffer (see
  ``MiniMaxM3ExpertLinear``).
* router weight, ``e_score_correction_bias``: FP32. Norms, embedding, LM head: BF16.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from verl.utils.device import get_device_name, get_torch_device

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MXFP8_BLOCK = 32
NVFP4_BLOCK = 16


def canonical_name(name: str) -> str:
    """Strip PEFT and wrapper components so the name matches checkpoint keys."""
    name = name.removeprefix("base_model.model.")
    for part in (".base_layer.", "._checkpoint_wrapped_module.", "._orig_mod.", "._fsdp_wrapped_module."):
        name = name.replace(part, ".")
    return name


def dequant_mxfp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """[rows, cols] F8_E4M3 and [rows, cols/32] E8M0 -> exact BF16."""
    rows, cols = weight.shape
    if scale_inv.shape != (rows, cols // MXFP8_BLOCK):
        raise ValueError(f"MXFP8 scale shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)}")
    exponent = scale_inv.to(torch.int32) - 127
    if bool((scale_inv == 255).any()):
        raise ValueError("E8M0 NaN scale in checkpoint")
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    value = weight.float().view(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK) * scale[..., None]
    out = value.view(rows, cols).to(torch.bfloat16)
    if not torch.equal(out.float(), value.view(rows, cols)):
        raise ValueError("MXFP8 dequantization is not exact in BF16")
    return out


def dequant_nvfp4_partial(packed: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """[rows, cols/2] U8 and [rows, cols/16] F8_E4M3 -> ``e2m1 * block_scale`` (exact BF16).

    The per-tensor global scale is not applied here.
    """
    rows, half = packed.shape
    cols = half * 2
    if block_scale.shape != (rows, cols // NVFP4_BLOCK):
        raise ValueError(f"NVFP4 scale shape {tuple(block_scale.shape)} does not match packed {tuple(packed.shape)}")
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low = lut[(packed & 0x0F).long()]
    high = lut[(packed >> 4).long()]
    values = torch.stack((low, high), dim=-1).view(rows, cols)
    value = values.view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK) * block_scale.float()[..., None]
    out = value.view(rows, cols).to(torch.bfloat16)
    if not torch.equal(out.float(), value.view(rows, cols)):
        raise ValueError("NVFP4 partial dequantization is not exact in BF16")
    return out


class _CheckpointReader:
    def __init__(self, path: str):
        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _handle(self, key: str):
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = safe_open(str(self.path / filename), framework="pt", device="cpu")
        return self._handles[filename]

    def rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        return self._handle(key).get_slice(key)[start:stop]

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._handle(key).get_slice(key).get_shape())

    def full(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)


def _local_rows(param: torch.Tensor) -> tuple[int, int]:
    """Global row range of this rank's shard (FSDP2 shards dim 0)."""
    if not isinstance(param, DTensor):
        return 0, param.shape[0]
    shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
    if any(o != 0 for o in offset[1:]) or tuple(shape[1:]) != tuple(param.shape[1:]):
        raise ValueError(f"Only dim-0 sharding is supported, got placements {param.placements}")
    return offset[0], offset[0] + shape[0]


def _local_tensor(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if isinstance(param, DTensor) else param.data


def _load_weight_rows(reader: _CheckpointReader, key: str, start: int, stop: int, device) -> torch.Tensor:
    """Dequantized BF16 rows [start, stop) of the checkpoint weight behind ``key`` (``.weight``)."""
    base = key[: -len(".weight")]
    if reader.has(base + ".weight_packed"):
        packed = reader.rows(base + ".weight_packed", start, stop).to(device)
        scale = reader.rows(base + ".weight_scale", start, stop).to(device)
        return dequant_nvfp4_partial(packed, scale)
    if reader.has(base + ".weight_scale_inv"):
        weight = reader.rows(key, start, stop).to(device)
        scale_inv = reader.rows(base + ".weight_scale_inv", start, stop).to(device)
        return dequant_mxfp8(weight, scale_inv)
    tensor = reader.rows(key, start, stop).to(device)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Unexpected dtype {tensor.dtype} for unquantized weight {key}")
    return tensor


def _checkpoint_weight_shape(reader: _CheckpointReader, key: str) -> tuple[int, ...]:
    """Shape of the dequantized checkpoint weight behind ``key``."""
    base = key[: -len(".weight")] if key.endswith(".weight") else key
    if reader.has(base + ".weight_packed"):
        rows, half = reader.shape(base + ".weight_packed")
        return rows, half * 2
    return reader.shape(key)


def _init_lora(name: str, param: torch.Tensor, device, seed: int) -> None:
    """PEFT default LoRA init (A: kaiming_uniform(a=sqrt(5)), B: zeros), rank-independent.

    The full tensor is generated from a per-name seed so all ranks agree, then
    only the local rows are kept.
    """
    local = _local_tensor(param)
    if ".lora_B." in name:
        local.zero_()
        return
    if ".lora_A." not in name:
        raise ValueError(f"Unsupported adapter parameter {name}")
    generator = torch.Generator(device=device).manual_seed(seed ^ zlib.crc32(name.encode()))
    full = torch.empty(param.shape, dtype=torch.float32, device=device)
    torch.nn.init.kaiming_uniform_(full, a=math.sqrt(5), generator=generator)
    start, stop = _local_rows(param)
    local.copy_(full[start:stop].to(local.dtype))


@torch.no_grad()
def load_minimax_m3_sharded(model: torch.nn.Module, path: str, device, seed: int = 0) -> dict[str, int]:
    """Materialize ``model`` (FSDP2-sharded, on meta) on ``device`` from the checkpoint.

    Returns counters for the load log. Raises on any checkpoint key of the text
    backbone that the model does not consume, and on any model tensor that has
    no checkpoint source.
    """
    if isinstance(device, int):
        device = torch.device(get_device_name(), device)
    device = torch.device(device)
    if device.type == get_device_name() and device.index is None:
        device = torch.device(device.type, get_torch_device().current_device())
    reader = _CheckpointReader(path)
    model.to_empty(device=device)
    consumed: set[str] = set()
    stats = defaultdict(int)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            _init_lora(name, param, device, seed)
            stats["lora_init"] += 1
            continue
        key = canonical_name(name)
        source_shape = _checkpoint_weight_shape(reader, key)
        if source_shape != tuple(param.shape):
            # copy_ would broadcast a size-1 checkpoint dimension silently.
            raise ValueError(f"Shape mismatch for {key}: checkpoint {source_shape}, model {tuple(param.shape)}")
        start, stop = _local_rows(param)
        local = _local_tensor(param)
        if stop > start:
            rows = _load_weight_rows(reader, key, start, stop, device)
            if rows.shape != local.shape:
                raise ValueError(
                    f"Shard shape mismatch for {key}: loaded {tuple(rows.shape)}, local {tuple(local.shape)}"
                )
            local.copy_(rows.to(local.dtype))
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        for suffix in (".weight", ".weight_scale_inv", ".weight_packed", ".weight_scale"):
            if reader.has(base + suffix):
                consumed.add(base + suffix)
        stats["parameters"] += 1

    for name, buffer in model.named_buffers():
        key = canonical_name(name)
        if not reader.has(key):
            raise KeyError(f"No checkpoint tensor for buffer {name} ({key})")
        value = reader.full(key).to(device)
        if value.shape != buffer.shape:
            raise ValueError(f"Shape mismatch for {key}: checkpoint {tuple(value.shape)}, model {tuple(buffer.shape)}")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected FP32 checkpoint tensor for {key}, got {value.dtype}")
        buffer.copy_(value)
        consumed.add(key)
        stats["buffers"] += 1

    ignored = getattr(model, "ignored_checkpoint_prefixes", None)
    if ignored is None:
        inner = getattr(model, "get_base_model", lambda: model)()
        ignored = getattr(inner, "ignored_checkpoint_prefixes", ())
    num_layers = _num_layers(model)
    # A model with fewer layers than the checkpoint is a different model; only the
    # layer-prefix test fixtures ask for it, explicitly.
    allow_layer_slice = os.environ.get("VERL_MINIMAX_M3_ALLOW_LAYER_SLICE", "0") == "1"
    unused = []
    for key in reader.weight_map:
        if key in consumed or key.startswith(tuple(ignored)):
            continue
        layer = _layer_index(key)
        if allow_layer_slice and layer is not None and layer >= num_layers:
            stats["layer_slice_skipped"] += 1
            continue
        unused.append(key)
    if unused:
        raise KeyError(f"{len(unused)} checkpoint tensors not consumed, e.g. {unused[:8]}")
    stats["checkpoint_tensors_consumed"] = len(consumed)
    return dict(stats)


def _layer_index(key: str):
    marker = ".layers."
    if marker not in key:
        return None
    return int(key.split(marker, 1)[1].split(".", 1)[0])


def _num_layers(model: torch.nn.Module) -> int:
    inner = getattr(model, "get_base_model", lambda: model)()
    return inner.args.num_hidden_layers
