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

"""Utilities for initializing, resetting, and synchronizing PEFT LoRA adapters."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn


def _iter_lora_layers(module: nn.Module, adapter_name: str) -> Iterator[tuple[str, nn.Module]]:
    for name, child in module.named_modules():
        lora_a = getattr(child, "lora_A", None)
        lora_b = getattr(child, "lora_B", None)
        if lora_a is not None and lora_b is not None and adapter_name in lora_a and adapter_name in lora_b:
            yield name, child


def _svd_projection(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Return ``Sigma_r V_r^T`` without materializing a full rectangular SVD."""
    if weight.ndim != 2:
        raise ValueError(f"SVD LoRA initialization requires a matrix, got shape {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if rank > min(out_features, in_features):
        raise ValueError(f"LoRA rank {rank} exceeds the maximum rank of a {out_features}x{in_features} weight")

    # The smaller Gram matrix is substantially cheaper for the very rectangular
    # gate/up/down projections used by transformer FFNs.
    work = weight.detach().to(dtype=torch.float32)
    if out_features >= in_features:
        eigenvalues, right_vectors = torch.linalg.eigh(work.mT @ work)
        indices = torch.arange(in_features - 1, in_features - rank - 1, -1, device=weight.device)
        singular_values = eigenvalues.index_select(0, indices).clamp_min_(0).sqrt_()
        projection = singular_values[:, None] * right_vectors.index_select(1, indices).mT
    else:
        _, left_vectors = torch.linalg.eigh(work @ work.mT)
        indices = torch.arange(out_features - 1, out_features - rank - 1, -1, device=weight.device)
        projection = left_vectors.index_select(1, indices).mT @ work
    return projection


@torch.no_grad()
def initialize_lora_with_svd(
    module: nn.Module,
    *,
    adapter_name: str = "default",
    freeze_a: bool = True,
    allow_meta: bool = False,
) -> list[str]:
    """Initialize PEFT LoRA as ``A = Sigma_r V_r^T, B = 0``.

    PEFT applies its configured ``alpha / rank`` scaling after ``BA``. The
    stored A matrix is divided by that scaling so the effective projection is
    exactly ``Sigma_r V_r^T`` even when alpha differs from rank. When
    ``allow_meta`` is set, initialization is deferred on FSDP meta ranks whose
    state will be populated by the subsequent module-state broadcast.

    Returns:
        Names of the initialized or deferred LoRA layers.
    """
    initialized = []
    for name, layer in _iter_lora_layers(module, adapter_name):
        base_layer = layer.get_base_layer() if hasattr(layer, "get_base_layer") else None
        weight = getattr(base_layer, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError(f"LoRA layer {name!r} does not wrap a two-dimensional base weight")

        lora_a = layer.lora_A[adapter_name].weight
        lora_b = layer.lora_B[adapter_name].weight
        rank = lora_a.shape[0]
        scaling = float(layer.scaling[adapter_name])
        if scaling == 0:
            raise ValueError(f"LoRA layer {name!r} has zero scaling")

        lora_a.requires_grad_(not freeze_a)
        lora_b.requires_grad_(True)
        if weight.is_meta:
            if not allow_meta:
                raise ValueError(f"Cannot SVD-initialize meta weight in LoRA layer {name!r}")
            # FSDP initializes non-loading ranks on meta and later broadcasts
            # the fully initialized adapter from a rank that loaded weights.
            initialized.append(name)
            continue

        projection = _svd_projection(weight, rank=rank).div_(scaling)
        lora_a.copy_(projection.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.zero_()
        initialized.append(name)

    if not initialized:
        raise ValueError(f"No PEFT LoRA layers found for adapter {adapter_name!r}")
    return initialized


def freeze_lora_a(module: nn.Module, *, adapter_name: str = "default") -> list[str]:
    """Freeze A and leave B trainable for all layers in one PEFT adapter."""
    frozen = []
    for name, layer in _iter_lora_layers(module, adapter_name):
        layer.lora_A[adapter_name].weight.requires_grad_(False)
        layer.lora_B[adapter_name].weight.requires_grad_(True)
        frozen.append(name)
    if not frozen:
        raise ValueError(f"No PEFT LoRA layers found for adapter {adapter_name!r}")
    return frozen


@torch.no_grad()
def reset_lora_b(module: nn.Module, *, adapter_name: str = "default") -> list[str]:
    """Reset the learnable B factor to zero at an episode boundary."""
    reset = []
    for name, layer in _iter_lora_layers(module, adapter_name):
        layer.lora_B[adapter_name].weight.zero_()
        reset.append(name)
    if not reset:
        raise ValueError(f"No PEFT LoRA layers found for adapter {adapter_name!r}")
    return reset


@torch.no_grad()
def copy_lora_weights(
    source: nn.Module,
    destination: nn.Module,
    *,
    adapter_name: str = "default",
) -> list[str]:
    """Synchronize A and B factors between equivalent PEFT model replicas."""
    source_layers = dict(_iter_lora_layers(source, adapter_name))
    destination_layers = dict(_iter_lora_layers(destination, adapter_name))
    if source_layers.keys() != destination_layers.keys():
        missing = sorted(source_layers.keys() - destination_layers.keys())
        extra = sorted(destination_layers.keys() - source_layers.keys())
        raise ValueError(f"LoRA layer mismatch; missing at destination={missing}, extra at destination={extra}")
    if not source_layers:
        raise ValueError(f"No PEFT LoRA layers found for adapter {adapter_name!r}")

    for name, source_layer in source_layers.items():
        destination_layer = destination_layers[name]
        for factor_name in ("lora_A", "lora_B"):
            source_weight = getattr(source_layer, factor_name)[adapter_name].weight
            destination_weight = getattr(destination_layer, factor_name)[adapter_name].weight
            if source_weight.shape != destination_weight.shape:
                raise ValueError(
                    f"{name}.{factor_name} shape mismatch: {tuple(source_weight.shape)} != "
                    f"{tuple(destination_weight.shape)}"
                )
            destination_weight.copy_(source_weight.to(device=destination_weight.device, dtype=destination_weight.dtype))
    return sorted(source_layers)


@torch.no_grad()
def iter_merged_lora_weights(
    module: nn.Module,
    *,
    adapter_name: str = "default",
    strip_prefix: str = "base_model.model.",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield only base weights changed by a LoRA adapter, with the delta merged."""
    found = False
    for name, layer in _iter_lora_layers(module, adapter_name):
        base_layer = layer.get_base_layer() if hasattr(layer, "get_base_layer") else None
        weight = getattr(base_layer, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError(f"LoRA layer {name!r} does not wrap a two-dimensional base weight")
        normalized_name = name.removeprefix(strip_prefix)
        yield f"{normalized_name}.weight", weight.detach() + layer.get_delta_weight(adapter_name).detach()
        found = True
    if not found:
        raise ValueError(f"No PEFT LoRA layers found for adapter {adapter_name!r}")
