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

import torch


def split_buffer_updates(
    model: torch.nn.Module, weights: list[tuple[str, torch.Tensor]]
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates."""
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[tuple[str, torch.Tensor]],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
