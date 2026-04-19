# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import NamedTuple

import torch


class BilinearInterpolationTensors(NamedTuple):
    device: torch.device
    idx_tensor: torch.Tensor
    weight_tensor: torch.Tensor
    grid_ts: torch.Tensor
    grid_hs: torch.Tensor
    grid_ws: torch.Tensor


def build_bilinear_interpolation_tensors(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    weight_dtype: torch.dtype,
) -> BilinearInterpolationTensors:
    device = grid_thw.device
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for h, w in zip(grid_hs, grid_ws, strict=False):
        h_size = int(h.item())
        w_size = int(w.item())
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h_size, device=device)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w_size, device=device)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs_floor + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs_floor + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side

        indices = [
            (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
            (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
        ]

        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.as_tensor(weight_list, dtype=weight_dtype, device=device)
    return BilinearInterpolationTensors(
        device=device,
        idx_tensor=idx_tensor,
        weight_tensor=weight_tensor,
        grid_ts=grid_ts,
        grid_hs=grid_hs,
        grid_ws=grid_ws,
    )


def merge_bilinear_interpolated_pos_embeds(
    pos_embeds: torch.Tensor,
    weight_tensor: torch.Tensor,
    grid_ts: torch.Tensor,
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
    merge_size: int,
) -> torch.Tensor:
    pos_embeds = pos_embeds * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    split_sizes = [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws, strict=False)]
    patch_pos_embeds = patch_pos_embeds.split(split_sizes)

    patch_pos_embeds_permute = []
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        t_size = int(t.item())
        h_size = int(h.item())
        w_size = int(w.item())
        pos_embed = pos_embed.repeat(t_size, 1)
        pos_embed = (
            pos_embed.view(t_size, h_size // merge_size, merge_size, w_size // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)

    return torch.cat(patch_pos_embeds_permute)
