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
"""
Vision Data Parallel utilities.

Distribute whole images across SP ranks, not patches within images.
Each rank runs ViT on its assigned images, then all-gather combines embeddings.
Backward all_reduce(SUM) recovers complete gradients before slicing by assignment.
"""

from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)


def get_image_patch_counts(grid_thw: torch.Tensor) -> List[int]:
    """Return [t*h*w for each image] from a [num_images, 3] grid_thw tensor."""
    if grid_thw.numel() == 0:
        raise ValueError("grid_thw is empty — Vision DP should only be called when images are present")
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def get_image_embedding_counts(grid_thw: torch.Tensor, spatial_merge_size: int = 1) -> List[int]:
    """Return per-image embedding counts after spatial merging: t * (h/merge) * (w/merge)."""
    if grid_thw.numel() == 0:
        raise ValueError("grid_thw is empty — Vision DP should only be called when images are present")

    if spatial_merge_size == 1:
        return get_image_patch_counts(grid_thw)

    # Apply spatial merging: h and w are divided by spatial_merge_size
    t = grid_thw[:, 0]
    h = grid_thw[:, 1] // spatial_merge_size
    w = grid_thw[:, 2] // spatial_merge_size
    return (t * h * w).tolist()


def assign_images_to_dp_ranks(
    patch_counts: List[int],
    dp_size: int,
) -> Tuple[List[List[int]], List[int]]:
    """Assign whole images to DP ranks via greedy contiguous bin-packing.

    Returns (image_assignments, rank_patch_counts). Images are kept contiguous
    so the gather result needs no reordering.
    """
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")

    num_images = len(patch_counts)
    if num_images == 0:
        raise ValueError("patch_counts is empty — Vision DP should only be called when images are present")

    image_assignments: List[List[int]] = [[] for _ in range(dp_size)]
    rank_loads = [0] * dp_size

    remaining_patches = sum(patch_counts)
    img_idx = 0
    for rank in range(dp_size):
        remaining_ranks = dp_size - rank
        remaining_images = num_images - img_idx

        if remaining_images <= 0:
            break

        # Dynamic target: distribute remaining patches evenly among remaining ranks
        target = remaining_patches / remaining_ranks

        # Must leave at least 1 image for each remaining rank
        max_images = remaining_images - (remaining_ranks - 1)

        # Greedily add images until we reach the target load or hit the max
        count = 0
        while img_idx < num_images and count < max_images:
            image_assignments[rank].append(img_idx)
            rank_loads[rank] += patch_counts[img_idx]
            img_idx += 1
            count += 1

            # Stop early once we've reached the target (always take at least 1)
            if rank_loads[rank] >= target:
                break

        remaining_patches -= rank_loads[rank]

    return image_assignments, rank_loads


def prepare_local_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    image_assignments: List[List[int]],
    dp_rank: int,
    patch_counts: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Extract pixel values and grid_thw for this DP rank's assigned images.

    Exploits contiguous assignment: a single slice instead of per-image cat.

    Args:
        patch_counts: Pre-computed per-image patch counts [t*h*w, ...].
    """
    if dp_rank < 0 or dp_rank >= len(image_assignments):
        raise ValueError(
            f"dp_rank={dp_rank} out of range for image_assignments with "
            f"{len(image_assignments)} ranks"
        )

    local_indices = image_assignments[dp_rank]

    if len(local_indices) == 0:
        # This rank has no images assigned
        return (
            torch.empty(
                (0, pixel_values.shape[1]) if pixel_values.dim() > 1 else (0,),
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            ),
            torch.empty((0, 3), dtype=grid_thw.dtype, device=grid_thw.device),
            [],
        )

    # local_indices are contiguous (e.g. [2, 3, 4]), so use tensor slicing
    first_img_idx = local_indices[0]
    last_img_idx = local_indices[-1]

    patch_counts_tensor = torch.tensor(patch_counts, device=grid_thw.device, dtype=torch.long)
    offsets = torch.cat(
        (
            torch.zeros(1, device=grid_thw.device, dtype=patch_counts_tensor.dtype),
            torch.cumsum(patch_counts_tensor, dim=0),
        )
    )

    start_patch = int(offsets[first_img_idx].item())
    end_patch = int(offsets[last_img_idx + 1].item())

    local_pixel_values = pixel_values[start_patch:end_patch]
    local_grid_thw = grid_thw[first_img_idx : last_img_idx + 1]

    # Verify against independently computed sum of per-image patch counts
    expected_patches = sum(patch_counts[i] for i in local_indices)
    assert local_pixel_values.shape[0] == expected_patches, (
        f"[Vision DP] Local patch count mismatch: "
        f"extracted={local_pixel_values.shape[0]}, expected={expected_patches}, "
        f"local_indices={local_indices}"
    )

    return local_pixel_values, local_grid_thw, local_indices


class GatherVisionEmbeddings(Function):
    """
    All-gather vision embeddings with gradient support.

    Since images are assigned contiguously (rank 0 gets [0,1], rank 1 gets [2,3], etc.),
    we can simply concat gathered results without reordering.

    Forward: all_gather + remove padding + concat
    Backward: all_reduce(SUM) to aggregate gradients from all sequence shards,
              then slice to extract this rank's image gradients
    """

    @staticmethod
    def forward(
        ctx,
        local_embeddings: torch.Tensor,
        dp_group,
        all_counts: List[int],
    ) -> torch.Tensor:
        dp_size = dist.get_world_size(dp_group)
        if dp_size <= 1:
            raise RuntimeError(
                "GatherVisionEmbeddings.forward called with dp_size=1. "
                "Caller should short-circuit before reaching here."
            )
        dp_rank = dist.get_rank(dp_group)
        ctx.dp_size = dp_size
        ctx.dp_group = dp_group
        ctx.all_counts = all_counts
        ctx.dp_rank = dp_rank

        if not all_counts or len(all_counts) != dp_size:
            raise ValueError(
                f"all_counts length ({len(all_counts) if all_counts else 0}) "
                f"must equal dp_size ({dp_size})"
            )

        max_count = max(all_counts)
        if max_count == 0:
            raise RuntimeError(
                "all_counts are all zero — Vision DP gather should not be called "
                "when no images are present"
            )

        hidden_size = local_embeddings.shape[1] if local_embeddings.dim() > 1 else 1

        # Pad to same length for all_gather
        if local_embeddings.shape[0] < max_count:
            pad_size = max_count - local_embeddings.shape[0]
            padding = torch.zeros(
                (pad_size, hidden_size),
                dtype=local_embeddings.dtype,
                device=local_embeddings.device,
            )
            local_padded = torch.cat([local_embeddings, padding], dim=0)
        else:
            local_padded = local_embeddings.contiguous()

        # All-gather
        gathered = [torch.empty_like(local_padded) for _ in range(dp_size)]
        dist.all_gather(gathered, local_padded, group=dp_group)

        # Remove padding and concat (no reordering needed - contiguous assignment)
        result_chunks = [gathered[r][: all_counts[r]] for r in range(dp_size)]
        result = torch.cat(result_chunks, dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        dp_size = ctx.dp_size
        assert dp_size > 1, (
            f"GatherVisionEmbeddings.backward reached with dp_size={dp_size}. "
            "Forward should never be called with dp_size<=1."
        )

        all_counts = ctx.all_counts
        dp_rank = ctx.dp_rank
        dp_group = ctx.dp_group

        # all_reduce(SUM) aggregates partial gradients from all SP ranks:
        # each rank only has non-zero grad for vision tokens in its sequence shard.
        grad = grad_output.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_group)

        # Extract gradients for this rank's images (contiguous slice)
        start = sum(all_counts[:dp_rank])
        end = start + all_counts[dp_rank]
        local_grad = grad[start:end]

        return local_grad, None, None


def _unpack_deepstack(local_embeddings, deepstack_merger_list, dtype, device):
    """Unpack Qwen3-VL deepstack from forward output.

    If local_embeddings is a tuple (normal rank), split into (embeddings, deepstack_list).
    Otherwise (empty rank), create matching empty deepstack tensors with requires_grad
    so they participate in the backward all_reduce (prevents NCCL deadlock).
    """
    if isinstance(local_embeddings, tuple):
        return local_embeddings[0], local_embeddings[1]

    # Empty rank: create matching empty deepstack tensors
    num_deepstack = len(deepstack_merger_list)
    h = local_embeddings.shape[1]
    deepstack = [
        torch.empty((0, h), dtype=dtype, device=device).requires_grad_()
        for _ in range(num_deepstack)
    ]
    return local_embeddings, deepstack


def create_dp_vision_forward(original_forward):
    """Wrap VisionTransformer.forward for Vision DP (Data Parallel across SP ranks).

    Strategy:
    1. Distribute whole images to DP ranks (not patches within images)
    2. Each rank processes its assigned images independently
    3. All-gather embeddings at the end (contiguous assignment, no reordering)

    Gradient correctness: after all-gather in forward, each SP rank's inputs_embeds
    contains vision tokens from ALL images. But Ulysses gives each rank only its
    sequence shard. In backward, each rank only has non-zero gradient for vision
    tokens in its own shard. The all_reduce(SUM) in GatherVisionEmbeddings.backward
    aggregates partial gradients from all ranks, recovering the complete gradient.
    """

    def dp_vision_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        sp_group = get_ulysses_sequence_parallel_group()
        sp_size = get_ulysses_sequence_parallel_world_size(sp_group)
        sp_rank = get_ulysses_sequence_parallel_rank(sp_group)

        if sp_size <= 1:
            raise RuntimeError(
                f"sp_size={sp_size}, Vision DP should not be active — "
                "monkey-patch is only applied when sp_size > 1"
            )

        # Move grid_thw to CPU once to avoid repeated GPU->CPU syncs in
        # metadata helpers (grid_thw is a tiny [num_images, 3] tensor).
        grid_thw_cpu = grid_thw.cpu()

        # Step 1: Get image assignment based on patch counts
        patch_counts = get_image_patch_counts(grid_thw_cpu)
        total_patches = sum(patch_counts)

        assert hidden_states.shape[0] == total_patches, (
            f"[Vision DP] Input patch count mismatch: "
            f"hidden_states.shape[0]={hidden_states.shape[0]}, "
            f"sum(grid_thw products)={total_patches}, "
            f"grid_thw.shape={grid_thw.shape}"
        )

        # Get spatial_merge_size from model (VLMs like Qwen use merger to reduce embeddings)
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)

        # Calculate embedding counts (after merger) for gather verification
        embedding_counts = get_image_embedding_counts(grid_thw_cpu, spatial_merge_size)
        total_embeddings = sum(embedding_counts)

        image_assignments, _ = assign_images_to_dp_ranks(patch_counts, sp_size)

        # Step 2: Extract local inputs (pass CPU grid_thw and pre-computed patch_counts
        # to avoid redundant computation and GPU→CPU syncs)
        local_pixels, local_grid_thw, local_indices = prepare_local_vision_inputs(
            hidden_states, grid_thw_cpu, image_assignments, sp_rank, patch_counts=patch_counts
        )
        local_grid_thw = local_grid_thw.to(grid_thw.device)

        # Detect Qwen3-VL deepstack: model attribute, not return type,
        # because empty ranks don't call original_forward and can't inspect the return.
        has_deepstack = hasattr(self, "deepstack_merger_list")

        # Step 3: Process local images
        if local_pixels.shape[0] > 0:
            local_embeddings = original_forward(
                self, local_pixels, local_grid_thw, **kwargs
            )
        else:
            # This rank has no images, create empty tensor with correct hidden size
            config = getattr(self, "config", None)
            hidden_size = getattr(config, "out_hidden_size", None) or getattr(config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError(
                    f"Cannot determine hidden_size: self.config has neither "
                    f"out_hidden_size nor hidden_size. Model type: {type(self).__name__}"
                )

            local_embeddings = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # Empty rank must participate in autograd for backward all_reduce
            local_embeddings.requires_grad_()

        # Unpack Qwen3-VL deepstack: forward returns (embeddings, list[3 × Tensor])
        local_deepstack = None
        if has_deepstack:
            local_embeddings, local_deepstack = _unpack_deepstack(
                local_embeddings, self.deepstack_merger_list, hidden_states.dtype, hidden_states.device
            )

        # Step 4: All-gather (contiguous assignment, no reordering needed)
        # Compute per-rank embedding counts locally (grid_thw is replicated on all ranks)
        all_counts = [
            sum(embedding_counts[i] for i in image_assignments[r])
            for r in range(sp_size)
        ]
        all_embeddings = GatherVisionEmbeddings.apply(
            local_embeddings, sp_group, all_counts
        )

        assert all_embeddings.shape[0] == total_embeddings, (
            f"[Vision DP] Output embedding count mismatch: "
            f"all_embeddings.shape[0]={all_embeddings.shape[0]}, "
            f"expected={total_embeddings}"
        )

        # Step 5: All-gather deepstack embeddings (all ranks must participate)
        if local_deepstack is not None:
            gathered_deepstack = [
                GatherVisionEmbeddings.apply(ds, sp_group, all_counts)
                for ds in local_deepstack
            ]
            return all_embeddings, gathered_deepstack

        return all_embeddings

    return dp_vision_forward
