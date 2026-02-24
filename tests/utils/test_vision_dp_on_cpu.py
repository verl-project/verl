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
Unit tests for Vision Data Parallel utilities (CPU-only, no distributed).
"""

import pytest
import torch

from verl.utils.vision_dp import (
    assign_images_to_dp_ranks,
    gather_vision_embeddings,
    get_image_embedding_counts,
    get_image_patch_counts,
    prepare_local_vision_inputs,
)


class TestGetImagePatchCounts:
    @pytest.mark.parametrize(
        "grid_thw,expected",
        [
            ([[2, 4, 4], [1, 2, 2], [1, 8, 8]], [32, 4, 64]),
            ([[1, 4, 4]], [16]),
            ([[4, 4, 4]], [64]),
        ],
        ids=["multi-image", "single-image", "video-frames"],
    )
    def test_patch_counts_various_grids_correct_products(self, grid_thw, expected):
        counts = get_image_patch_counts(torch.tensor(grid_thw))
        assert counts == expected

    def test_patch_counts_empty_input_returns_empty_list(self):
        counts = get_image_patch_counts(torch.empty((0, 3), dtype=torch.long))
        assert counts == []


class TestGetImageEmbeddingCounts:
    @pytest.mark.parametrize(
        "grid_thw,merge_size,expected",
        [
            ([[1, 8, 8]], 1, [64]),
            ([[1, 8, 8]], 2, [16]),
            ([[1, 6, 6], [1, 4, 4]], 2, [9, 4]),
        ],
        ids=["no-merge", "merge-2", "multi-image-merge"],
    )
    def test_embedding_counts_with_merge_size_correct(self, grid_thw, merge_size, expected):
        counts = get_image_embedding_counts(torch.tensor(grid_thw), merge_size)
        assert counts == expected


class TestAssignImagesToDpRanks:
    @pytest.mark.parametrize(
        "patch_counts,dp_size,expected_all_assigned",
        [
            ([100, 100, 100, 100], 2, True),
            ([100, 200, 300], 1, True),
            ([100, 100, 100, 100, 100, 100], 3, True),
        ],
        ids=["balanced-2ranks", "single-rank", "balanced-3ranks"],
    )
    def test_assign_all_images_distributed(self, patch_counts, dp_size, expected_all_assigned):
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size)
        all_assigned = []
        for a in assignments:
            all_assigned.extend(a)
        assert sorted(all_assigned) == list(range(len(patch_counts)))
        assert sum(loads) == sum(patch_counts)

    def test_assign_fewer_images_than_ranks_all_assigned(self):
        assignments, loads = assign_images_to_dp_ranks([100, 200], dp_size=4)
        non_empty = sum(1 for a in assignments if len(a) > 0)
        assert non_empty == 2
        all_assigned = set()
        for a in assignments:
            all_assigned.update(a)
        assert all_assigned == {0, 1}

    def test_assign_empty_input_returns_empty(self):
        assignments, loads = assign_images_to_dp_ranks([], dp_size=4)
        assert all(len(a) == 0 for a in assignments)
        assert all(load == 0 for load in loads)

    def test_assign_image_order_preserved_contiguous(self):
        assignments, _ = assign_images_to_dp_ranks([10, 20, 30, 40, 50], dp_size=2)
        for rank_assignment in assignments:
            assert rank_assignment == sorted(rank_assignment)

    def test_assign_load_balanced_unequal_patches(self):
        """With unequal patch counts, greedy balancing should reduce imbalance."""
        # 4096 + 256 + 256 + 256 = 4864, target per rank = 2432
        patch_counts = [4096, 256, 256, 256]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        # All images must be assigned
        all_assigned = []
        for a in assignments:
            all_assigned.extend(a)
        assert sorted(all_assigned) == [0, 1, 2, 3]
        # Load imbalance should be less than the naive count-based split (8.5x)
        max_load = max(loads)
        min_load = min(load for load in loads if load > 0)
        assert max_load / min_load < 8.0


class TestPrepareLocalVisionInputs:
    def test_prepare_two_images_splits_correctly(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 6, 6], [1, 8, 8]])  # 36 + 64 = 100
        image_assignments = [[0], [1]]

        # Rank 0
        pix, grid, indices = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=0)
        assert pix.shape[0] == 36
        assert grid.shape[0] == 1
        assert indices == [0]
        assert torch.allclose(pix, pixel_values[:36])

        # Rank 1
        pix, grid, indices = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=1)
        assert pix.shape[0] == 64
        assert indices == [1]
        assert torch.allclose(pix, pixel_values[36:100])

    def test_prepare_multiple_contiguous_images_per_rank(self):
        pixel_values = torch.randn(200, 768)
        grid_thw = torch.tensor([[1, 5, 10]] * 4)  # 4 x 50 patches
        image_assignments = [[0, 1], [2, 3]]

        pix, grid, indices = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=0)
        assert pix.shape[0] == 100
        assert grid.shape[0] == 2
        assert indices == [0, 1]
        assert torch.allclose(pix, pixel_values[:100])

    def test_prepare_empty_rank_returns_empty(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])
        image_assignments = [[0], []]

        pix, grid, indices = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=1)
        assert pix.shape[0] == 0
        assert grid.shape[0] == 0
        assert indices == []

    def test_prepare_grid_thw_preserved(self):
        pixel_values = torch.randn(150, 768)
        grid_thw = torch.tensor([[1, 5, 5], [2, 5, 5], [3, 5, 5]])  # 25 + 50 + 75
        image_assignments = [[0, 1], [2]]

        _, local_grid, _ = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=0)
        assert local_grid.shape == (2, 3)
        assert torch.equal(local_grid[0], grid_thw[0])
        assert torch.equal(local_grid[1], grid_thw[1])


class TestGatherVisionEmbeddings:
    def test_gather_none_group_returns_input(self):
        embeddings = torch.randn(10, 64)
        result = gather_vision_embeddings(embeddings, dp_group=None, all_counts=[10])
        assert torch.equal(result, embeddings)


class TestIntegration:
    def test_full_workflow_all_patches_covered(self):
        grid_thw = torch.tensor([[1, 4, 4], [1, 8, 8], [1, 4, 4], [1, 6, 6], [1, 4, 4]])
        total_patches = 16 + 64 + 16 + 36 + 16  # 148
        pixel_values = torch.randn(total_patches, 768)

        patch_counts = get_image_patch_counts(grid_thw)
        assert patch_counts == [16, 64, 16, 36, 16]

        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        all_assigned = []
        for a in assignments:
            all_assigned.extend(a)
        assert sorted(all_assigned) == [0, 1, 2, 3, 4]

        total_local_patches = 0
        for rank in range(2):
            pix, grid, indices = prepare_local_vision_inputs(pixel_values, grid_thw, assignments, dp_rank=rank)
            expected = sum(patch_counts[i] for i in indices)
            assert pix.shape[0] == expected
            assert grid.shape[0] == len(indices)
            total_local_patches += pix.shape[0]

        assert total_local_patches == total_patches

    def test_same_size_images_4_ranks_balanced(self):
        num_images = 50
        grid_thw = torch.tensor([[1, 8, 8]] * num_images)
        patch_counts = get_image_patch_counts(grid_thw)
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)

        for rank in range(4):
            assert 12 <= len(assignments[rank]) <= 13
        for load in loads:
            assert load in [768, 832]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
