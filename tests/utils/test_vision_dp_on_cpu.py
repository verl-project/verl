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
Unit tests for Vision Data Parallel utilities.
"""

import pytest
import torch

from verl.utils.vision_dp import (
    assign_images_to_dp_ranks,
    get_image_patch_counts,
    prepare_local_vision_inputs,
)


class TestGetImagePatchCounts:
    """Tests for get_image_patch_counts function."""

    def test_basic_patch_counts(self):
        """Test basic patch count computation."""
        grid_thw = torch.tensor(
            [
                [2, 4, 4],  # 2*4*4 = 32
                [1, 2, 2],  # 1*2*2 = 4
                [1, 8, 8],  # 1*8*8 = 64
            ]
        )
        counts = get_image_patch_counts(grid_thw)
        assert counts == [32, 4, 64]

    def test_single_image(self):
        """Test with a single image."""
        grid_thw = torch.tensor([[1, 4, 4]])  # 16 patches
        counts = get_image_patch_counts(grid_thw)
        assert counts == [16]

    def test_empty_input(self):
        """Test with empty input."""
        grid_thw = torch.empty((0, 3), dtype=torch.long)
        counts = get_image_patch_counts(grid_thw)
        assert counts == []

    def test_video_frames(self):
        """Test with video (multiple temporal frames)."""
        grid_thw = torch.tensor(
            [
                [4, 4, 4],  # 4 frames, 4*4 patches each = 64 total
            ]
        )
        counts = get_image_patch_counts(grid_thw)
        assert counts == [64]


class TestAssignImagesToDpRanks:
    """Tests for assign_images_to_dp_ranks function."""

    def test_balanced_assignment(self):
        """Test balanced assignment with equal-sized images."""
        patch_counts = [100, 100, 100, 100]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Each rank should get 2 images
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2
        # Loads should be equal
        assert loads[0] == 200
        assert loads[1] == 200

    def test_imbalanced_images(self):
        """Test with one large image and several small ones."""
        patch_counts = [500, 100, 100, 100]  # One large image
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Large image (index 0) should be on one rank
        # Small images should fill the other rank
        total_assigned = sum(len(a) for a in assignments)
        assert total_assigned == 4

        # The greedy algorithm should assign large image to one rank
        # and remaining images to fill up the other
        assert 0 in assignments[0] or 0 in assignments[1]

    def test_fewer_images_than_ranks(self):
        """Test when number of images is less than dp_size."""
        patch_counts = [100, 200]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)

        # Only 2 ranks should have images
        non_empty_ranks = sum(1 for a in assignments if len(a) > 0)
        assert non_empty_ranks == 2

        # All images should be assigned
        all_assigned = set()
        for a in assignments:
            all_assigned.update(a)
        assert all_assigned == {0, 1}

    def test_empty_input(self):
        """Test with no images."""
        patch_counts = []
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)

        assert all(len(a) == 0 for a in assignments)
        assert all(load == 0 for load in loads)

    def test_single_rank(self):
        """Test with dp_size=1 (no parallelism)."""
        patch_counts = [100, 200, 300]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=1)

        # All images should go to the single rank
        assert assignments == [[0, 1, 2]]
        assert loads == [600]

    def test_equal_images_equal_size(self):
        """Test perfect balance: same number of equal-sized images per rank."""
        patch_counts = [100, 100, 100, 100, 100, 100]  # 6 images
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=3)

        # Each rank should get 2 images
        assert all(len(a) == 2 for a in assignments)
        # All loads should be equal
        assert all(load == 200 for load in loads)

    def test_image_order_preserved(self):
        """Test that image indices within each rank are sorted."""
        patch_counts = [10, 20, 30, 40, 50]
        assignments, _ = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Indices within each rank should be sorted
        for rank_assignment in assignments:
            assert rank_assignment == sorted(rank_assignment)


class TestPrepareLocalVisionInputs:
    """Tests for prepare_local_vision_inputs function."""

    def test_basic_extraction(self):
        """Test basic local input extraction."""
        # Create test data: 100 patches total
        pixel_values = torch.randn(100, 768)  # 100 patches, 768 dim
        grid_thw = torch.tensor(
            [
                [1, 6, 6],  # 36 patches (indices 0-35)
                [1, 8, 8],  # 64 patches (indices 36-99)
            ]
        )

        # Assignment: rank 0 -> [0], rank 1 -> [1]
        image_assignments = [[0], [1]]

        # Rank 0's inputs
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )

        assert local_pix.shape[0] == 36
        assert local_grid.shape[0] == 1
        assert local_indices == [0]
        assert torch.allclose(local_pix, pixel_values[:36])

        # Rank 1's inputs
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )

        assert local_pix.shape[0] == 64
        assert local_grid.shape[0] == 1
        assert local_indices == [1]
        assert torch.allclose(local_pix, pixel_values[36:100])

    def test_multiple_images_per_rank(self):
        """Test extraction when a rank has multiple images."""
        # Create test data: 200 patches total (50 + 50 + 50 + 50)
        pixel_values = torch.randn(200, 768)
        grid_thw = torch.tensor(
            [
                [1, 5, 10],  # 50 patches
                [1, 5, 10],  # 50 patches
                [1, 5, 10],  # 50 patches
                [1, 5, 10],  # 50 patches
            ]
        )

        # Assignment: rank 0 -> [0, 2], rank 1 -> [1, 3]
        image_assignments = [[0, 2], [1, 3]]

        # Rank 0's inputs (images 0 and 2)
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )

        assert local_pix.shape[0] == 100  # 50 + 50
        assert local_grid.shape[0] == 2
        assert local_indices == [0, 2]

        # Verify correct patches are extracted
        expected = torch.cat([pixel_values[0:50], pixel_values[100:150]], dim=0)
        assert torch.allclose(local_pix, expected)

    def test_empty_rank(self):
        """Test extraction when a rank has no images assigned."""
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])  # 100 patches

        # Only rank 0 has the image, rank 1 is empty
        image_assignments = [[0], []]

        # Rank 1's inputs (empty)
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )

        assert local_pix.shape[0] == 0
        assert local_grid.shape[0] == 0
        assert local_indices == []

    def test_grid_thw_preserved(self):
        """Test that grid_thw values are correctly extracted."""
        pixel_values = torch.randn(150, 768)
        grid_thw = torch.tensor(
            [
                [1, 5, 5],  # 25 patches
                [2, 5, 5],  # 50 patches
                [3, 5, 5],  # 75 patches
            ]
        )

        image_assignments = [[0, 2], [1]]

        # Rank 0 should have grids for images 0 and 2
        _, local_grid, _ = prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=0)

        assert local_grid.shape == (2, 3)
        assert torch.equal(local_grid[0], grid_thw[0])
        assert torch.equal(local_grid[1], grid_thw[2])


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test the complete workflow of image distribution."""
        # Simulate 5 images with different sizes
        grid_thw = torch.tensor(
            [
                [1, 4, 4],  # 16 patches
                [1, 8, 8],  # 64 patches
                [1, 4, 4],  # 16 patches
                [1, 6, 6],  # 36 patches
                [1, 4, 4],  # 16 patches
            ]
        )

        total_patches = 16 + 64 + 16 + 36 + 16  # 148 patches
        pixel_values = torch.randn(total_patches, 768)

        # Step 1: Get patch counts
        patch_counts = get_image_patch_counts(grid_thw)
        assert patch_counts == [16, 64, 16, 36, 16]

        # Step 2: Assign images to 2 ranks
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Verify all images are assigned
        all_assigned = []
        for a in assignments:
            all_assigned.extend(a)
        assert sorted(all_assigned) == [0, 1, 2, 3, 4]

        # Step 3: Extract local inputs for each rank
        total_local_patches = 0
        for rank in range(2):
            local_pix, local_grid, local_indices = prepare_local_vision_inputs(
                pixel_values, grid_thw, assignments, dp_rank=rank
            )

            # Verify consistency
            expected_patches = sum(patch_counts[i] for i in local_indices)
            assert local_pix.shape[0] == expected_patches
            assert local_grid.shape[0] == len(local_indices)

            total_local_patches += local_pix.shape[0]

        # Total patches across all ranks should equal original
        assert total_local_patches == total_patches

    def test_same_size_images(self):
        """Test with all same-size images (user's scenario)."""
        num_images = 50
        patch_per_image = 64  # 8x8 patches

        grid_thw = torch.tensor([[1, 8, 8]] * num_images)
        total_patches = num_images * patch_per_image
        _ = torch.randn(total_patches, 768)

        patch_counts = get_image_patch_counts(grid_thw)
        assert all(c == 64 for c in patch_counts)

        # With 4 DP ranks
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)

        # Each rank should get approximately 12-13 images
        for rank in range(4):
            assert 12 <= len(assignments[rank]) <= 13

        # Loads should be balanced (either 12*64=768 or 13*64=832)
        for load in loads:
            assert load in [768, 832]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
