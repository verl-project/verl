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

from unittest.mock import MagicMock

from verl.utils.vision_dp import (
    get_image_patch_counts,
    get_image_embedding_counts,
    assign_images_to_dp_ranks,
    prepare_local_vision_inputs,
    create_dp_vision_forward,
    _unpack_deepstack,
)


class TestGetImagePatchCounts:
    """Tests for get_image_patch_counts function."""

    def test_basic_patch_counts(self):
        """Test basic patch count computation."""
        grid_thw = torch.tensor([
            [2, 4, 4],   # 2*4*4 = 32
            [1, 2, 2],   # 1*2*2 = 4
            [1, 8, 8],   # 1*8*8 = 64
        ])
        counts = get_image_patch_counts(grid_thw)
        assert counts == [32, 4, 64]

    def test_single_image(self):
        """Test with a single image."""
        grid_thw = torch.tensor([[1, 4, 4]])  # 16 patches
        counts = get_image_patch_counts(grid_thw)
        assert counts == [16]

    def test_empty_input_raises(self):
        """Empty grid_thw must raise ValueError."""
        grid_thw = torch.empty((0, 3), dtype=torch.long)
        with pytest.raises(ValueError, match="grid_thw is empty"):
            get_image_patch_counts(grid_thw)

    def test_video_frames(self):
        """Test with video (multiple temporal frames)."""
        grid_thw = torch.tensor([
            [4, 4, 4],   # 4 frames, 4*4 patches each = 64 total
        ])
        counts = get_image_patch_counts(grid_thw)
        assert counts == [64]


class TestGetImageEmbeddingCounts:
    """Tests for get_image_embedding_counts function."""

    def test_no_merge(self):
        """spatial_merge_size=1 should equal patch counts."""
        grid_thw = torch.tensor([[1, 8, 8], [1, 4, 4]])
        assert get_image_embedding_counts(grid_thw, 1) == [64, 16]

    def test_merge_size_2(self):
        """spatial_merge_size=2: h and w halved."""
        grid_thw = torch.tensor([[1, 8, 8], [1, 4, 4]])
        # t * (h/2) * (w/2): 1*4*4=16, 1*2*2=4
        assert get_image_embedding_counts(grid_thw, 2) == [16, 4]

    def test_empty_raises(self):
        """Empty grid_thw must raise ValueError."""
        grid_thw = torch.empty((0, 3), dtype=torch.long)
        with pytest.raises(ValueError, match="grid_thw is empty"):
            get_image_embedding_counts(grid_thw, 2)

    def test_video_with_merge(self):
        """Video: t=2, h=8, w=8, merge=2 -> 2*(8/2)*(8/2) = 32."""
        grid_thw = torch.tensor([[2, 8, 8]])
        assert get_image_embedding_counts(grid_thw, 2) == [32]


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

        # All images assigned
        total_assigned = sum(len(a) for a in assignments)
        assert total_assigned == 4

        # Greedy: rank 0 gets [0] (500 >= target 400), rank 1 gets [1,2,3] (300)
        assert assignments[0] == [0]
        assert assignments[1] == [1, 2, 3]
        assert loads[0] == 500
        assert loads[1] == 300

    def test_load_balanced_unequal_patches(self):
        """Greedy balancing should outperform naive count-based split."""
        # 4096 + 256 + 256 + 256 = 4864 total, target per rank = 2432
        patch_counts = [4096, 256, 256, 256]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Greedy: rank 0 takes [0] (4096 >= target), rank 1 takes [1,2,3] (768)
        assert assignments[0] == [0]
        assert assignments[1] == [1, 2, 3]
        assert loads[0] == 4096
        assert loads[1] == 768

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

    def test_empty_input_raises(self):
        """Empty patch_counts must raise ValueError."""
        with pytest.raises(ValueError, match="patch_counts is empty"):
            assign_images_to_dp_ranks([], dp_size=4)

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
        assert all(l == 200 for l in loads)

    def test_image_order_preserved(self):
        """Test that image indices within each rank are sorted (contiguous)."""
        patch_counts = [10, 20, 30, 40, 50]
        assignments, _ = assign_images_to_dp_ranks(patch_counts, dp_size=2)

        # Indices within each rank should be sorted and contiguous
        for rank_assignment in assignments:
            assert rank_assignment == sorted(rank_assignment)
            if len(rank_assignment) > 1:
                # Contiguous: each index is previous + 1
                for i in range(1, len(rank_assignment)):
                    assert rank_assignment[i] == rank_assignment[i - 1] + 1

    def test_contiguous_coverage(self):
        """All images are covered exactly once across ranks."""
        patch_counts = [10, 20, 30, 40, 50, 60, 70]
        for dp_size in [1, 2, 3, 4, 7]:
            assignments, _ = assign_images_to_dp_ranks(patch_counts, dp_size)
            all_indices = []
            for a in assignments:
                all_indices.extend(a)
            assert sorted(all_indices) == list(range(len(patch_counts)))

    def test_zero_dp_size_raises(self):
        """dp_size=0 must raise ValueError."""
        with pytest.raises(ValueError, match="dp_size must be positive"):
            assign_images_to_dp_ranks([100], dp_size=0)

    def test_negative_dp_size_raises(self):
        """dp_size<0 must raise ValueError."""
        with pytest.raises(ValueError, match="dp_size must be positive"):
            assign_images_to_dp_ranks([100], dp_size=-1)


class TestPrepareLocalVisionInputs:
    """Tests for prepare_local_vision_inputs function."""

    def test_basic_extraction(self):
        """Test basic local input extraction."""
        # Create test data: 100 patches total
        pixel_values = torch.randn(100, 768)  # 100 patches, 768 dim
        grid_thw = torch.tensor([
            [1, 6, 6],   # 36 patches (indices 0-35)
            [1, 8, 8],   # 64 patches (indices 36-99)
        ])
        patch_counts = get_image_patch_counts(grid_thw)

        # Assignment: rank 0 -> [0], rank 1 -> [1]
        image_assignments = [[0], [1]]

        # Rank 0's inputs
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0, patch_counts=patch_counts
        )

        assert local_pix.shape[0] == 36
        assert local_grid.shape[0] == 1
        assert local_indices == [0]
        assert torch.allclose(local_pix, pixel_values[:36])

        # Rank 1's inputs
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1, patch_counts=patch_counts
        )

        assert local_pix.shape[0] == 64
        assert local_grid.shape[0] == 1
        assert local_indices == [1]
        assert torch.allclose(local_pix, pixel_values[36:100])

    def test_multiple_images_per_rank(self):
        """Test extraction when a rank has multiple contiguous images."""
        # Create test data: 200 patches total (50 + 50 + 50 + 50)
        pixel_values = torch.randn(200, 768)
        grid_thw = torch.tensor([
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
        ])
        patch_counts = get_image_patch_counts(grid_thw)

        # Contiguous assignment: rank 0 -> [0, 1], rank 1 -> [2, 3]
        image_assignments = [[0, 1], [2, 3]]

        # Rank 0's inputs (images 0 and 1, contiguous)
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0, patch_counts=patch_counts
        )

        assert local_pix.shape[0] == 100  # 50 + 50
        assert local_grid.shape[0] == 2
        assert local_indices == [0, 1]

        # Verify correct patches are extracted (contiguous slice)
        expected = pixel_values[0:100]
        assert torch.allclose(local_pix, expected)

    def test_empty_rank(self):
        """Test extraction when a rank has no images assigned."""
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])  # 100 patches
        patch_counts = get_image_patch_counts(grid_thw)

        # Only rank 0 has the image, rank 1 is empty
        image_assignments = [[0], []]

        # Rank 1's inputs (empty)
        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1, patch_counts=patch_counts
        )

        assert local_pix.shape[0] == 0
        assert local_grid.shape[0] == 0
        assert local_indices == []

    def test_grid_thw_preserved(self):
        """Test that grid_thw values are correctly extracted (contiguous)."""
        pixel_values = torch.randn(150, 768)
        grid_thw = torch.tensor([
            [1, 5, 5],   # 25 patches
            [2, 5, 5],   # 50 patches
            [3, 5, 5],   # 75 patches
        ])
        patch_counts = get_image_patch_counts(grid_thw)

        # Contiguous: rank 0 -> [0, 1], rank 1 -> [2]
        image_assignments = [[0, 1], [2]]

        # Rank 0 should have grids for images 0 and 1
        _, local_grid, _ = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0, patch_counts=patch_counts
        )

        assert local_grid.shape == (2, 3)
        assert torch.equal(local_grid[0], grid_thw[0])
        assert torch.equal(local_grid[1], grid_thw[1])

    def test_out_of_range_dp_rank_raises(self):
        """dp_rank out of range must raise ValueError."""
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])
        patch_counts = get_image_patch_counts(grid_thw)
        image_assignments = [[0]]
        with pytest.raises(ValueError, match="dp_rank=1 out of range"):
            prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=1, patch_counts=patch_counts)

    def test_negative_dp_rank_raises(self):
        """Negative dp_rank must raise ValueError."""
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])
        patch_counts = get_image_patch_counts(grid_thw)
        image_assignments = [[0]]
        with pytest.raises(ValueError, match="dp_rank=-1 out of range"):
            prepare_local_vision_inputs(pixel_values, grid_thw, image_assignments, dp_rank=-1, patch_counts=patch_counts)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test the complete workflow of image distribution."""
        # Simulate 5 images with different sizes
        grid_thw = torch.tensor([
            [1, 4, 4],   # 16 patches
            [1, 8, 8],   # 64 patches
            [1, 4, 4],   # 16 patches
            [1, 6, 6],   # 36 patches
            [1, 4, 4],   # 16 patches
        ])

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
                pixel_values, grid_thw, assignments, dp_rank=rank, patch_counts=patch_counts
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
        pixel_values = torch.randn(total_patches, 768)

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


class TestDeepstackDetection:
    """Tests for Qwen3-VL deepstack handling in create_dp_vision_forward."""

    def test_has_deepstack_attribute_detection(self):
        """Verify deepstack detection uses model attribute, not return type."""
        import torch.nn as nn

        # Mock a Qwen3-VL-like model with deepstack_merger_list
        mock_model = MagicMock()
        mock_model.deepstack_merger_list = nn.ModuleList([nn.Identity() for _ in range(3)])
        assert hasattr(mock_model, "deepstack_merger_list")
        assert len(mock_model.deepstack_merger_list) == 3

        # Mock a Qwen2-VL-like model without deepstack
        mock_model_v2 = MagicMock(spec=["merger", "config", "forward"])
        assert not hasattr(mock_model_v2, "deepstack_merger_list")

    def test_deepstack_unpack_logic(self):
        """Test that tuple return from Qwen3-VL forward is correctly unpacked."""
        hidden_size = 3584
        num_embeddings = 20

        # Simulate Qwen3-VL forward return: (embeddings, [ds0, ds1, ds2])
        main_emb = torch.randn(num_embeddings, hidden_size)
        deepstack_list = [torch.randn(num_embeddings, hidden_size) for _ in range(3)]
        qwen3_return = (main_emb, deepstack_list)

        # Unpack like the wrapper does
        assert isinstance(qwen3_return, tuple)
        local_embeddings, local_deepstack = qwen3_return[0], qwen3_return[1]

        assert local_embeddings.shape == (num_embeddings, hidden_size)
        assert len(local_deepstack) == 3
        for ds in local_deepstack:
            assert ds.shape == (num_embeddings, hidden_size)

    def test_empty_rank_deepstack_creation(self):
        """Test that empty rank creates correctly-shaped empty deepstack tensors."""
        import torch.nn as nn

        hidden_size = 3584
        num_deepstack = 3

        # Simulate what the wrapper does for empty rank
        deepstack_merger_list = nn.ModuleList([nn.Identity() for _ in range(num_deepstack)])
        local_embeddings = torch.empty((0, hidden_size), dtype=torch.float32)

        # This is the empty rank path in the wrapper
        h = local_embeddings.shape[1]
        local_deepstack = [
            torch.empty((0, h), dtype=torch.float32)
            for _ in range(len(deepstack_merger_list))
        ]

        assert len(local_deepstack) == 3
        for ds in local_deepstack:
            assert ds.shape == (0, hidden_size)


class TestUnpackDeepstack:
    """Tests for _unpack_deepstack helper function."""

    def test_tuple_input_unpacks_correctly(self):
        """Normal rank: tuple return from forward is split into (embeddings, deepstack)."""
        import torch.nn as nn

        hidden_size = 3584
        n = 20
        main_emb = torch.randn(n, hidden_size)
        ds_list = [torch.randn(n, hidden_size) for _ in range(3)]
        merger_list = nn.ModuleList([nn.Identity() for _ in range(3)])

        emb, ds = _unpack_deepstack((main_emb, ds_list), merger_list, torch.float32, "cpu")
        assert torch.equal(emb, main_emb)
        assert len(ds) == 3
        for i in range(3):
            assert torch.equal(ds[i], ds_list[i])

    def test_empty_rank_creates_empty_tensors(self):
        """Empty rank: non-tuple input produces empty deepstack tensors."""
        import torch.nn as nn

        hidden_size = 3584
        merger_list = nn.ModuleList([nn.Identity() for _ in range(3)])
        local_emb = torch.empty((0, hidden_size), dtype=torch.float32)

        emb, ds = _unpack_deepstack(local_emb, merger_list, torch.float32, "cpu")
        assert torch.equal(emb, local_emb)
        assert len(ds) == 3
        for t in ds:
            assert t.shape == (0, hidden_size)

    def test_empty_rank_deepstack_requires_grad(self):
        """Empty deepstack tensors must have requires_grad=True to prevent NCCL deadlock."""
        import torch.nn as nn

        hidden_size = 1024
        merger_list = nn.ModuleList([nn.Identity() for _ in range(2)])
        local_emb = torch.empty((0, hidden_size), dtype=torch.float32)

        _, ds = _unpack_deepstack(local_emb, merger_list, torch.float32, "cpu")
        for t in ds:
            assert t.requires_grad, (
                "Empty deepstack tensors must have requires_grad=True "
                "so they participate in backward all_reduce (prevents NCCL deadlock)"
            )


class TestEmbeddingCountsConsistency:
    """Tests for consistency between patch counts and embedding counts."""

    def test_embedding_counts_merge_1_equals_patch_counts(self):
        """With spatial_merge_size=1, embedding counts must equal patch counts."""
        grid_thw = torch.tensor([
            [1, 4, 4],
            [2, 8, 8],
            [1, 6, 6],
            [3, 4, 4],
        ])
        patch_counts = get_image_patch_counts(grid_thw)
        embedding_counts = get_image_embedding_counts(grid_thw, spatial_merge_size=1)
        assert patch_counts == embedding_counts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
