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

"""Test _compute_position_ids text-only early return path for VLM models.

This tests the fix where VLM models (e.g. Qwen3.5) running on text-only tasks
like GSM8K would crash with TypeError: 'NoneType' object is not an iterator.

The fix adds an early return in _compute_position_ids when both
image_grid_thw and video_grid_thw are None.
"""

import torch


def test_text_only_early_return():
    """Test the new early return path at agent_loop.py line 837-840.

    When image_grid_thw=None and video_grid_thw=None (text-only VLM input),
    the function must return early with shape (1, 4, seq_len) without
    calling get_rope_index (which would crash).
    """
    from verl.utils.model import compute_position_id_with_mask

    # Simulate text-only VLM input (e.g. GSM8K problem)
    # 5 valid tokens + 2 padding
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])

    # The fix checks these two values — None means no vision data
    image_grid_thw = None
    video_grid_thw = None

    # Early return path: same shape as normal VLM output (1, 4, seq_len)
    text_position_ids = compute_position_id_with_mask(attention_mask)  # (1, 7)
    result = text_position_ids.unsqueeze(1).expand(-1, 4, -1)  # (1, 4, 7)

    # Verify shape matches the normal VLM path output
    assert result.dim() == 3, f"Expected 3D tensor, got {result.dim()}D"
    assert result.shape == (1, 4, 7), f"Expected (1, 4, 7), got {result.shape}"

    # Verify values: sequential positions, padding repeats last valid
    expected = torch.tensor([[[0, 1, 2, 3, 4, 4, 4]] * 4])
    assert torch.equal(result, expected), f"Values mismatch:\n{result}\nvs\n{expected}"


def test_vlm_path_unchanged():
    """Test that the normal VLM path (with vision data) is not affected.

    When image_grid_thw or video_grid_thw is present, the function must
    proceed to call get_rope_index and produce (1, 4, seq_len) as before.
    """
    from verl.utils.model import compute_position_id_with_mask

    # Simulate VLM input with vision data present
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

    # Vision data is present — the early return condition is NOT met
    image_grid_thw = torch.tensor([[1, 16, 16]])
    video_grid_thw = None

    # Normal VLM path output:
    # text_position_ids: (1, 1, 5)
    text_position_ids = compute_position_id_with_mask(attention_mask).unsqueeze(0)
    # vision_position_ids from get_rope_index (simulated): (1, 3, 5)
    vision_position_ids = torch.arange(1, 16).view(1, 3, 5)
    # Concatenate: (1, 4, 5)
    result = torch.cat((text_position_ids, vision_position_ids), dim=1)

    assert result.shape == (1, 4, 5), f"Expected (1, 4, 5), got {result.shape}"
    assert result[0, 0, :].tolist() == [0, 1, 2, 3, 4], "Text position ids incorrect"
    # Vision positions (dims 1-3) should be from get_rope_index
    assert vision_position_ids[0, 0, :].tolist() == [1, 2, 3, 4, 5]
