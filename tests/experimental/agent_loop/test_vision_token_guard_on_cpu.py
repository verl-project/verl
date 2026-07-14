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
"""A vision placeholder token the policy sampled itself must not reach get_rope_index.

mm_token_type_ids is built over prompt + response, so a token the model generated is marked as a
modality span like any other -- but there is no grid behind it. get_rope_index takes one grid per
span, so the extra span either exhausts the iterator (StopIteration) or indexes a modality that
was never populated (TypeError: 'NoneType' object is not an iterator), and the training job dies.
"""

from __future__ import annotations

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

drop = AgentLoopWorker._drop_ungrounded_vision_tokens

TEXT, IMAGE, VIDEO = 0, 1, 2


def grids(count: int) -> torch.Tensor:
    """`count` grids, shaped like image_grid_thw / video_grid_thw: (n, 3)."""
    return torch.tensor([[1, 2, 2]] * count, dtype=torch.long)


def test_generated_video_token_is_demoted_to_text():
    # a real image in the prompt, and a <|video_pad|> the policy emitted in its response
    mm = torch.tensor([[IMAGE, IMAGE, TEXT, TEXT, VIDEO]])

    drop(mm, image_grid_thw=grids(1), video_grid_thw=None)

    # the ungrounded video span is gone; the grounded image span is untouched
    assert torch.equal(mm, torch.tensor([[IMAGE, IMAGE, TEXT, TEXT, TEXT]]))


def test_generated_image_token_beyond_the_grids_is_demoted_to_text():
    # the prompt's image is grounded; the second span is one the policy emitted
    mm = torch.tensor([[IMAGE, IMAGE, TEXT, IMAGE]])

    drop(mm, image_grid_thw=grids(1), video_grid_thw=None)

    assert torch.equal(mm, torch.tensor([[IMAGE, IMAGE, TEXT, TEXT]]))


def test_grounded_spans_survive():
    """A multi-turn image tool injects image tokens into the response -- those have grids."""
    mm = torch.tensor([[IMAGE, IMAGE, TEXT, IMAGE, IMAGE, TEXT]])
    before = mm.clone()

    drop(mm, image_grid_thw=grids(2), video_grid_thw=None)

    assert torch.equal(mm, before)


def test_text_only_sequence_is_untouched():
    mm = torch.zeros((1, 6), dtype=torch.long)
    before = mm.clone()

    drop(mm, image_grid_thw=None, video_grid_thw=None)

    assert torch.equal(mm, before)


def test_video_spans_are_kept_when_video_grids_exist():
    mm = torch.tensor([[VIDEO, VIDEO, TEXT, VIDEO]])
    before = mm.clone()

    drop(mm, image_grid_thw=None, video_grid_thw=grids(2))

    assert torch.equal(mm, before)


def test_only_the_spans_past_the_grid_count_are_dropped():
    """Grids pair with spans in order, so the first N survive and the rest become text."""
    mm = torch.tensor([[IMAGE, TEXT, IMAGE, TEXT, IMAGE]])

    drop(mm, image_grid_thw=grids(2), video_grid_thw=None)

    assert torch.equal(mm, torch.tensor([[IMAGE, TEXT, IMAGE, TEXT, TEXT]]))
