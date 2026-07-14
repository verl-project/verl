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
"""compute_logits masks the tokens the sampler must never pick.

The out-of-vocabulary tail was already masked. The vision placeholders sit *inside* the vocabulary,
so nothing stopped the policy from sampling one -- and a sampled <|image_pad|> has no image behind
it, which breaks every consumer that pairs placeholders with grids.
"""

from __future__ import annotations

import torch

from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

VOCAB_SIZE = 6  # ids 6 and 7 are the padded, out-of-vocabulary tail
PADDED_WIDTH = 8


class FakeModel:
    """Stands in for the vLLM model: compute_logits hands back fresh logits on every call."""

    def compute_logits(self, *args, **kwargs) -> torch.Tensor:
        return torch.arange(PADDED_WIDTH, dtype=torch.float32).repeat(2, 1)


def test_banned_tokens_and_oov_tail_are_masked():
    model = FakeModel()

    monkey_patch_compute_logits(model, VOCAB_SIZE, banned_token_ids=[2, 4])
    logits = model.compute_logits()

    assert (logits[:, [2, 4]] == float("-inf")).all()
    assert (logits[:, VOCAB_SIZE:] == float("-inf")).all()
    # every token that is still legal to sample keeps the value the model produced
    assert torch.equal(logits[:, [0, 1, 3, 5]], torch.tensor([[0.0, 1.0, 3.0, 5.0]] * 2))


def test_without_banned_tokens_only_the_oov_tail_is_masked():
    """A text-only model passes no ids, and must sample exactly as it did before."""
    model = FakeModel()

    monkey_patch_compute_logits(model, VOCAB_SIZE)
    logits = model.compute_logits()

    assert torch.equal(logits[:, :VOCAB_SIZE], torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * 2))
    assert (logits[:, VOCAB_SIZE:] == float("-inf")).all()


def test_empty_banned_token_ids_behaves_like_none():
    model = FakeModel()

    monkey_patch_compute_logits(model, VOCAB_SIZE, banned_token_ids=[])
    logits = model.compute_logits()

    assert torch.equal(logits[:, :VOCAB_SIZE], torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * 2))
