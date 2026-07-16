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
"""The vision placeholders are masked out of the rollout logits, so the policy cannot sample them.

A sampled <|image_pad|> has no image behind it, and every consumer of the sequence assumes it does.
The out-of-vocabulary tail was already masked in compute_logits; the placeholders sit *inside* the
vocabulary, so nothing stopped the policy from picking one.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from verl.workers.rollout.utils import get_vision_placeholder_token_ids
from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

TOKENS = {151655: "<|image_pad|>", 151656: "<|video_pad|>"}


def make_processor(**attrs):
    tokenizer = SimpleNamespace(convert_ids_to_tokens=TOKENS.get)
    return SimpleNamespace(tokenizer=tokenizer, **attrs)


class TestGetVisionPlaceholderTokenIds:
    """Which ids to ban, resolved from the processor."""

    def test_resolves_both_placeholders_from_token_ids(self):
        processor = make_processor(image_token_id=151655, video_token_id=151656)

        assert get_vision_placeholder_token_ids(processor) == [151655, 151656]

    def test_resolves_placeholders_from_token_strings(self):
        """Newer processors expose image_token/video_token strings instead of the ids."""
        processor = make_processor(image_token="<|image_pad|>", video_token="<|video_pad|>")
        processor.tokenizer.convert_tokens_to_ids = {v: k for k, v in TOKENS.items()}.get

        assert get_vision_placeholder_token_ids(processor) == [151655, 151656]

    def test_image_only_processor_bans_only_the_image_placeholder(self):
        processor = make_processor(image_token_id=151655)

        assert get_vision_placeholder_token_ids(processor) == [151655]

    def test_text_only_model_leaves_sampling_untouched(self):
        assert get_vision_placeholder_token_ids(None) == []


VOCAB_SIZE = 6  # ids 6 and 7 are the padded, out-of-vocabulary tail
PADDED_WIDTH = 8
UNMASKED = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * 2)


class FakeModel:
    """Stands in for the vLLM model: compute_logits hands back fresh logits on every call."""

    def compute_logits(self, *args, **kwargs) -> torch.Tensor:
        return torch.arange(PADDED_WIDTH, dtype=torch.float32).repeat(2, 1)


class TestMonkeyPatchComputeLogits:
    """Those ids, masked where the sampler will see them."""

    def test_banned_tokens_and_oov_tail_are_masked(self):
        model = FakeModel()

        monkey_patch_compute_logits(model, VOCAB_SIZE, banned_token_ids=[2, 4])
        logits = model.compute_logits()

        assert (logits[:, [2, 4]] == float("-inf")).all()
        assert (logits[:, VOCAB_SIZE:] == float("-inf")).all()
        # every token that is still legal to sample keeps the value the model produced
        assert torch.equal(logits[:, [0, 1, 3, 5]], torch.tensor([[0.0, 1.0, 3.0, 5.0]] * 2))

    def test_without_banned_tokens_only_the_oov_tail_is_masked(self):
        """A text-only model passes no ids, and must sample exactly as it did before."""
        model = FakeModel()

        monkey_patch_compute_logits(model, VOCAB_SIZE)
        logits = model.compute_logits()

        assert torch.equal(logits[:, :VOCAB_SIZE], UNMASKED)
        assert (logits[:, VOCAB_SIZE:] == float("-inf")).all()

    def test_empty_banned_token_ids_behaves_like_none(self):
        model = FakeModel()

        monkey_patch_compute_logits(model, VOCAB_SIZE, banned_token_ids=[])
        logits = model.compute_logits()

        assert torch.equal(logits[:, :VOCAB_SIZE], UNMASKED)
