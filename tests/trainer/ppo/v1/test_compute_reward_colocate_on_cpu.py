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

"""CPU-only unit tests for the colocated reward model data plumbing in the V1 PPO trainer.

These tests cover the dependency-light pieces of ``PPOTrainer._compute_reward_colocate``:

- ``_lengths_to_mask``: building right-padded masks from per-row valid lengths.
- The attention-mask layout contract expected by ``RewardManagerBase.assemble_rm_scores``
  (i.e. ``attention_mask[:, prompt_width:].sum(dim=1)`` must equal the valid response
  lengths, with right-padded prompts/responses).
- Constructing a 1-D object array for ``raw_prompt`` that does not collapse equal-length
  chat-message lists into a 2-D numpy array.

The full path (TransferQueue + RewardLoopManager + GPU rollout) is exercised by the
GPU integration test ``tests/experimental/reward_loop/test_agent_reward_loop_colocate.py``.
"""

import numpy as np
import torch


def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    """Standalone copy of ``PPOTrainer._lengths_to_mask`` for unit testing.

    Kept in sync with ``verl/trainer/ppo/v1/trainer_base.py``; importing the trainer
    directly would pull in heavy runtime deps (ray, transfer_queue, vllm).
    """
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(torch.int64)


def _assemble_rm_scores(prompts, attention_mask, responses, scores):
    """Replica of ``RewardManagerBase.assemble_rm_scores`` to assert the contract."""
    prompt_length = prompts.size(1)
    valid_response_length = attention_mask[:, prompt_length:].sum(dim=1)
    rm_scores = torch.zeros_like(responses, dtype=torch.float32)
    rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = rm_scores.new_tensor(scores)
    return rm_scores


class TestLengthsToMask:
    def test_basic_right_padding(self):
        lengths = torch.tensor([1, 3, 2])
        mask = _lengths_to_mask(lengths, width=4)
        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        assert torch.equal(mask, expected)

    def test_full_and_empty_rows(self):
        lengths = torch.tensor([0, 4])
        mask = _lengths_to_mask(lengths, width=4)
        expected = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int64)
        assert torch.equal(mask, expected)

    def test_row_count_matches_lengths(self):
        lengths = torch.tensor([2, 2, 2, 2, 2])
        mask = _lengths_to_mask(lengths, width=3)
        assert mask.shape == (5, 3)


class TestAttentionMaskContract:
    """The mask built by ``_compute_reward_colocate`` must satisfy assemble_rm_scores."""

    def test_valid_response_length_recovered(self):
        prompt_lengths = torch.tensor([2, 1, 3])
        response_lengths = torch.tensor([3, 2, 1])
        prompt_width, response_width = 4, 4

        prompt_mask = _lengths_to_mask(prompt_lengths, prompt_width)
        response_mask = _lengths_to_mask(response_lengths, response_width)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # assemble_rm_scores slices off the prompt portion and sums the rest.
        recovered = attention_mask[:, prompt_width:].sum(dim=1)
        assert torch.equal(recovered, response_lengths)

    def test_score_lands_on_last_valid_response_token(self):
        prompt_lengths = torch.tensor([2, 1])
        response_lengths = torch.tensor([3, 2])
        prompt_width, response_width = 4, 5

        prompts = torch.zeros((2, prompt_width), dtype=torch.int64)
        responses = torch.zeros((2, response_width), dtype=torch.int64)
        prompt_mask = _lengths_to_mask(prompt_lengths, prompt_width)
        response_mask = _lengths_to_mask(response_lengths, response_width)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        scores = [0.5, -1.0]
        rm_scores = _assemble_rm_scores(prompts, attention_mask, responses, scores)

        # The score must be placed at index (response_length - 1) and nowhere else.
        assert rm_scores[0, 2].item() == 0.5
        assert rm_scores[1, 1].item() == -1.0
        assert rm_scores.sum().item() == (0.5 - 1.0)
        assert torch.equal(rm_scores.sum(dim=1), torch.tensor(scores))


class TestRmScoresJaggedWriteBack:
    """rm_scores must be written back as per-sample jagged rows (response-aligned).

    The streaming reward path stores rm_scores as a jagged tensor whose per-row length
    equals each sample's response length; ``_compute_advantage`` then calls
    ``to_padded_tensor()`` on it. ``_compute_reward_colocate`` must produce the same layout
    from the (bsz, response_width) tensor returned by ``assemble_rm_scores``.
    """

    @staticmethod
    def _to_jagged(padded_rm_scores, response_lengths):
        return torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(response_lengths))],
            layout=torch.jagged,
        )

    def test_jagged_layout_and_lengths(self):
        padded = torch.zeros(3, 5)
        padded[0, 2] = 0.5
        padded[1, 1] = -1.0
        padded[2, 0] = 0.9
        response_lengths = torch.tensor([3, 2, 1])

        nt = self._to_jagged(padded, response_lengths)

        # get_tensordict requires nested + jagged + contiguous.
        assert nt.is_nested
        assert nt.layout == torch.jagged
        assert nt.is_contiguous()
        assert nt.offsets().diff().tolist() == [3, 2, 1]

    def test_padded_roundtrip_preserves_scores(self):
        padded = torch.zeros(2, 6)
        padded[0, 3] = 1.25
        padded[1, 0] = -2.0
        response_lengths = torch.tensor([4, 1])

        nt = self._to_jagged(padded, response_lengths)
        pad_back = nt.to_padded_tensor(0.0)

        # Width collapses to max valid response length, scores preserved per row.
        assert pad_back.shape == (2, 4)
        assert torch.allclose(pad_back.sum(dim=1), torch.tensor([1.25, -2.0]))


class TestRawPromptObjectArray:
    """raw_prompt must remain a 1-D object array (one chat-message list per sample)."""

    @staticmethod
    def _build(raw_prompts):
        # Mirror the production normalization in _compute_reward_colocate: the TransferQueue
        # field may be a tensordict LinkedList (list subclass), NonTensorStack or numpy array,
        # so it is wrapped with list(...) (not .tolist(), which only exists on numpy/tensors).
        raw_prompts = list(raw_prompts)
        arr = np.empty(len(raw_prompts), dtype=object)
        arr[:] = raw_prompts
        return arr

    def test_equal_length_messages_not_collapsed(self):
        # Two samples, each with a single-message chat of equal structure.
        raw_prompts = [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ]
        arr = self._build(raw_prompts)
        assert arr.shape == (2,)
        assert arr.dtype == object
        # Each element is exactly one sample's message list.
        assert list(arr[0]) == [{"role": "user", "content": "a"}]
        assert list(arr[1]) == [{"role": "user", "content": "b"}]

    def test_naive_np_array_would_collapse(self):
        # Documents why element-wise assignment is required: np.array collapses
        # equal-length nested lists into a 2-D array, breaking per-sample indexing.
        raw_prompts = [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ]
        naive = np.array(raw_prompts, dtype=object)
        assert naive.shape == (2, 1)  # collapsed - wrong for downstream chunk/index

        safe = self._build(raw_prompts)
        assert safe.shape == (2,)  # preserved - correct

    def test_chunk_and_index_roundtrip(self):
        raw_prompts = [[{"role": "user", "content": str(i)}] for i in range(4)]
        arr = self._build(raw_prompts)
        # Simulate DataProto.chunk -> np.array_split along axis 0, then per-row index.
        chunks = np.array_split(arr, 2)
        assert len(chunks) == 2
        first_item = chunks[0][0]
        assert list(first_item) == [{"role": "user", "content": "0"}]

    def test_list_subclass_input_like_tensordict_linkedlist(self):
        # Regression: TransferQueue may return raw_prompt as a tensordict LinkedList,
        # which is a `list` subclass without `.tolist()`. The production code uses
        # list(...) to normalize it; verify that path yields a correct 1-D object array.
        class _FakeLinkedList(list):
            """Stand-in for tensordict.utils.LinkedList (a list subclass)."""

        raw_prompts = _FakeLinkedList(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        )
        assert not hasattr(raw_prompts, "tolist")  # the exact cause of the original crash
        arr = self._build(raw_prompts)
        assert arr.shape == (2,)
        assert list(arr[0]) == [{"role": "user", "content": "a"}]
        assert list(arr[1]) == [{"role": "user", "content": "b"}]
