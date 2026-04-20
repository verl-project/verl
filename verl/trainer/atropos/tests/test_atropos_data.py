"""Tests for scored_data_to_dataproto conversion."""

import pytest
from recipe.atropos.atropos_data import _find_response_start, scored_data_to_dataproto


def _make_scored_data(
    prompt_tokens: list[int],
    response_tokens: list[int],
    score: float,
    advantages: list[float] | None = None,
    overrides: dict | None = None,
    group_size: int = 1,
):
    """Build a single ScoredData dict with one group of sequences."""
    tokens_list = []
    masks_list = []
    scores_list = []
    adv_list = [] if advantages is not None else None

    for _ in range(group_size):
        full_tokens = prompt_tokens + response_tokens
        # mask: -100 for prompt, actual tokens for response
        mask = [-100] * len(prompt_tokens) + response_tokens
        tokens_list.append(full_tokens)
        masks_list.append(mask)
        scores_list.append(score)
        if adv_list is not None:
            # full-sequence advantages (prompt portion + response portion)
            full_adv = [0.0] * len(prompt_tokens) + advantages
            adv_list.append(full_adv)

    result = {"tokens": tokens_list, "masks": masks_list, "scores": scores_list}
    if adv_list is not None:
        result["advantages"] = adv_list
    if overrides is not None:
        result["overrides"] = [overrides] * group_size
    return result


class TestFindResponseStart:
    def test_basic(self):
        mask = [-100, -100, -100, 5, 6, 7]
        assert _find_response_start(mask) == 3

    def test_all_prompt(self):
        mask = [-100, -100, -100]
        assert _find_response_start(mask) == 3

    def test_all_response(self):
        mask = [1, 2, 3]
        assert _find_response_start(mask) == 0

    def test_empty(self):
        assert _find_response_start([]) == 0


class TestScoredDataToDataproto:
    def test_basic_conversion(self):
        """Single group, single sequence — verify tensor shapes and values."""
        scored_data = _make_scored_data(
            prompt_tokens=[10, 20, 30],
            response_tokens=[40, 50],
            score=1.0,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        assert dp.batch["input_ids"].shape == (1, 8)  # 4 prompt + 4 response
        assert dp.batch["prompts"].shape == (1, 4)
        assert dp.batch["responses"].shape == (1, 4)
        assert dp.batch["attention_mask"].shape == (1, 8)
        assert dp.batch["token_level_rewards"].shape == (1, 4)

        # prompt should be left-padded: [0, 10, 20, 30]
        assert dp.batch["prompts"][0].tolist() == [0, 10, 20, 30]
        # response should be right-padded: [40, 50, 0, 0]
        assert dp.batch["responses"][0].tolist() == [40, 50, 0, 0]

        # attention mask: 0 for pad, 1 for real tokens
        assert dp.batch["attention_mask"][0].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]

        # reward at last valid response position (index 1)
        assert dp.batch["token_level_rewards"][0][1].item() == 1.0
        assert dp.batch["token_level_rewards"][0][0].item() == 0.0

    def test_grpo_grouping(self):
        """Multiple sequences in one group should share the same UID."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2],
            response_tokens=[3, 4],
            score=0.5,
            group_size=4,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        assert dp.batch["input_ids"].shape[0] == 4
        uids = dp.non_tensor_batch["uid"]
        # all 4 sequences in the group should share a UID
        assert len(set(uids)) == 1

    def test_multiple_groups(self):
        """Two groups should get different UIDs."""
        sd1 = _make_scored_data(prompt_tokens=[1, 2], response_tokens=[3], score=1.0, group_size=2)
        sd2 = _make_scored_data(prompt_tokens=[4, 5], response_tokens=[6], score=-1.0, group_size=2)
        dp = scored_data_to_dataproto([sd1, sd2], max_prompt_length=4, max_response_length=4)

        assert dp.batch["input_ids"].shape[0] == 4
        uids = dp.non_tensor_batch["uid"]
        # first 2 share one UID, last 2 share another
        assert uids[0] == uids[1]
        assert uids[2] == uids[3]
        assert uids[0] != uids[2]

    def test_prompt_truncation(self):
        """Long prompts should be truncated from the left (keep last tokens)."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            response_tokens=[7],
            score=0.0,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=3, max_response_length=2)

        # should keep last 3 prompt tokens: [4, 5, 6]
        assert dp.batch["prompts"][0].tolist() == [4, 5, 6]

    def test_response_truncation(self):
        """Long responses should be truncated from the right."""
        scored_data = _make_scored_data(
            prompt_tokens=[1],
            response_tokens=[2, 3, 4, 5, 6],
            score=0.0,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=2, max_response_length=3)

        # should keep first 3 response tokens: [2, 3, 4]
        assert dp.batch["responses"][0].tolist() == [2, 3, 4]

    def test_token_level_advantages(self):
        """When advantages are provided, they should appear in the output."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2],
            response_tokens=[3, 4, 5],
            score=1.0,
            advantages=[0.1, 0.2, 0.3],
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        assert "token_level_advantages" in dp.batch
        adv = dp.batch["token_level_advantages"][0].tolist()
        # advantages should be response-aligned, right-padded with 0
        assert adv == pytest.approx([0.1, 0.2, 0.3, 0.0])

    def test_no_advantages(self):
        """When no advantages are provided, the field should be absent."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2],
            response_tokens=[3, 4],
            score=1.0,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        assert "token_level_advantages" not in dp.batch

    def test_set_advantage_to_zero_override(self):
        """Override should zero both score and token-level advantages."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2],
            response_tokens=[3, 4],
            score=1.0,
            advantages=[0.5, 0.5],
            overrides={"set_advantage_to_zero": True},
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        # score should be zeroed
        assert dp.batch["token_level_rewards"][0].sum().item() == 0.0
        # advantages should be zeroed
        assert dp.batch["token_level_advantages"][0].sum().item() == 0.0

    def test_empty_response(self):
        """Sequence with all-prompt tokens should produce zero reward."""
        scored_data = {
            "tokens": [[1, 2, 3]],
            "masks": [[-100, -100, -100]],
            "scores": [1.0],
        }
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        # reward should be zero (no valid response position to place it)
        assert dp.batch["token_level_rewards"][0].sum().item() == 0.0
        # response should be all padding
        assert dp.batch["responses"][0].tolist() == [0, 0, 0, 0]

    def test_empty_input_raises(self):
        """Empty scored_data_list should raise ValueError."""
        with pytest.raises(ValueError, match="No data to convert"):
            scored_data_to_dataproto([], max_prompt_length=4, max_response_length=4)

    def test_position_ids(self):
        """Position IDs should be present and correctly computed."""
        scored_data = _make_scored_data(
            prompt_tokens=[1, 2],
            response_tokens=[3, 4],
            score=0.0,
        )
        dp = scored_data_to_dataproto([scored_data], max_prompt_length=4, max_response_length=4)

        assert "position_ids" in dp.batch
        assert dp.batch["position_ids"].shape == dp.batch["input_ids"].shape
