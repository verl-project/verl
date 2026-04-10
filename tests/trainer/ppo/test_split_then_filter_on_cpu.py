# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import unittest

import torch
from parameterized import parameterized
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.metric_utils import ZERO_ADV_EPS
from verl.workers.actor.dp_actor import filter_zero_adv_micro_batch


def _make_batch(advantages_1d, seq_len, response_lengths=None):
    """Helper to construct a DataProto batch with explicit per-sample advantage magnitudes.

    Args:
        advantages_1d: 1D list/tensor of per-sample max advantage values.
            Values < ZERO_ADV_EPS are treated as zero-adv by filter_zero_adv_micro_batch.
        seq_len: Sequence length for all tensors.
        response_lengths: Optional list of per-sample response token counts.
            If None, all samples have full response_mask.
    """
    bs = len(advantages_1d)
    advantages = torch.zeros(bs, seq_len)
    for i, a in enumerate(advantages_1d):
        advantages[i, :] = a

    if response_lengths is not None:
        response_mask = torch.zeros(bs, seq_len)
        for i, rl in enumerate(response_lengths):
            response_mask[i, :rl] = 1.0
    else:
        response_mask = torch.ones(bs, seq_len)

    attention_mask = torch.ones(bs, seq_len)

    td = TensorDict(
        {
            "advantages": advantages,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
        batch_size=(bs,),
    )
    return DataProto(batch=td)


class TestFilterZeroAdvMicroBatch(unittest.TestCase):
    """Tests for filter_zero_adv_micro_batch in dp_actor.py."""

    # ------------------------------------------------------------------ #
    #  All nonzero: no filtering, same object returned
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, advs, seq_len, resp_lengths, expected_orig_tok)
            ("uniform_4", [1.0] * 4, 4, None, 16),
            ("varied", [1.0, 2.0, 0.5, 3.0], 8, None, 32),
            ("variable_resp", [1.0, 1.0], 10, [3, 7], 10),
        )
    )
    def test_no_filter_all_nonzero(self, _name, advs, seq_len, resp_lengths, expected_orig_tok):
        batch = _make_batch(advs, seq_len, response_lengths=resp_lengths)
        filtered, num_nz, orig_sz, orig_tok = filter_zero_adv_micro_batch(batch)

        self.assertIs(filtered, batch)
        self.assertEqual(num_nz, len(advs))
        self.assertEqual(orig_sz, len(advs))
        self.assertEqual(orig_tok, expected_orig_tok)

    # ------------------------------------------------------------------ #
    #  Mixed: keeps only nonzero-adv samples
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, advs, seq_len, resp_lengths, expected_kept, expected_orig_tok,
            #  expected_seq_corr, expected_tok_corr)
            ("half_zero", [1.0, 0.0, 1.0, 0.0], 4, None, 2, 16, 0.5, 0.5),
            ("one_zero", [1.0, 1.0, 1.0, 0.0], 4, None, 3, 16, 0.75, 0.75),
            ("one_nonzero", [0.0, 0.0, 0.0, 1.0], 4, None, 1, 16, 0.25, 0.25),
            ("variable_resp", [1.0, 0.0], 10, [8, 2], 1, 10, 0.5, 0.8),
        )
    )
    def test_mixed_keeps_nonzero(
        self, _name, advs, seq_len, resp_lengths, expected_kept, expected_orig_tok, expected_seq_corr, expected_tok_corr
    ):
        batch = _make_batch(advs, seq_len, response_lengths=resp_lengths)
        filtered, num_nz, orig_sz, orig_tok = filter_zero_adv_micro_batch(batch)

        self.assertEqual(len(filtered), expected_kept)
        self.assertEqual(num_nz, expected_kept)
        self.assertEqual(orig_sz, len(advs))
        self.assertEqual(orig_tok, expected_orig_tok)
        # All kept samples have nonzero advantage
        response_mask = filtered.batch["response_mask"]
        max_abs_adv = (filtered.batch["advantages"].abs() * response_mask).max(dim=-1).values
        self.assertTrue((max_abs_adv >= ZERO_ADV_EPS).all())
        # Correction factors
        seq_corr = len(filtered) / orig_sz
        filtered_tok = filtered.batch["response_mask"].sum().item()
        tok_corr = filtered_tok / orig_tok
        self.assertAlmostEqual(seq_corr, expected_seq_corr)
        self.assertAlmostEqual(tok_corr, expected_tok_corr)

    # ------------------------------------------------------------------ #
    #  All zero: keeps 1 shortest sample
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, advs, seq_len, resp_lengths, attn_lengths, expected_orig_tok)
            ("uniform_len", [0.0] * 4, 4, None, [4, 4, 4, 4], 16),
            ("decreasing", [0.0] * 4, 8, None, [8, 6, 4, 2], 32),
            ("increasing", [0.0] * 4, 8, None, [2, 4, 6, 8], 32),
            ("variable_resp", [0.0] * 3, 10, [8, 2, 5], [10, 10, 10], 15),
        )
    )
    def test_all_zero_keeps_shortest(self, _name, advs, seq_len, resp_lengths, attn_lengths, expected_orig_tok):
        batch = _make_batch(advs, seq_len, response_lengths=resp_lengths)
        for i, length in enumerate(attn_lengths):
            batch.batch["attention_mask"][i, :] = 0.0
            batch.batch["attention_mask"][i, :length] = 1.0

        filtered, num_nz, orig_sz, orig_tok = filter_zero_adv_micro_batch(batch)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(num_nz, 0)
        self.assertEqual(orig_sz, len(advs))
        self.assertEqual(orig_tok, expected_orig_tok)
        actual_len = filtered.batch["attention_mask"].sum().item()
        self.assertEqual(actual_len, min(attn_lengths))

    # ------------------------------------------------------------------ #
    #  Eps threshold
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            ("half_eps", ZERO_ADV_EPS * 0.5, 0),
            ("just_below", ZERO_ADV_EPS * 0.99, 0),
            ("at_eps", ZERO_ADV_EPS, 1),
            ("double_eps", ZERO_ADV_EPS * 2.0, 1),
        )
    )
    def test_eps_threshold(self, _name, adv_value, expected_nz_for_first):
        advs = [adv_value, 1.0]  # second sample always nonzero
        batch = _make_batch(advs, 4)
        _, num_nz, _, _ = filter_zero_adv_micro_batch(batch)

        self.assertEqual(num_nz, expected_nz_for_first + 1)  # +1 for the 1.0 sample

    # ------------------------------------------------------------------ #
    #  Meta_info isolation: filtered gets own dict
    # ------------------------------------------------------------------ #

    def test_meta_info_isolation(self):
        batch = _make_batch([1.0, 0.0, 1.0, 0.0], 4)
        batch.meta_info["shared_key"] = "original"

        filtered, _, _, _ = filter_zero_adv_micro_batch(batch)

        # Filtered is a new object.
        self.assertIsNot(filtered, batch)
        # Its own meta_info dict, same content.
        self.assertIsNot(filtered.meta_info, batch.meta_info)
        self.assertEqual(filtered.meta_info["shared_key"], "original")

    def test_meta_info_not_copied_when_no_filter(self):
        """When all nonzero, same object is returned."""
        batch = _make_batch([1.0, 1.0], 4)
        batch.meta_info["shared_key"] = "original"
        filtered, _, _, _ = filter_zero_adv_micro_batch(batch)
        self.assertIs(filtered, batch)
        self.assertEqual(filtered.meta_info["shared_key"], "original")


if __name__ == "__main__":
    unittest.main()
