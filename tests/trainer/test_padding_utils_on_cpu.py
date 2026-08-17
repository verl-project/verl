# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""The synthetic padding row must survive the THD context-parallel split.

``upsample_batch_to_divisible_size`` appends no-op samples when the batch size is not
divisible by ``dp_size``, which happens as soon as the number of trajectories per prompt
varies. Those rows used to be 2 tokens long, which is below what the CP split can handle:
``preprocess_packed_seqs`` pads each row to ``align_size = tp * cp * 2`` and then hands CP
rank *r* the slice ``d[half * r : half * (r + 1)]``, where ``half`` comes from the *padded*
length while ``d`` holds only the *valid* tokens. For a 2-token row at tp=cp=2 the row pads
to 8, ``half`` is 2, and rank 1 asks for ``d[2:4]`` of a 2-element tensor -- empty, which
raises on assignment.

verl's own ``preprocess_packed_seqs`` clamps that slice (#6001); the vendored copies in
mbridge and Megatron-Bridge do not. These tests pin the property that makes the row safe
regardless of which copy runs: its length is a multiple of the alignment for any realistic
topology, so it never needs alignment padding and the slice is always in range.
"""

import pytest
import torch

from verl.trainer.ppo.padding_utils import (
    _PADDING_TOKENS_PER_SIDE,
    construct_minimal_padding_template,
)


def _source_sample() -> tuple[dict, dict]:
    """Stand-in for one real sample fetched from TransferQueue."""
    sample = {
        "prompts": torch.zeros(4, dtype=torch.int64),
        "responses": torch.zeros(4, dtype=torch.int64),
        "input_ids": torch.zeros(8, dtype=torch.int64),
        "attention_mask": torch.ones(8, dtype=torch.int64),
        "position_ids": torch.arange(8, dtype=torch.int64),
        "response_mask": torch.ones(4, dtype=torch.int64),
        "uid": "real",
    }
    return sample, {"prompt_len": 4, "response_len": 4, "seq_len": 8}


def _first_cp_chunk(valid_len: int, *, tp: int, cp: int, cp_rank: int, clamped: bool):
    """The first-chunk assignment from ``preprocess_packed_seqs``, both variants.

    ``clamped=True`` is what verl's own copy does; ``clamped=False`` is what the vendored
    bridge copies still do.
    """
    align_size = tp * cp * 2
    padded = valid_len + (align_size - valid_len % align_size) % align_size
    half = (padded // cp) // 2
    out = torch.zeros(padded // cp, dtype=torch.int64)
    d = torch.arange(valid_len, dtype=torch.int64)
    first_start, first_end = half * cp_rank, half * (cp_rank + 1)
    if clamped:
        first_end = min(first_end, d.shape[0])
        length = max(first_end - first_start, 0)
        if length > 0:
            out[0:length] = d[first_start:first_end]
    else:
        out[0:half] = d[first_start:first_end]
    return out


@pytest.mark.parametrize("tp_times_cp", [1, 2, 4, 8, 16, 32, 64])
def test_padding_row_needs_no_alignment_padding_for_any_topology(tp_times_cp):
    """A row already a multiple of ``align_size`` is never re-padded, so every CP chunk
    index stays inside the valid tokens -- with or without the clamp."""
    sample, _ = construct_minimal_padding_template(*_source_sample(), eos_token_id=7)
    valid_len = int(sample["attention_mask"].sum())
    align_size = tp_times_cp * 2
    assert valid_len % align_size == 0, f"{valid_len} tokens is not a multiple of {align_size}"


def test_padding_row_still_contributes_no_gradient_and_no_reward():
    """Making the row longer must not make it count."""
    sample, tag = construct_minimal_padding_template(*_source_sample(), eos_token_id=7)
    assert sample["response_mask"].sum() == 0
    assert sample["loss_mask"].sum() == 0
    assert sample["rm_scores"].sum() == 0
    assert sample["num_turns"] == 0
    assert tag["is_padding"] is True
    # The tag must describe the real shape: metrics and the seqlen balancer read it.
    assert tag["prompt_len"] == _PADDING_TOKENS_PER_SIDE
    assert tag["response_len"] == _PADDING_TOKENS_PER_SIDE
    assert tag["seq_len"] == 2 * _PADDING_TOKENS_PER_SIDE
    assert int(sample["attention_mask"].sum()) == 2 * _PADDING_TOKENS_PER_SIDE


def test_padding_row_shapes_stay_internally_consistent():
    sample, _ = construct_minimal_padding_template(*_source_sample(), eos_token_id=7)
    n = _PADDING_TOKENS_PER_SIDE
    assert sample["prompts"].shape == (n,)
    assert sample["responses"].shape == (n,)
    assert sample["input_ids"].shape == (2 * n,)
    assert sample["attention_mask"].shape == (2 * n,)
    assert sample["position_ids"].shape[-1] == 2 * n


def test_a_two_token_row_breaks_the_unclamped_cp_split():
    """Why the constant exists: the exact failure, at tp=cp=2 on CP rank 1."""
    with pytest.raises(RuntimeError, match=r"existing size \(0\)"):
        _first_cp_chunk(2, tp=2, cp=2, cp_rank=1, clamped=False)
    # verl's own copy survives it; the vendored bridge copies are what still raise.
    _first_cp_chunk(2, tp=2, cp=2, cp_rank=1, clamped=True)


def test_the_padding_row_length_survives_the_unclamped_split():
    """What the constant buys: no exception even on the unclamped path."""
    valid_len = 2 * _PADDING_TOKENS_PER_SIDE
    for cp_rank in range(2):
        _first_cp_chunk(valid_len, tp=2, cp=2, cp_rank=cp_rank, clamped=False)


@pytest.mark.parametrize("valid_len", [128, 4096, 14000])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_clamping_is_a_pure_guard_for_long_rows(valid_len, cp_rank):
    """Rows long enough to be unaffected must be bit-identical either way, so the guard
    cannot change training behaviour for real sequences."""
    unclamped = _first_cp_chunk(valid_len, tp=2, cp=2, cp_rank=cp_rank, clamped=False)
    clamped = _first_cp_chunk(valid_len, tp=2, cp=2, cp_rank=cp_rank, clamped=True)
    assert torch.equal(unclamped, clamped)
