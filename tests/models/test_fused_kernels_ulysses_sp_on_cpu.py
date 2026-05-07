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
"""Regression test for issue #6068.

When `use_fused_kernels=True` is combined with `ulysses_sequence_parallel_size > 1`,
the fused-forward functions in `verl/models/transformers/*.py` were computing
`rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)` *after* Ulysses had
already SP-sliced the input. `torch.roll` wraps around the local-shard boundary
rather than the global sequence, so the last position on every SP rank ended up
predicting the wrong label (the first token of the rank's local shard instead of
the global next-token). This biased ~1 position per rank per micro-batch and
manifested as a slow training-quality regression at SP > 1 (issue #6068).

The non-fused path was always correct because the FSDP engine pre-computes a
globally-rolled `input_ids_rmpad_rolled`, then SP-slices *that* (see
`verl/workers/engine/fsdp/transformer_impl.py:951-984`), and feeds it to
`logprobs_from_logits` directly.

The fix plumbs the engine's pre-rolled `input_ids_rmpad_rolled` into the fused
forward functions via a new `shift_labels` kwarg, mirroring what the veomni
engine already does (see `verl/workers/engine/veomni/transformer_impl.py:659`).

These tests run on CPU and do not require any GPUs.
"""

import torch

from verl.models.transformers import dense_common, qwen2_vl, qwen3_5, qwen3_vl


def _global_then_slice(input_ids: torch.Tensor, sp_size: int) -> list[torch.Tensor]:
    """The correct behavior: roll on the full sequence, then SP-slice.

    Mirrors the engine's `input_ids_rmpad_rolled` -> `ulysses_pad_and_slice_inputs`
    pipeline in `verl/workers/engine/fsdp/transformer_impl.py`.
    """
    rolled = torch.roll(input_ids, shifts=-1, dims=-1)
    return list(torch.chunk(rolled, sp_size, dim=-1))


def _slice_then_local_roll(input_ids: torch.Tensor, sp_size: int) -> list[torch.Tensor]:
    """The buggy behavior: SP-slice first, then roll on the local shard.

    Reproduces what `torch.roll(input_ids, shifts=-1, dims=-1)` did inside the
    fused-forward functions when SP > 1.
    """
    sliced = list(torch.chunk(input_ids, sp_size, dim=-1))
    return [torch.roll(s, shifts=-1, dims=-1) for s in sliced]


def test_local_roll_diverges_from_global_roll_under_sp():
    """Demonstrates the root cause of #6068.

    For SP > 1, slice-then-local-roll produces different labels than
    global-roll-then-slice at exactly the shard-boundary position on every rank.
    """
    torch.manual_seed(0)
    total_nnz, sp_size = 32, 4
    input_ids = torch.randint(0, 10000, (1, total_nnz))

    correct_shards = _global_then_slice(input_ids, sp_size)
    buggy_shards = _slice_then_local_roll(input_ids, sp_size)

    # Assertion 1: every interior position matches.
    for correct, buggy in zip(correct_shards, buggy_shards, strict=True):
        torch.testing.assert_close(correct[..., :-1], buggy[..., :-1])

    # Assertion 2: the last position of every shard is wrong under buggy.
    # On every rank the buggy version wraps to its local shard's first token;
    # the correct version uses the next shard's first token (or the global
    # first token on the final rank).
    for rank in range(sp_size):
        correct_last = correct_shards[rank][..., -1]
        buggy_last = buggy_shards[rank][..., -1]
        assert not torch.equal(correct_last, buggy_last), (
            f"rank {rank}: expected divergence at shard boundary but got {correct_last}=={buggy_last}"
        )


def _make_fake_lm(hidden_size: int, vocab_size: int):
    """Minimal stand-in for the language-model wrapper expected by the fused forwards.

    The fused forward functions only touch `self.model(...)` (returns hidden states)
    and `self.lm_head.weight`, so we don't need a real transformer here.
    """

    class FakeBaseModel(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden)

        def forward(self, input_ids, **kwargs):
            # Return a tuple so `outputs[0]` works (mirrors HF convention).
            hidden_states = self.embed(input_ids).to(torch.float32)
            return (hidden_states,)

    class FakeLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel(hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    torch.manual_seed(42)
    return FakeLM()


def _run_fused_forward(forward_fn, model, input_ids, shift_labels):
    """Invoke a fused forward, passing both `input_ids` and the new `shift_labels`."""
    return forward_fn(
        model,
        input_ids=input_ids,
        labels=None,
        temperature=1.0,
        shift_labels=shift_labels,
        return_dict=True,
    )


def _assert_fused_forward_uses_shift_labels(forward_fn):
    """Contract: when `shift_labels` is provided, the fused forward must use it
    verbatim (i.e. not re-roll). We verify by feeding `shift_labels` that point
    to a deliberately wrong vocab id and checking the resulting log-prob
    matches what the kernel would produce for that wrong id, not for the
    locally-rolled id.
    """
    hidden_size, vocab_size = 8, 64
    seq_len = 4
    model = _make_fake_lm(hidden_size, vocab_size)

    # Pick input_ids and a `shift_labels` that disagrees with torch.roll(input_ids).
    # If the fix is in place, log_probs target `shift_labels`. If not, they target
    # torch.roll(input_ids), which differs on the last position.
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    shift_labels = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)  # != torch.roll(input_ids)

    out_with_shift = _run_fused_forward(forward_fn, model, input_ids, shift_labels)

    # Recompute log_probs "by hand" using the same fused kernel against the
    # explicit labels to nail down the expected value.
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    with torch.no_grad():
        hidden = model.model(input_ids)[0]
        ref_log_probs, _ = FusedLinearForPPO().forward(
            hidden_states=hidden,
            vocab_weights=model.lm_head.weight,
            input_ids=shift_labels,
            temperature=1.0,
        )

    torch.testing.assert_close(out_with_shift.log_probs, ref_log_probs, atol=1e-5, rtol=1e-5)


def test_dense_common_torch_backend_honors_shift_labels():
    _assert_fused_forward_uses_shift_labels(dense_common.forward_with_torch_backend)


def test_qwen3_5_torch_backend_honors_shift_labels():
    _assert_fused_forward_uses_shift_labels(qwen3_5.forward_with_torch_backend)


def test_qwen2_vl_torch_backend_honors_shift_labels():
    _assert_fused_forward_uses_shift_labels(qwen2_vl.forward_with_torch_backend)


def test_qwen3_vl_torch_backend_honors_shift_labels():
    _assert_fused_forward_uses_shift_labels(qwen3_vl.forward_with_torch_backend)


def test_dense_common_falls_back_to_local_roll_when_shift_labels_absent():
    """Backward-compat: callers that don't pass `shift_labels` see unchanged behavior
    (local roll over the input). This preserves the SP=1 path and any non-engine callers.
    """
    hidden_size, vocab_size = 8, 64
    model = _make_fake_lm(hidden_size, vocab_size)
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

    out_no_shift = dense_common.forward_with_torch_backend(
        model,
        input_ids=input_ids,
        labels=None,
        temperature=1.0,
        return_dict=True,
    )

    # Equivalent: explicit shift_labels = torch.roll(input_ids, -1).
    out_explicit = dense_common.forward_with_torch_backend(
        model,
        input_ids=input_ids,
        labels=None,
        temperature=1.0,
        shift_labels=torch.roll(input_ids, shifts=-1, dims=-1),
        return_dict=True,
    )

    torch.testing.assert_close(out_no_shift.log_probs, out_explicit.log_probs, atol=1e-5, rtol=1e-5)
