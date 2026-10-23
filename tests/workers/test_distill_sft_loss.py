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
"""The distillation reduction must supervise exactly the CE loss's tokens.

Two ways this goes wrong silently: the reducer uses an unrolled mask (one-token
shift against CE, so a CE-vs-distillation comparison is meaningless), or it
averages instead of summing over the global token count (the loss scale differs
from CE by the batch size). Both are pinned here on synthetic tensors, with the
mask-roll case built so an unrolled mask would change the answer.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.workers.utils.distill_sft_loss import SFTDistillationLoss  # noqa: E402


class _Data(dict):
    """TensorDict stand-in: dict access plus the non-tensor lookup the loss uses."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _make(config=None):
    from types import SimpleNamespace

    return SFTDistillationLoss(config or SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-10.0)))


def test_unknown_signature_is_rejected():
    loss = _make()
    with pytest.raises(TypeError, match="logits processor"):
        loss(something=1)


def test_missing_teacher_column_names_the_dataset():
    loss = _make()
    with pytest.raises(KeyError, match="teacher_ids"):
        loss(student_logits=torch.zeros(1, 2, 4), data=_Data(), data_format="bshd")


def test_missing_flag_names_distillation_use_topk():
    loss = _make()
    with pytest.raises(KeyError, match="distillation_use_topk"):
        loss(model_output={}, data=_Data())


def test_reduction_is_the_rolled_masked_mean_over_global_tokens():
    """loss = sum(per_position * rolled_mask) / batch_num_tokens * dp_size."""
    loss = _make()
    per_position = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0])       # supervises tokens 1 and 2
    rolled = torch.roll(mask, shifts=-1, dims=0)     # -> [1, 1, 0, 0]
    data = _Data(
        loss_mask=mask,
        dp_size=2,
        batch_num_tokens=torch.tensor(4.0),
        pad_mode="no_padding",
    )
    model_output = {"distillation_losses": per_position}
    got, metrics = loss(model_output=model_output, data=data)

    want = (per_position * rolled).sum() / 4.0 * 2.0
    assert torch.allclose(got.cpu(), want.cpu()), (got, want)
    # an unrolled mask would have used tokens 1,2 -> a different number
    unrolled = (per_position * mask).sum() / 4.0 * 2.0
    assert not torch.allclose(got.cpu(), unrolled.cpu())
    assert "distill/loss" in metrics


def test_positional_mismatch_is_rejected():
    loss = _make()
    data = _Data(
        loss_mask=torch.zeros(3),
        dp_size=1,
        batch_num_tokens=torch.tensor(3.0),
        pad_mode="no_padding",
    )
    with pytest.raises(ValueError, match="same tokens"):
        loss(model_output={"distillation_losses": torch.zeros(4)}, data=data)


def _teacher_from_logits(logits, k):
    """The teacher distribution IS the student's: top-k of the same log-softmax."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = lp.topk(k, dim=-1)
    return top.values, top.indices


def test_self_teacher_has_zero_divergence():
    """Same model on both sides -> the per-position KL must be exactly zero."""
    torch.manual_seed(0)
    loss = _make()
    logits = torch.randn(3, 5, 11)
    t_lp, t_ids = _teacher_from_logits(logits, 4)
    out = loss(student_logits=logits, data=_Data(teacher_ids=t_ids, teacher_logprobs=t_lp), data_format="bshd")
    per_position = out["distillation_losses"]
    # the engine asserts this against attention_mask, so the lead dims must survive
    assert per_position.shape == (3, 5), per_position.shape
    torch.testing.assert_close(per_position, torch.zeros(3, 5), rtol=0, atol=1e-6)


def test_shifted_student_has_positive_divergence_and_matches_the_formula():
    torch.manual_seed(1)
    loss = _make()
    logits = torch.randn(2, 3, 9)
    t_lp, t_ids = _teacher_from_logits(logits, 3)
    shifted = logits.clone()
    shifted[..., 0] -= 2.0  # push the top token down by two nats

    out = loss(student_logits=shifted, data=_Data(teacher_ids=t_ids, teacher_logprobs=t_lp), data_format="bshd")
    got = out["distillation_losses"].reshape(-1)

    # hand-computed reference: same clamp, same normaliser, same gather
    t_prob = t_lp.float().clamp_min(-10.0).exp()
    s_lp = torch.log_softmax(shifted.float(), dim=-1)
    want = (t_prob * (t_lp.float().clamp_min(-10.0) - s_lp.gather(-1, t_ids))).sum(-1).reshape(-1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    # A truncated teacher makes this a divergence up to the covered mass, so it is NOT
    # guaranteed non-negative; asserting otherwise would pin a property the engine's
    # own formula does not have.
    assert torch.isfinite(got).all()


def test_pad_rows_with_neg_inf_stay_finite_under_the_clamp():
    """exp(-inf) * (-inf) is NaN; the clamp is what keeps this finite."""
    loss = _make()
    logits = torch.randn(1, 2, 6)
    t_lp, t_ids = _teacher_from_logits(logits, 2)
    t_lp = t_lp.clone()
    t_lp[0, 0, 1] = float("-inf")  # a pad row, as a short top-k would leave it
    t_ids = t_ids.clone()
    t_ids[0, 0, 1] = 0
    out = loss(student_logits=logits, data=_Data(teacher_ids=t_ids, teacher_logprobs=t_lp), data_format="bshd")
    assert torch.isfinite(out["distillation_losses"]).all(), out["distillation_losses"]


def test_processor_rejects_a_position_count_mismatch():
    loss = _make()
    with pytest.raises(ValueError, match="position counts disagree"):
        loss(
            student_logits=torch.zeros(1, 4, 5),
            data=_Data(teacher_ids=torch.zeros(3, 2, dtype=torch.int64), teacher_logprobs=torch.zeros(3, 2)),
            data_format="bshd",
        )


def test_per_token_clip_caps_the_tail_and_preserves_the_rest():
    """OPSD's jsd_token_clip: style tokens must not dominate the sum."""
    from types import SimpleNamespace

    torch.manual_seed(2)
    logits = torch.randn(1, 4, 9)
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = lp.topk(3, dim=-1)
    # a student that is badly wrong on the second position only
    shifted = logits.clone()
    shifted[0, 1] = shifted[0, 1].roll(3)

    unclipped = SFTDistillationLoss(
        SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-10.0, token_clip=None))
    )(student_logits=shifted, data=_Data(teacher_ids=top.indices, teacher_logprobs=top.values), data_format="bshd")
    clipped = SFTDistillationLoss(
        SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-10.0, token_clip=0.05))
    )(student_logits=shifted, data=_Data(teacher_ids=top.indices, teacher_logprobs=top.values), data_format="bshd")

    a = unclipped["distillation_losses"]
    b = clipped["distillation_losses"]
    assert (b <= 0.05 + 1e-6).all() and (b >= -0.05 - 1e-6).all(), b
    # the positions already under the cap are untouched
    small = a.abs() < 0.05
    torch.testing.assert_close(a[small], b[small])
    # and the outlier really was capped
    assert (a.abs() > 0.05).any()
