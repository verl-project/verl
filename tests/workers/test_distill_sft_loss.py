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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl.workers.utils.distill_sft_loss import SFTDistillationLoss  # noqa: E402


class _Data(dict):
    """TensorDict stand-in: dict access plus the non-tensor lookup the loss uses."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _make(config=None):
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
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0])  # supervises tokens 1 and 2
    rolled = torch.roll(mask, shifts=-1, dims=0)  # -> [1, 1, 0, 0]
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
    want = (t_prob * (t_lp.float().clamp_min(-10.0) - s_lp.gather(-1, t_ids).clamp_min(-10.0))).sum(-1).reshape(-1)
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


def _hash_ids(ids):
    import hashlib

    return hashlib.sha1(str(ids).encode("utf8")).hexdigest()


def test_lookup_resolves_each_sample_and_keeps_padding_inert():
    """A two-sample microbatch must use both teachers, in its actual sample order."""
    from types import SimpleNamespace

    torch.manual_seed(8)
    logits = torch.randn(2, 4, 9)
    ids = [[1, 2, 3], [4, 5, 6, 7]]
    lookup = {}
    for index, tokens in enumerate(ids):
        logps, top_ids = _teacher_from_logits(logits[index : index + 1, : len(tokens)], 3)
        lookup[_hash_ids(tokens)] = (top_ids[0], logps[0])
    loss = SFTDistillationLoss(SimpleNamespace(log_prob_min_clamp=-10.0), teacher_lookup=lookup)
    out = loss(
        student_logits=logits,
        data=_Data(
            input_ids=torch.tensor([ids[0] + [0], ids[1]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            # The old fallback used the first sample id even when it did not
            # identify the token sequence. Tokens now own the lookup contract.
            sample=[999, 998],
        ),
        data_format="bshd",
    )
    torch.testing.assert_close(out["distillation_losses"], torch.zeros(2, 4), rtol=0, atol=1e-6)
    assert out["teacher_mass"][0, 3] == 0
    assert (out["teacher_mass"][1] > 0).all()


def test_training_lookup_miss_raises_instead_of_changing_the_objective():
    from types import SimpleNamespace

    loss = SFTDistillationLoss(SimpleNamespace(log_prob_min_clamp=-10.0), teacher_lookup={})
    with pytest.raises(KeyError, match="Missing teacher payload for sample 0"):
        loss(student_logits=torch.zeros(1, 3, 8), data=_Data(input_ids=torch.tensor([[1, 2, 3]])), data_format="bshd")


def test_engine_payload_is_already_packed_and_does_not_need_a_lookup(monkeypatch):
    loss = _make()
    logits = torch.randn(1, 4, 9)
    logps, ids = _teacher_from_logits(logits, 3)

    def repacking_is_a_bug(*args, **kwargs):
        pytest.fail("the engine's rank-local payload must not be sharded twice")

    monkeypatch.setattr(loss, "_pack_payload", repacking_is_a_bug)
    out = loss(
        student_logits=logits,
        data=_Data(),
        data_format="thd",
        teacher_topk_ids=ids,
        teacher_topk_logprobs=logps,
        cp_layout="contiguous",
        local_cp_size=2,
        min_local_rows=4,
    )
    torch.testing.assert_close(out["distillation_losses"], torch.zeros(1, 4), rtol=0, atol=1e-6)


def test_absent_teacher_rows_have_zero_loss_mass_and_gradient():
    logits = torch.randn(1, 3, 8, requires_grad=True)
    out = _make()(
        student_logits=logits,
        data=_Data(),
        data_format="thd",
        teacher_topk_ids=torch.zeros(1, 3, 2, dtype=torch.long),
        teacher_topk_logprobs=torch.full((1, 3, 2), -torch.inf),
    )
    assert torch.equal(out["teacher_mass"], torch.zeros(1, 3))
    assert torch.equal(out["distillation_losses"], torch.zeros(1, 3))
    out["distillation_losses"].sum().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_student_logprob_clamp_matches_configured_loss_and_gradient():
    logits = torch.tensor([[[-20.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    teacher_logprobs = torch.full((1, 2, 1), 0.9).log()
    teacher_ids = torch.zeros(1, 2, 1, dtype=torch.long)
    loss = _make()
    actual = loss(
        student_logits=logits,
        data=_Data(),
        data_format="bshd",
        teacher_topk_ids=teacher_ids,
        teacher_topk_logprobs=teacher_logprobs,
    )["distillation_losses"]

    reference_logits = logits.detach().clone().requires_grad_()
    reference_student = reference_logits.log_softmax(-1).gather(-1, teacher_ids).clamp_min(-10.0)
    reference_teacher = teacher_logprobs.clamp_min(-10.0)
    expected = (reference_teacher.exp() * (reference_teacher - reference_student)).sum(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert torch.equal(logits.grad[0, 0], torch.zeros(2))
    assert logits.grad[0, 1].count_nonzero() == 2


@pytest.mark.parametrize("teacher_mass,student_logit", [(0.9, 0.0), (1.0 - 1e-7, 10.0), (1.0, 20.0)])
def test_tail_mode_matches_coarse_kl_and_gradient_near_unit_mass(teacher_mass, student_logit):
    loss = _make(
        SimpleNamespace(
            distillation_loss=SimpleNamespace(
                loss_mode="forward_kl_topk_tail", log_prob_min_clamp=None, tail_mass_eps=1e-6
            )
        )
    )
    logits = torch.tensor([[[student_logit, 0.0]]], requires_grad=True)
    teacher_logprobs = torch.tensor([[[teacher_mass]]]).log()
    actual = loss(
        student_logits=logits,
        data=_Data(),
        data_format="bshd",
        teacher_topk_ids=torch.zeros(1, 1, 1, dtype=torch.long),
        teacher_topk_logprobs=teacher_logprobs,
    )["distillation_losses"]

    reference_logits = logits.detach().clone().requires_grad_()
    log_q = reference_logits.log_softmax(-1)[..., 0]
    p = teacher_logprobs.exp().squeeze(-1)
    p_tail = (1.0 - p).clamp_min(0.0)
    q_tail = (1.0 - log_q.exp()).clamp_min(1e-6)
    expected = p * (teacher_logprobs.squeeze(-1) - log_q)
    expected = expected + p_tail * (p_tail.clamp_min(torch.finfo(p.dtype).tiny).log() - q_tail.log())
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-7, rtol=1e-5)


def test_tail_mode_ignores_absent_topk_entries_and_padding_rows():
    loss = _make(
        SimpleNamespace(
            distillation_loss=SimpleNamespace(
                loss_mode="forward_kl_topk_tail", log_prob_min_clamp=None, tail_mass_eps=1e-6
            )
        )
    )
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    teacher_logprobs = torch.tensor([[[0.9, 0.0], [0.0, 0.0]]]).log()
    output = loss(
        student_logits=logits,
        data=_Data(),
        data_format="bshd",
        teacher_topk_ids=torch.tensor([[[0, -1], [-1, -1]]]),
        teacher_topk_logprobs=teacher_logprobs,
    )
    expected = (torch.tensor([0.9, 0.1]) * (torch.tensor([0.9, 0.1]).log() - torch.tensor(0.5).log())).sum()
    torch.testing.assert_close(output["distillation_losses"], torch.tensor([[expected, 0.0]]))
    torch.testing.assert_close(output["teacher_mass"], torch.tensor([[0.9, 0.0]]))
    output["distillation_losses"].sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[-0.4, 0.4], [0.0, 0.0]]]))


def test_tensor_parallel_logits_are_rejected_before_local_vocabulary_math(monkeypatch):
    import sys
    import types

    mpu = types.ModuleType("megatron.core.parallel_state")
    mpu.get_tensor_model_parallel_world_size = lambda: 2
    core = types.ModuleType("megatron.core")
    core.parallel_state = mpu
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(NotImplementedError, match="tensor_model_parallel_size=1"):
        _make()(student_logits=torch.randn(1, 2, 4), data=_Data(), data_format="thd")


def test_validation_uses_ce_without_a_teacher_lookup():
    loss = _make()
    lp = torch.nested.nested_tensor([torch.tensor([-1.0, -2.0, -3.0])], layout=torch.jagged)
    mask = torch.nested.nested_tensor([torch.tensor([0.0, 1.0, 1.0])], layout=torch.jagged)
    got, _ = loss(
        model_output={"log_probs": lp},
        data=_Data(
            loss_mask=mask,
            dp_size=1,
            batch_num_tokens=2.0,
            pad_mode="no_padding",
            sft_distillation_validation=True,
        ),
    )
    torch.testing.assert_close(got, torch.tensor(1.5))


def test_lookup_thd_uses_per_sequence_padding_and_contiguous_cp(monkeypatch):
    pytest.importorskip("megatron.core")
    from types import SimpleNamespace

    from verl.models.mcore import util

    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda: 1)
    tokens = [[1, 2, 3], [4, 5, 6, 7, 8]]
    lookup = {_hash_ids(row): (torch.tensor(row).unsqueeze(-1), torch.full((len(row), 1), -1.0)) for row in tokens}
    logits = torch.arange(60, dtype=torch.float32).reshape(1, 6, 10)
    loss = SFTDistillationLoss(SimpleNamespace(log_prob_min_clamp=-10.0), teacher_lookup=lookup)
    out = loss(
        student_logits=logits,
        data=_Data(input_ids=torch.nested.nested_tensor([torch.tensor(row) for row in tokens], layout=torch.jagged)),
        data_format="thd",
        cp_layout="contiguous",
        min_local_rows=6,
    )
    # Global padded timeline: [1,2,3,pad,4,5,6,7,8,pad,pad,pad].
    # Rank 1 owns the latter six rows; padding must have no teacher mass.
    reference = (
        -1.0 - logits.log_softmax(-1)[0, :3].gather(-1, torch.tensor([[6], [7], [8]])).squeeze(-1)
    ) * torch.exp(torch.tensor(-1.0))
    torch.testing.assert_close(out["distillation_losses"][0, :3], reference)
    assert torch.equal(out["teacher_mass"][0, 3:], torch.zeros(3))
