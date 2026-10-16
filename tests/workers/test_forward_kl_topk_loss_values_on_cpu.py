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
"""Value-level CPU tests for ``compute_forward_kl_topk`` (FSDP backend).

The existing CPU coverage (``test_distillation_topk_symmetry_on_cpu``) only
asserts the two diagnostic outputs (``overlap_count`` /
``overlap_token_advantage``). The load-bearing outputs of the GKD-OPD loss --
``distillation_losses`` (the objective), ``student_mass`` / ``teacher_mass``,
the default-on ``log_prob_min_clamp`` branch, and the ``use_chunked_topk``
path -- had no value-level assertions, so a regression in ``kl_divergence`` or
the clamp/chunk handling would pass CI. These tests pin the loss math against
a from-scratch reference on tiny CPU tensors.
"""

from types import SimpleNamespace

import torch

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk


def _cfg(log_prob_min_clamp=None, use_chunked_topk=False):
    return SimpleNamespace(
        distillation_loss=SimpleNamespace(
            log_prob_min_clamp=log_prob_min_clamp,
            use_chunked_topk=use_chunked_topk,
            chunked_topk_chunk_size=4096,
        )
    )


def _nested(rows: torch.Tensor) -> torch.Tensor:
    # compute_forward_kl_topk expects jagged nested teacher tensors and reads
    # ``.values().unsqueeze(0)``; one batch row of (seqlen, topk) reproduces a
    # single-sequence micro-batch.
    return torch.nested.nested_tensor([rows], layout=torch.jagged)


def _make_inputs(seqlen=4, vocab=10, topk=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(1, seqlen, vocab, generator=g)
    teacher_log_probs = torch.log_softmax(torch.randn(seqlen, vocab, generator=g), dim=-1)
    t_lp, t_ids = torch.topk(teacher_log_probs, topk, dim=-1)
    return student_logits, t_lp, t_ids.long()


def _forward_kl_reference(student_logits, t_lp, t_ids, clamp=None):
    s_at_t = torch.gather(torch.log_softmax(student_logits, dim=-1), dim=-1, index=t_ids.unsqueeze(0))
    t_lp_b = t_lp.unsqueeze(0)
    if clamp is not None:
        t_lp_b, s_at_t = t_lp_b.clamp_min(clamp), s_at_t.clamp_min(clamp)
    loss = (t_lp_b.exp() * (t_lp_b - s_at_t)).sum(dim=-1)
    return loss, s_at_t, t_lp_b


def test_forward_kl_topk_loss_matches_reference():
    student_logits, t_lp, t_ids = _make_inputs()
    out = compute_forward_kl_topk(student_logits, _nested(t_lp), _nested(t_ids), _cfg(log_prob_min_clamp=None), "thd")
    # student_mass / teacher_mass are taken on the *unclamped* log-probs.
    s_at_t = torch.gather(torch.log_softmax(student_logits, dim=-1), dim=-1, index=t_ids.unsqueeze(0))
    ref_loss, _, _ = _forward_kl_reference(student_logits, t_lp, t_ids, clamp=None)
    torch.testing.assert_close(out["distillation_losses"], ref_loss)
    torch.testing.assert_close(out["student_mass"], s_at_t.exp().sum(dim=-1))
    torch.testing.assert_close(out["teacher_mass"], t_lp.unsqueeze(0).exp().sum(dim=-1))


def test_log_prob_min_clamp_branch_changes_loss():
    # Force student log-probs far below the clamp at the teacher's top-k ids:
    # a hard peak on id 0 drives log-prob ~ -60 at every other id.
    seqlen, vocab, clamp = 2, 12, -10.0
    student_logits = torch.full((1, seqlen, vocab), -30.0)
    student_logits[..., 0] = 30.0
    t_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t_lp = torch.log_softmax(torch.randn(seqlen, vocab), dim=-1).gather(dim=-1, index=t_ids)

    out = compute_forward_kl_topk(
        student_logits, _nested(t_lp), _nested(t_ids.long()), _cfg(log_prob_min_clamp=clamp), "thd"
    )
    ref_clamped, _, _ = _forward_kl_reference(student_logits, t_lp, t_ids, clamp=clamp)
    ref_unclamped, _, _ = _forward_kl_reference(student_logits, t_lp, t_ids, clamp=None)
    torch.testing.assert_close(out["distillation_losses"], ref_clamped)
    # The clamp is on by default and materially changes the loss here.
    assert not torch.allclose(ref_clamped, ref_unclamped)


def test_chunked_topk_matches_default_path():
    student_logits, t_lp, t_ids = _make_inputs(seqlen=6, vocab=16, topk=4, seed=1)
    default = compute_forward_kl_topk(
        student_logits, _nested(t_lp), _nested(t_ids), _cfg(use_chunked_topk=False), "thd"
    )
    chunked = compute_forward_kl_topk(student_logits, _nested(t_lp), _nested(t_ids), _cfg(use_chunked_topk=True), "thd")
    assert set(chunked) == set(default)
    for key in default:
        torch.testing.assert_close(chunked[key], default[key])
