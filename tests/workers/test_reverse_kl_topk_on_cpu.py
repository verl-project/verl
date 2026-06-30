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
"""Unit tests for the student-top-K reverse-KL distillation loss (verl#6676).

These tests live next to ``test_distillation_topk_symmetry_on_cpu.py`` and follow
the same CPU-only contract: no FSDP, no CUDA, no flash-attn — just direct calls
into :func:`verl.trainer.distillation.fsdp.losses.compute_reverse_kl_topk` and the
top-level wrapper registered as ``loss_mode='reverse_kl_topk'``.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import compute_reverse_kl_topk as compute_fsdp_reverse_kl_topk
from verl.trainer.distillation.losses import (
    compute_reverse_kl_topk as collect_reverse_kl_topk_metrics,
)
from verl.trainer.distillation.losses import (
    get_distillation_loss_fn,
    get_distillation_loss_settings,
)


def _config(log_prob_min_clamp=None):
    return SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=log_prob_min_clamp))


def _student_topk_inputs(B=2, T=3, V=8, K=3, seed=0):
    """Random student logits + the matching student-top-K IDs (stops the gradient
    on the indices, like the trainer does in practice)."""
    torch.manual_seed(seed)
    student_logits = torch.randn(B, T, V, requires_grad=True)
    with torch.no_grad():
        topk = torch.topk(F.log_softmax(student_logits, dim=-1), k=K, dim=-1)
    return student_logits, topk.indices.detach()


def test_reverse_kl_topk_is_nonnegative():
    """KL(Q || P) on the student-top-K support is >= 0 up to numerical noise.

    Use independent random tensors for student logits and teacher log-probs (so
    Q and P do not coincide and the KL is materially positive)."""
    student_logits, student_topk_ids = _student_topk_inputs()
    torch.manual_seed(123)
    teacher_logits = torch.randn(*student_logits.shape)
    teacher_on_student_logp = torch.gather(F.log_softmax(teacher_logits, dim=-1), dim=-1, index=student_topk_ids)

    out = compute_fsdp_reverse_kl_topk(
        student_logits=student_logits,
        teacher_on_student_logp=teacher_on_student_logp,
        student_topk_ids=student_topk_ids,
        config=_config(),
        data_format="thd",
    )
    assert torch.all(out["distillation_losses"] >= -1e-6), out["distillation_losses"]
    # For independent Q and P the KL should be materially positive somewhere.
    assert out["distillation_losses"].max().item() > 1e-3


def test_reverse_kl_topk_is_zero_when_distributions_match():
    """KL(Q || Q) == 0 — feeding student log-probs as the teacher log-probs gives a zero loss."""
    student_logits, student_topk_ids = _student_topk_inputs()
    student_log_probs = F.log_softmax(student_logits.detach(), dim=-1)
    teacher_on_student_logp = torch.gather(student_log_probs, dim=-1, index=student_topk_ids)

    out = compute_fsdp_reverse_kl_topk(
        student_logits=student_logits,
        teacher_on_student_logp=teacher_on_student_logp,
        student_topk_ids=student_topk_ids,
        config=_config(),
        data_format="thd",
    )
    torch.testing.assert_close(
        out["distillation_losses"], torch.zeros_like(out["distillation_losses"]), atol=1e-6, rtol=0.0
    )


def test_reverse_kl_topk_gradient_flows_to_student():
    """``loss.sum().backward()`` must populate ``student_logits.grad`` while leaving
    the teacher tensor's gradient untouched (teacher is treated as a constant)."""
    student_logits, student_topk_ids = _student_topk_inputs()
    torch.manual_seed(7)
    teacher_logits = torch.randn(*student_logits.shape)
    teacher_on_student_logp = torch.gather(F.log_softmax(teacher_logits, dim=-1), dim=-1, index=student_topk_ids)

    out = compute_fsdp_reverse_kl_topk(
        student_logits=student_logits,
        teacher_on_student_logp=teacher_on_student_logp,
        student_topk_ids=student_topk_ids,
        config=_config(),
        data_format="thd",
    )
    out["distillation_losses"].sum().backward()

    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0
    # The teacher tensor is detached input from the worker; we did not set requires_grad on it.
    assert teacher_on_student_logp.grad is None


def test_reverse_kl_topk_matches_reference_formula():
    """The token-wise loss equals the manual reference reverse-KL computed with ``q~`` /
    ``p~``, the student / teacher probabilities renormalized onto the shared
    student-top-K support (softmax over the K gathered logits)."""
    student_logits, student_topk_ids = _student_topk_inputs(B=1, T=4, V=10, K=4, seed=2)
    torch.manual_seed(2)
    teacher_logits = torch.randn(*student_logits.shape)
    teacher_on_student_logp = torch.gather(F.log_softmax(teacher_logits, dim=-1), dim=-1, index=student_topk_ids)

    out = compute_fsdp_reverse_kl_topk(
        student_logits=student_logits,
        teacher_on_student_logp=teacher_on_student_logp,
        student_topk_ids=student_topk_ids,
        config=_config(),
        data_format="thd",
    )
    student_log_probs = F.log_softmax(student_logits.detach(), dim=-1)
    student_topk_logp = torch.gather(student_log_probs, dim=-1, index=student_topk_ids)
    # Renormalize both sides onto the student-top-K support (matches the loss).
    student_norm = student_topk_logp - torch.logsumexp(student_topk_logp, dim=-1, keepdim=True)
    teacher_norm = teacher_on_student_logp - torch.logsumexp(teacher_on_student_logp, dim=-1, keepdim=True)
    expected = (student_norm.exp() * (student_norm - teacher_norm)).sum(dim=-1)
    torch.testing.assert_close(out["distillation_losses"], expected, atol=1e-6, rtol=1e-6)


def test_reverse_kl_topk_invariant_to_mass_outside_topk():
    """Renormalizing onto the student-top-K support makes the loss depend only on
    the within-top-K logits, so leaking mass outside the top-K must not change it.
    Regression guard for the de-peaking collapse."""
    student_logits, student_topk_ids = _student_topk_inputs(B=2, T=3, V=8, K=3, seed=5)
    torch.manual_seed(9)
    teacher_logits = torch.randn(*student_logits.shape)
    teacher_on_student_logp = torch.gather(F.log_softmax(teacher_logits, dim=-1), dim=-1, index=student_topk_ids)

    def _loss(logits):
        return compute_fsdp_reverse_kl_topk(
            student_logits=logits,
            teacher_on_student_logp=teacher_on_student_logp,
            student_topk_ids=student_topk_ids,
            config=_config(),
            data_format="thd",
        )["distillation_losses"]

    base = _loss(student_logits)

    # Raise every non-top-K logit -> mass leaks outside the (fixed) top-K support.
    # With renormalization the loss must not move.
    bumped = student_logits.detach().clone()
    outside_mask = torch.ones_like(bumped, dtype=torch.bool)
    outside_mask.scatter_(-1, student_topk_ids, False)
    bumped[outside_mask] += 5.0
    bumped.requires_grad_(True)

    torch.testing.assert_close(_loss(bumped), base, atol=1e-5, rtol=1e-5)



def test_reverse_kl_topk_registered_and_collects_mass_metrics():
    """``loss_mode='reverse_kl_topk'`` is registered in the distillation loss registry
    and its top-level wrapper aggregates the student/teacher mass diagnostics over
    the response mask."""
    settings = get_distillation_loss_settings("reverse_kl_topk")
    assert settings.use_topk and not settings.use_estimator
    assert get_distillation_loss_fn("reverse_kl_topk") is collect_reverse_kl_topk_metrics

    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.10, 0.20, 0.30]),
        "student_mass": torch.tensor([0.97, 0.95, 0.10]),
        "teacher_mass": torch.tensor([0.50, 0.70, 0.05]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=8))

    losses, metrics = collect_reverse_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    # Loss is clamped non-negative and shape-aligns with the response mask.
    assert torch.all(losses >= 0.0)
    assert losses.shape == data["response_mask"].shape

    # student_mass / teacher_mass means are computed only over response-masked tokens.
    assert metrics["distillation/student_mass"] == pytest.approx((0.97 + 0.95) / 2)
    assert metrics["distillation/teacher_mass"] == pytest.approx((0.50 + 0.70) / 2)
