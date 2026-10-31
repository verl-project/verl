# Copyright 2026 Individual Contributor: Penghui Yang and the ResOPD authors
# SPDX-License-Identifier: Apache-2.0
"""Finite-action gradient checks; no models, accelerator or verl runtime needed."""

import pytest
import torch

from verl.trainer.distillation.resopd import resopd_loss


@pytest.mark.parametrize("k", [1, 2, 4, 5])
@pytest.mark.parametrize("seed", [3, 17, 42])
def test_all_actions_match_paper_and_expected_full_kl_gradient(k, seed):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(5, generator=generator, dtype=torch.float64)
    log_q = torch.randn(5, generator=generator, dtype=torch.float64).log_softmax(-1)
    log_p = logits.log_softmax(-1)
    p = log_p.exp()
    support = log_q.topk(k).indices
    score = torch.eye(5, dtype=torch.float64) - p[None, :]
    token_gradient = (log_p - log_q)[:, None] * score
    head_gradient = (p[support, None] * token_gradient[support]).sum(0)
    tau_p = 1 - p[support].sum()
    tau_q = 1 - log_q[support].exp().sum()
    if k < 5:
        bucket_gradient = (tau_p.log() - tau_q.log()) * (-(p[support, None] * score[support]).sum(0) / tau_p)
    else:
        bucket_gradient = torch.zeros_like(head_gradient)
    gradients, values, hets = [], [], []
    for action in range(5):
        z = logits.clone().requires_grad_()
        output = resopd_loss(z, log_q[support], support, log_q[action : action + 1], torch.tensor([action]))
        loss = output["distillation_losses"]
        actual = torch.autograd.grad(loss, z)[0]
        tail_hit = action not in support.tolist()
        expected = head_gradient + tau_p * bucket_gradient
        if tail_hit:
            expected = expected + token_gradient[action] - bucket_gradient
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        gradients.append(actual)
        values.append(loss.detach())
        hets.append(head_gradient + (token_gradient[action] if tail_hit else 0))

    z = logits.clone().requires_grad_()
    full_log_p = z.log_softmax(-1)
    full_kl = (full_log_p.exp() * (full_log_p - log_q)).sum()
    full_gradient = torch.autograd.grad(full_kl, z)[0]
    gradients = torch.stack(gradients)
    torch.testing.assert_close((p[:, None] * gradients).sum(0), full_gradient, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close((p * torch.stack(values)).sum(), full_kl.detach(), atol=2e-12, rtol=2e-12)

    # Verify the variance identity, without assuming every tail control improves variance.
    if k < 5:
        tail_mean = (full_gradient - head_gradient) / tau_p
        actual_reduction = (
            p * ((torch.stack(hets) - full_gradient).square().sum(-1) - (gradients - full_gradient).square().sum(-1))
        ).sum()
        predicted_reduction = (
            tau_p * (1 - tau_p) * (tail_mean.square().sum() - (tail_mean - bucket_gradient).square().sum())
        )
        torch.testing.assert_close(actual_reduction, predicted_reduction, atol=2e-12, rtol=2e-12)


def test_batched_tokens_have_independent_gradients_and_detached_teacher():
    z = torch.tensor([[[1.0, 0.2, -0.3], [0.3, -0.4, 0.8]]], requires_grad=True)
    log_q = torch.tensor([[[0.2, 0.8, -0.1], [0.4, 0.1, 0.7]]]).log_softmax(-1).requires_grad_()
    support = log_q.detach().topk(1).indices
    actions = torch.tensor([[[0], [1]]])
    result = resopd_loss(z, log_q.gather(-1, support), support, log_q.gather(-1, actions), actions)
    assert result["distillation_losses"].shape == (1, 2)
    result["distillation_losses"][0, 0].backward()
    assert log_q.grad is None
    assert z.grad[0, 1].eq(0).all()
    assert z.grad[0, 0].abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_input_computes_finite_fp32_loss(dtype):
    z = torch.tensor([2.0, 0.4, -0.7], dtype=dtype, requires_grad=True)
    log_q = torch.tensor([0.5, 1.0, -0.2]).log_softmax(-1)
    result = resopd_loss(z, log_q[:1], torch.tensor([0]), log_q[2:], torch.tensor([2]))
    loss = result["distillation_losses"]
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_matching_teacher_and_student_gives_zero_gradient():
    z = torch.tensor([1.3, 0.7, -0.3], dtype=torch.float64, requires_grad=True)
    log_q = z.detach().log_softmax(-1)
    for action in range(3):
        loss = resopd_loss(z, log_q[:2], torch.tensor([0, 1]), log_q[action : action + 1], torch.tensor([action]))
        gradient = torch.autograd.grad(loss["distillation_losses"], z)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(z))


def test_misaligned_sample_payload_fails():
    with pytest.raises(ValueError, match="one actual sampled token"):
        resopd_loss(
            torch.zeros(2, 5),
            torch.zeros(2, 1),
            torch.zeros(2, 1, dtype=torch.long),
            torch.zeros(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
        )
