# Copyright 2026 Individual Contributor
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
"""Numerics of ``kl_loss_type: topk_rb`` against the exact full-vocabulary KL(q || p) and its gradient.

Unbiasedness is checked by exact expectation over the sampled token (every token, weighted by its sampling
probability), not by Monte Carlo: rare tail tokens make Monte Carlo error estimates unreliable.
"""

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import verl.trainer.ppo.topk_rb_kl as topk_rb_kl
from verl.trainer.ppo.topk_rb_kl import compute_coarse_reverse_kl, compute_topk_log_probs, compute_topk_rb_kl


def _logits(V, drift, seed=0, ref_scale=3.0):
    """Policy and reference logits; ``drift`` sets how far the policy is from the reference."""
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(V, generator=g, dtype=torch.float64) * ref_scale
    return ref + drift * torch.randn(V, generator=g, dtype=torch.float64), ref


def _exact(z, ref):
    """Exact KL(softmax(z) || softmax(ref)) and its gradient wrt z."""
    lq, lp = z.log_softmax(-1), ref.log_softmax(-1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    return kl, lq.exp() * ((lq - lp) - kl.unsqueeze(-1))


def _coarse_oracle(z, ref_ids, ref):
    """Coarse KL and log policy tail mass by plain autograd over the full log-softmax (rows of z)."""
    lq, lp = z.log_softmax(-1), ref.log_softmax(-1)
    in_topk = torch.zeros_like(lq, dtype=torch.bool).scatter(-1, ref_ids, True)
    log_q_tail = lq.masked_fill(in_topk, float("-inf")).logsumexp(-1)
    log_p_tail = lp.masked_fill(in_topk, float("-inf")).logsumexp(-1)
    topk_term = torch.where(in_topk, lq.exp() * (lq - lp), 0.0).sum(-1)
    return topk_term + log_q_tail.exp() * (log_q_tail - log_p_tail), log_q_tail


def _per_token(z, ref, k, y, old_log_probs=None, rollout_weights=None):
    """Single-sample surrogate gradient (wrt z) and value for each sampled token in ``y``."""
    ref_lp, ref_ids = compute_topk_log_probs(ref.unsqueeze(0), k)
    n = y.shape[0]
    ref_lp, ref_ids = ref_lp.expand(n, -1), ref_ids.expand(n, -1)
    zz = z.expand(n, -1).clone().requires_grad_(True)
    coarse, log_q_tail = compute_coarse_reverse_kl(zz, ref_ids, ref_lp)
    log_prob = zz.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)
    loss, value = compute_topk_rb_kl(
        coarse,
        log_q_tail,
        ref_lp[:, -1],
        (ref_ids == y[:, None]).any(-1),
        log_prob,
        ref.log_softmax(-1)[y],
        old_log_prob=None if old_log_probs is None else old_log_probs[y],
        rollout_is_weights=None if rollout_weights is None else rollout_weights[y],
    )
    (grad,) = torch.autograd.grad(loss.sum(), zz)
    return grad, value


def _expected(z, ref, k, sampling_log_probs, **weights):
    """Exact expectation of gradient and value over ``y ~ exp(sampling_log_probs)``."""
    grad, value = _per_token(z, ref, k, torch.arange(z.shape[0]), **weights)
    w = sampling_log_probs.exp()
    return (w[:, None] * grad).sum(0), (w * value).sum()


def test_topk_log_probs_match_log_softmax_and_tail():
    logits = torch.randn(37, 50, dtype=torch.float64)
    log_probs, ids = compute_topk_log_probs(logits, k=7, chunk_size=5)
    expected, expected_ids = logits.log_softmax(-1).topk(7, -1)
    tail = 1 - expected.exp().sum(-1)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(log_probs, torch.cat([expected, tail.log().unsqueeze(-1)], -1).float())


def test_tail_mass_is_exact_when_the_topk_holds_nearly_everything():
    """fp32, top-k mass 1 - 1e-12: 1 - sum(top-k) is pure rounding, the complement log-sum-exp is not."""
    logits = torch.full((1, 40), -30.0)
    logits[0, :4] = 0.0  # tail of 36 tokens at relative mass e^-30 each
    log_probs, _ = compute_topk_log_probs(logits, k=4)
    exact = (36 * torch.tensor(-30.0, dtype=torch.float64).exp() / 4).log()  # ~log(3.4e-13)
    assert abs(log_probs[0, -1].item() - exact.item()) < 1e-4


def test_deep_tail_is_exact_and_its_gradient_finite():
    """fp32, tail 200 nats below the top: exp(-200) underflows relative to the row maximum, so the tail must be
    summed relative to its own maximum. Exact log tail, finite gradient, and r stays bounded (log p_t >= log p_y)."""
    z = torch.full((2, 40), -200.0)
    z[:, :4] = 0.0
    z[1, 4] = -150.0  # a policy tail token far above the rest of the tail
    ref_lp, ref_ids = compute_topk_log_probs(z, k=4)
    exact = torch.tensor([-200.0 + math.log(36 / 4), -150.0 - math.log(4)])
    torch.testing.assert_close(ref_lp[:, -1], exact, atol=1e-4, rtol=0)

    zz = z.clone().requires_grad_(True)
    coarse, log_q_tail = compute_coarse_reverse_kl(zz, ref_ids, ref_lp)
    # token 4 sampled from the tail: r = log(q_y / q_t) - log(p_y / p_t), both ratios <= 1 if the tails are exact
    log_prob, ref_log_prob = zz.log_softmax(-1)[:, 4], z.log_softmax(-1)[:, 4]
    assert (ref_lp[:, -1] >= ref_log_prob - 1e-4).all()
    in_topk = torch.tensor([False, False])
    loss, value = compute_topk_rb_kl(coarse, log_q_tail, ref_lp[:, -1], in_topk, log_prob, ref_log_prob)
    (g,) = torch.autograd.grad((loss + coarse + log_q_tail).sum(), zz)
    assert torch.isfinite(g).all() and torch.isfinite(coarse).all()
    assert value.abs().max() < 10


def test_chunk_size_zero_means_no_chunking():
    z = torch.randn(5, 30)
    ref_lp, ref_ids = compute_topk_log_probs(z + torch.randn(5, 30), 3, chunk_size=0)
    torch.testing.assert_close(
        compute_coarse_reverse_kl(z, ref_ids, ref_lp, chunk_size=0), compute_coarse_reverse_kl(z, ref_ids, ref_lp)
    )


@pytest.mark.parametrize("chunk_size", [3, 4096])
@pytest.mark.parametrize("k", [3, 40])
def test_coarse_kl_and_gradients_match_autograd_oracle(chunk_size, k):
    """Both outputs (coarse KL and log q_tail) carry the oracle's gradient; k = V exercises the empty tail."""
    torch.manual_seed(0)
    z = torch.randn(8, 40, dtype=torch.float64, requires_grad=True)
    ref = z.detach() + torch.randn(8, 40, dtype=torch.float64)
    ref_lp, ref_ids = compute_topk_log_probs(ref, k)
    w1, w2 = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)

    coarse, log_q_tail = compute_coarse_reverse_kl(z, ref_ids, ref_lp, chunk_size=chunk_size)
    (g,) = torch.autograd.grad((coarse * w1 + log_q_tail * w2).sum(), z)
    tol = dict(atol=1e-5, rtol=1e-5)  # the top-k log-probs are computed in fp32
    if k == 40:  # empty tail: no tail term, exact KL, and log q_tail is the (constant) floor
        torch.testing.assert_close(coarse.double(), _exact(z, ref)[0], **tol)
        (g_oracle,) = torch.autograd.grad((_exact(z, ref)[0] * w1).sum(), z)
    else:
        coarse_o, log_q_tail_o = _coarse_oracle(z, ref_ids, ref)
        torch.testing.assert_close(coarse.double(), coarse_o, **tol)
        torch.testing.assert_close(log_q_tail.double(), log_q_tail_o, **tol)
        (g_oracle,) = torch.autograd.grad((coarse_o * w1 + log_q_tail_o * w2).sum(), z)
    torch.testing.assert_close(g, g_oracle, **tol)


def test_saves_no_copy_of_bf16_logits_and_returns_their_dtype():
    z = torch.randn(4, 30, dtype=torch.bfloat16, requires_grad=True)
    ref_lp, ref_ids = compute_topk_log_probs(torch.randn(4, 30), 3)
    coarse, log_q_tail = compute_coarse_reverse_kl(z, ref_ids, ref_lp)
    assert log_q_tail.grad_fn.saved_tensors[0].data_ptr() == z.data_ptr()
    (g,) = torch.autograd.grad(coarse.sum(), z)
    assert g.dtype == torch.bfloat16 and torch.isfinite(g.float()).all()


def test_full_vocab_topk_is_exact_for_every_draw():
    z, ref = _logits(40, drift=1.0)
    kl, grad = _exact(z, ref)
    g, value = _per_token(z, ref, 40, torch.multinomial(z.softmax(-1), 16, replacement=True))
    torch.testing.assert_close(value, kl.expand_as(value), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(g, grad.expand_as(g), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("k", [1, 5, 50])
def test_coarse_kl_is_a_lower_bound(k):
    for seed in range(5):
        z, ref = _logits(200, drift=2.0, seed=seed)
        ref_lp, ref_ids = compute_topk_log_probs(ref.unsqueeze(0), k)
        coarse, _ = compute_coarse_reverse_kl(z.unsqueeze(0), ref_ids, ref_lp)
        assert -1e-6 <= coarse.item() <= _exact(z, ref)[0].item() + 1e-6


@pytest.mark.parametrize("drift,k", [(0.5, 1), (0.5, 20), (1.5, 20), (1.5, 100), (3.0, 5)])
def test_unbiased_on_policy(drift, k):
    z, ref = _logits(400, drift=drift, seed=1)
    kl, grad = _exact(z, ref)
    g_mean, v_mean = _expected(z, ref, k, z.log_softmax(-1))
    assert (g_mean - grad).norm() < 1e-5 * grad.norm()
    assert abs(v_mean.item() - kl.item()) < 1e-6


def test_unbiased_off_policy_only_with_both_weights():
    """y ~ pi_rollout: unbiased with pi_theta/pi_old and rollout_is_weights = pi_old/pi_rollout; each is needed."""
    z, ref = _logits(300, drift=1.5, seed=2)
    g = torch.Generator().manual_seed(9)
    old = (z + 0.7 * torch.randn(300, dtype=torch.float64, generator=g)).log_softmax(-1)
    rollout = (old + 0.5 * torch.randn(300, dtype=torch.float64, generator=g)).log_softmax(-1)
    kl, grad = _exact(z, ref)
    weights = dict(old_log_probs=old, rollout_weights=(old - rollout).exp())

    g_mean, v_mean = _expected(z, ref, 10, rollout, **weights)
    assert (g_mean - grad).norm() < 1e-5 * grad.norm()
    assert abs(v_mean.item() - kl.item()) < 1e-6
    for missing in weights:
        g_mean, _ = _expected(z, ref, 10, rollout, **{**weights, missing: None})
        assert (g_mean - grad).norm() > 1e-2 * grad.norm(), missing


def test_ratio_is_one_on_policy():
    z, ref = _logits(60, drift=1.0, seed=3)
    y = torch.multinomial(z.softmax(-1), 64, replacement=True)
    torch.testing.assert_close(_per_token(z, ref, 5, y, old_log_probs=z.log_softmax(-1)), _per_token(z, ref, 5, y))


def test_gradient_noise_is_below_k2():
    """The point of the estimator: lower single-sample gradient error than k2 (the k3+ gradient)."""
    z, ref = _logits(400, drift=0.5, seed=1)
    _, grad = _exact(z, ref)
    q = z.softmax(-1)
    g, _ = _per_token(z, ref, 20, torch.arange(400))
    mse = (q * (g - grad).square().sum(-1)).sum()
    d = z.log_softmax(-1) - ref.log_softmax(-1)  # k2 gradient for draw y: d_y * (onehot(y) - q)
    mse_k2 = (q * (d[:, None] * (torch.eye(400, dtype=torch.float64) - q) - grad).square().sum(-1)).sum()
    assert mse < 0.5 * mse_k2


def test_padded_vocab_columns_count_as_tokens():
    """Padded LM-head columns with mass: unbiased for the KL over the full (padded) support, not the unpadded one."""
    z, ref = _logits(128, drift=1.0, seed=4, ref_scale=2.0)
    z[120:] += 3.0
    kl, grad = _exact(z, ref)
    assert abs(kl.item() - _exact(z[:120], ref[:120])[0].item()) > 0.05
    g_mean, v_mean = _expected(z, ref, 10, z.log_softmax(-1))
    assert abs(v_mean.item() - kl.item()) < 1e-6
    assert (g_mean - grad).norm() < 1e-5 * grad.norm()


def test_policy_masked_token_in_reference_topk_stays_finite():
    """The policy masks a token (-inf logit) that the reference ranks first: its 0 log 0 term must be 0, with
    finite gradients, rather than 0 * inf = NaN."""
    z = torch.randn(3, 20, dtype=torch.float64)
    z[:, 0] = float("-inf")
    ref = torch.randn(3, 20, dtype=torch.float64)
    ref[:, 0] = 10.0
    ref_lp, ref_ids = compute_topk_log_probs(ref, 4)
    assert (ref_ids == 0).any(-1).all()
    zz = z.clone().requires_grad_(True)
    coarse, log_q_tail = compute_coarse_reverse_kl(zz, ref_ids, ref_lp)
    (g,) = torch.autograd.grad((coarse + log_q_tail).sum(), zz)
    assert torch.isfinite(coarse).all() and torch.isfinite(log_q_tail).all() and torch.isfinite(g).all()


def test_off_policy_draw_of_a_policy_masked_token_contributes_only_the_coarse_term():
    """pi_old drew a token the current policy masks (-inf): its weight is 0, and r = -inf must not turn the loss,
    the value or the gradient into NaN."""
    z = torch.randn(1, 20, dtype=torch.float64)
    z[0, 7] = float("-inf")
    ref_lp, ref_ids = compute_topk_log_probs(torch.randn(1, 20, dtype=torch.float64), 3)
    assert not (ref_ids == 7).any()
    zz = z.clone().requires_grad_(True)
    coarse, log_q_tail = compute_coarse_reverse_kl(zz, ref_ids, ref_lp)
    log_prob = zz.log_softmax(-1)[:, 7]
    loss, value = compute_topk_rb_kl(
        coarse, log_q_tail, ref_lp[:, -1], torch.tensor([False]), log_prob, torch.tensor([-3.0]), torch.tensor([-2.0])
    )
    (g,) = torch.autograd.grad(loss.sum(), zz)
    torch.testing.assert_close(value, coarse.detach().double())
    assert torch.isfinite(loss).all() and torch.isfinite(g).all()


def test_topk_columns_marks_only_this_shards_ids():
    """Shard covers ids 10..19. Off-shard ids are clamped onto an edge column, here after a real id on that same
    column (3 -> column 0, 25 -> column 9), so a plain scatter would let them erase its mark."""
    ids = torch.tensor([[10, 13, 3], [19, 12, 25]])
    expected = torch.zeros(2, 10, dtype=torch.bool)
    expected[0, [0, 3]] = True
    expected[1, [9, 2]] = True
    assert torch.equal(topk_rb_kl._topk_columns(ids, vocab_start=10, vocab_size=10), expected)


def _vocab_parallel_worker(rank, world_size, rendezvous_file, out_dir):
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    try:
        logits, ref = _logits_and_ref()
        shard = logits.shape[-1] // world_size
        local = slice(rank * shard, (rank + 1) * shard)
        group = dist.group.WORLD
        ref_lp, ref_ids = compute_topk_log_probs(ref[:, local], 5, chunk_size=4, process_group=group)
        vp = logits[:, local].clone().requires_grad_(True)
        coarse, log_q_tail = compute_coarse_reverse_kl(vp, ref_ids, ref_lp, chunk_size=4, process_group=group)
        (coarse * _W_COARSE + log_q_tail * _W_TAIL).sum().backward()
        out = {"ref_lp": ref_lp, "ref_ids": ref_ids, "coarse": coarse.detach(), "log_q_tail": log_q_tail.detach()}
        torch.save({**out, "grad": vp.grad}, f"{out_dir}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _logits_and_ref():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 48, generator=g, dtype=torch.float64) * 2
    return logits, logits + torch.randn(6, 48, generator=g, dtype=torch.float64)


_W_COARSE, _W_TAIL = torch.linspace(0.5, 1.5, 6, dtype=torch.float64), torch.linspace(-1.0, 2.0, 6)


@pytest.mark.parametrize("world_size", [2, 3])
def test_vocab_parallel_matches_single_shard_oracle(tmp_path, world_size):
    """Sharded top-k (with tail) and sharded coarse KL, with gradients, == the single-device oracle."""
    mp.spawn(_vocab_parallel_worker, args=(world_size, str(tmp_path / "rdzv"), str(tmp_path)), nprocs=world_size)
    logits, ref = _logits_and_ref()
    full = logits.clone().requires_grad_(True)
    ref_lp, ref_ids = compute_topk_log_probs(ref, 5)
    coarse, log_q_tail = _coarse_oracle(full, ref_ids, ref)
    (coarse * _W_COARSE + log_q_tail * _W_TAIL).sum().backward()
    shards = [torch.load(tmp_path / f"rank{r}.pt") for r in range(world_size)]
    tol = dict(atol=1e-5, rtol=1e-5)
    for shard in shards:
        assert torch.equal(shard["ref_ids"], ref_ids)
        torch.testing.assert_close(shard["ref_lp"], ref_lp, **tol)
        torch.testing.assert_close(shard["coarse"].double(), coarse.detach(), **tol)
        torch.testing.assert_close(shard["log_q_tail"].double(), log_q_tail.detach(), **tol)
    torch.testing.assert_close(torch.cat([s["grad"] for s in shards], dim=-1), full.grad, **tol)
