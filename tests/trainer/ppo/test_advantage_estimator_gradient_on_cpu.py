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

"""Pins what each advantage estimator does to the *policy gradient*, exactly.

The existing estimator tests check shapes, registry mechanics, and that a
vectorized implementation matches its reference on random tensors. None of them
checks the property the estimators exist for: what gradient they produce in
expectation. A refactor can keep every shape correct, keep the vectorized and
reference paths agreeing, and still change the direction the policy moves.

This module closes that gap. It builds a policy small enough to enumerate
completely, so three quantities are available in closed form:

    g_true  = d/dtheta E[R]  = sum_tau P(tau) grad log P(tau) R(tau)
    g_est   = E over groups of the estimator's gradient contribution
    bias    = g_est - g_true

Nothing is sampled. The group expectation runs over every multiset of `G` draws
weighted by its multinomial probability, so a nonzero bias is a property of the
estimator and not Monte-Carlo noise. That matters here because these estimators
are variance-reduction devices: with sampling you cannot tell a biased estimator
from a noisy measurement of an unbiased one.

Three behaviours get pinned, and they are deliberately different assertions:

  * exactly unbiased -- RLOO and its vectorized twin, and REMAX with a zero
    baseline. Asserted at 1e-12.
  * unbiased in direction, rescaled by exactly (G-1)/G -- OPO, and GRPO with
    ``norm_adv_by_std_in_grpo=False`` (Dr. GRPO). A group mean that includes the
    sample shrinks the step but does not turn it. Asserted against the closed
    form for G = 2..6.
  * a pinned non-rescaling -- GRPO with std normalisation on, which is the
    default. The full expected gradient vector is asserted, and its
    per-coordinate ratio to the true gradient is confirmed to be non-constant.
    This records the documented Dr. GRPO trade-off without judging it as wrong.

``grpo_passk`` is deliberately not covered: it optimises pass@k rather than
E[R], so comparing it against the E[R] gradient would measure the objective
mismatch, not the estimator. ``gae`` is not covered either -- it takes a critic
rather than a prompt group, and with a zero value function it degenerates to
discounted return-to-go, which says nothing about the estimator as used.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_opo_outcome_advantage,
    compute_remax_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
)

# Enumeration is exponential in both, so keep them small; the whole module runs
# in a couple of seconds at these sizes.
RESPONSE_LEN = 3
THETA = np.array([0.4, -0.3, 0.2])
GROUP_SIZE = 4

EXACTLY_UNBIASED_TOL = 1e-12
CLOSED_FORM_TOL = 1e-9
EXPECTED_GRPO_STD_NORMALISED_GRADIENT = np.array([0.28770953939275357, 0.13498216697212817, -0.13699747277377736])


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _reward(tokens):
    """Asymmetric in the token index so a direction error cannot cancel out."""
    return float(2 * tokens[0] + tokens[1] - tokens[2])


class _EnumerablePolicy:
    """A response is RESPONSE_LEN binary tokens, each from its own logit.

    The policy factorises over positions, so every response and its exact
    probability and score function are enumerable.
    """

    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=np.float64)
        self.length = len(self.theta)
        self.responses = [np.array(c) for c in itertools.product((0, 1), repeat=self.length)]
        self.probs = np.array([self._prob(r) for r in self.responses])
        self.scores = np.array([self._score(r) for r in self.responses])
        self.rewards = np.array([_reward(r) for r in self.responses])
        assert abs(self.probs.sum() - 1.0) < 1e-12

    def _prob(self, tokens):
        out = 1.0
        for t, a in enumerate(tokens):
            p = _sigmoid(self.theta[t])
            out *= p if a == 1 else 1.0 - p
        return out

    def _score(self, tokens):
        """grad_theta log pi(response); position t only touches theta[t]."""
        g = np.zeros(self.length)
        for t, a in enumerate(tokens):
            p = _sigmoid(self.theta[t])
            g[t] = (1.0 - p) if a == 1 else -p
        return g

    def true_gradient(self):
        return (self.probs[:, None] * self.scores * self.rewards[:, None]).sum(axis=0)

    def finite_difference_gradient(self, eps=1e-6):
        """Independent check on true_gradient, used by the harness self-test."""
        out = np.zeros(self.length)
        for k in range(self.length):
            hi, lo = self.theta.copy(), self.theta.copy()
            hi[k] += eps
            lo[k] -= eps
            out[k] = (_EnumerablePolicy(hi).expected_reward() - _EnumerablePolicy(lo).expected_reward()) / (2 * eps)
        return out

    def expected_reward(self):
        return float((self.probs * self.rewards).sum())


def _make_batch(policy, combo):
    """One prompt group in verl's tensor contract; outcome reward on the last token."""
    g, length = len(combo), policy.length
    token_level_rewards = torch.zeros(g, length, dtype=torch.float64)
    for i, idx in enumerate(combo):
        token_level_rewards[i, length - 1] = float(policy.rewards[idx])
    response_mask = torch.ones(g, length, dtype=torch.float64)
    index = np.zeros(g, dtype=np.int64)
    return token_level_rewards, response_mask, index


def expected_estimator_gradient(policy, estimator, group_size=GROUP_SIZE, **kwargs):
    """E[estimator gradient], exactly, over every group multiset."""
    n = len(policy.responses)
    total = np.zeros(policy.length)
    weight = 0.0

    for combo in itertools.combinations_with_replacement(range(n), group_size):
        counts = {}
        for i in combo:
            counts[i] = counts.get(i, 0) + 1
        coeff = math.factorial(group_size)
        for c in counts.values():
            coeff //= math.factorial(c)
        prob = coeff * float(np.prod(policy.probs[list(combo)]))
        if prob == 0.0:
            continue
        weight += prob

        token_level_rewards, response_mask, index = _make_batch(policy, combo)
        call = dict(token_level_rewards=token_level_rewards, response_mask=response_mask, **kwargs)
        if estimator is not compute_remax_outcome_advantage:
            call["index"] = index
        else:
            call["reward_baselines"] = torch.zeros(group_size, dtype=torch.float64)
        advantages, _ = estimator(**call)
        advantages = advantages.detach().double().numpy()

        contribution = np.zeros(policy.length)
        for i, idx in enumerate(combo):
            # position t of member i carries advantage[i, t] and score scores[idx][t]
            contribution += advantages[i] * policy.scores[idx]
        total += prob * contribution

    assert abs(weight - 1.0) < 1e-12, weight
    return total / group_size


@pytest.fixture(scope="module")
def policy():
    return _EnumerablePolicy(THETA)


def test_harness_is_sound(policy):
    """The closed-form gradient agrees with a finite difference, and a
    no-baseline estimator measures as unbiased. If either fails, every other
    assertion in this module is meaningless."""
    np.testing.assert_allclose(policy.true_gradient(), policy.finite_difference_gradient(), atol=1e-8)

    def no_baseline(token_level_rewards, response_mask, index, **_):
        scores = token_level_rewards.sum(dim=-1, keepdim=True)
        advantages = scores * response_mask
        return advantages, advantages

    bias = expected_estimator_gradient(policy, no_baseline) - policy.true_gradient()
    assert np.abs(bias).max() < EXACTLY_UNBIASED_TOL, bias


@pytest.mark.parametrize(
    "estimator",
    [compute_rloo_outcome_advantage, compute_rloo_vectorized_outcome_advantage, compute_remax_outcome_advantage],
    ids=["rloo", "rloo_vectorized", "remax_zero_baseline"],
)
def test_leave_one_out_estimators_are_exactly_unbiased(policy, estimator):
    """A baseline built only from the other group members cannot bias the gradient."""
    bias = expected_estimator_gradient(policy, estimator) - policy.true_gradient()
    assert np.abs(bias).max() < EXACTLY_UNBIASED_TOL, f"{estimator.__name__} bias {bias}"


@pytest.mark.parametrize("group_size", [2, 3, 4, 5, 6])
def test_self_inclusive_group_mean_rescales_but_does_not_turn(policy, group_size):
    """OPO and Dr. GRPO subtract a group mean that includes the sample itself.

    That shrinks every advantage by exactly (G-1)/G, so the expected gradient is
    a positive multiple of the true one: the ascent direction is preserved and
    only the effective step size changes.
    """
    expected_ratio = (group_size - 1) / group_size
    g_true = policy.true_gradient()

    for estimator, kwargs in (
        (compute_opo_outcome_advantage, {}),
        (compute_grpo_outcome_advantage, {"norm_adv_by_std_in_grpo": False}),
    ):
        g_est = expected_estimator_gradient(policy, estimator, group_size=group_size, **kwargs)
        np.testing.assert_allclose(g_est, expected_ratio * g_true, atol=CLOSED_FORM_TOL, err_msg=estimator.__name__)


def test_grpo_std_normalisation_has_pinned_direction_and_scale(policy):
    """With std normalisation on -- the default -- pin the complete expected
    gradient, including both its direction and scale. Its ratio to the true
    gradient is non-constant; turning normalisation off recovers a pure scaling.

    The golden vector comes from the exact enumeration performed by this module,
    not from a sampled run. Pinning it ensures that a sign or scale regression
    cannot pass merely because the result remains a non-pure rescaling.
    """
    g_true = policy.true_gradient()

    ratio_off = (
        expected_estimator_gradient(policy, compute_grpo_outcome_advantage, norm_adv_by_std_in_grpo=False) / g_true
    )
    assert np.ptp(ratio_off) < CLOSED_FORM_TOL, ratio_off

    g_est_on = expected_estimator_gradient(policy, compute_grpo_outcome_advantage, norm_adv_by_std_in_grpo=True)
    np.testing.assert_allclose(
        g_est_on,
        EXPECTED_GRPO_STD_NORMALISED_GRADIENT,
        rtol=0.0,
        atol=CLOSED_FORM_TOL,
    )

    ratio_on = g_est_on / g_true
    assert np.ptp(ratio_on) > 1e-3, (
        f"std normalisation looks like a pure rescaling (ratios {ratio_on}); "
        "if this became true the Dr. GRPO distinction would have disappeared"
    )


@pytest.mark.parametrize(
    "reference,vectorized,kwargs",
    [
        (compute_rloo_outcome_advantage, compute_rloo_vectorized_outcome_advantage, {}),
        (compute_grpo_outcome_advantage, compute_grpo_vectorized_outcome_advantage, {}),
        (
            compute_grpo_outcome_advantage,
            compute_grpo_vectorized_outcome_advantage,
            {"norm_adv_by_std_in_grpo": False},
        ),
    ],
    ids=["rloo", "grpo_std_on", "grpo_std_off"],
)
def test_vectorized_variants_agree_in_expected_gradient(policy, reference, vectorized, kwargs):
    """Stronger than tensor equality on random inputs: the two paths must induce
    the same expected gradient, over the whole response distribution."""
    np.testing.assert_allclose(
        expected_estimator_gradient(policy, reference, **kwargs),
        expected_estimator_gradient(policy, vectorized, **kwargs),
        atol=EXACTLY_UNBIASED_TOL,
    )
