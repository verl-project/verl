# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""
Unit tests for DPO loss functions.

Tests verify:
- Registry pattern works (lookup, registration, error handling)
- All 7 loss variants produce valid outputs
- Loss decreases when preference is correct vs incorrect
- Gradient flows through all losses
- Hyperparameter effects (beta, label_smoothing, reference_free)
"""

import math

import pytest
import torch

from verl.trainer.dpo.dpo_losses import DPO_LOSS_REGISTRY, DPOLossConfig, get_dpo_loss_fn, register_dpo_loss

# ─── Test Data ───────────────────────────────────────────────────────────

BATCH_SIZE = 4
BETA = 0.1

ALL_LOSS_NAMES = list(DPO_LOSS_REGISTRY.keys())


def _make_logps(correct_preference=True):
    """Create test log-probabilities.

    If correct_preference=True: chosen has higher log-prob ratio than rejected.
    If correct_preference=False: rejected has higher log-prob ratio.
    """
    if correct_preference:
        chosen_logps = torch.tensor([-1.0, -0.5, -1.5, -0.8], requires_grad=True)
        rejected_logps = torch.tensor([-3.0, -2.5, -4.0, -3.5])
    else:
        chosen_logps = torch.tensor([-3.0, -2.5, -4.0, -3.5], requires_grad=True)
        rejected_logps = torch.tensor([-1.0, -0.5, -1.5, -0.8])

    ref_chosen_logps = torch.tensor([-1.5, -1.0, -2.0, -1.2])
    ref_rejected_logps = torch.tensor([-2.5, -2.0, -3.5, -3.0])

    return chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps


# ─── Registry Tests ──────────────────────────────────────────────────────


class TestDPOLossRegistry:
    def test_all_expected_losses_registered(self):
        """All 7 loss variants should be in the registry."""
        expected = {"sigmoid", "robust", "ipo", "hinge", "exo_pair", "aot_pair", "aot"}
        assert expected.issubset(set(DPO_LOSS_REGISTRY.keys())), (
            f"Missing losses: {expected - set(DPO_LOSS_REGISTRY.keys())}"
        )

    def test_get_dpo_loss_fn_returns_callable(self):
        """get_dpo_loss_fn should return a callable."""
        fn = get_dpo_loss_fn("sigmoid")
        assert callable(fn)

    def test_unknown_loss_raises_error(self):
        """Unknown loss name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown DPO loss"):
            get_dpo_loss_fn("nonexistent_loss")

    def test_custom_loss_registration(self):
        """Custom losses can be registered via decorator."""

        @register_dpo_loss("test_custom")
        def my_custom_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, **kwargs):
            return torch.tensor(0.0), torch.zeros(1), torch.zeros(1)

        fn = get_dpo_loss_fn("test_custom")
        assert fn is my_custom_loss

        # Cleanup
        del DPO_LOSS_REGISTRY["test_custom"]


# ─── Correctness Tests (all variants) ───────────────────────────────────


class TestDPOLossOutputs:
    """Test that all loss variants produce valid outputs."""

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_returns_three_tensors(self, loss_name):
        """Every loss should return (loss, chosen_rewards, rejected_rewards)."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        loss, c_rewards, r_rewards = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)

        assert isinstance(loss, torch.Tensor), f"{loss_name}: loss should be a tensor"
        assert isinstance(c_rewards, torch.Tensor), f"{loss_name}: chosen_rewards should be a tensor"
        assert isinstance(r_rewards, torch.Tensor), f"{loss_name}: rejected_rewards should be a tensor"

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_loss_is_scalar(self, loss_name):
        """Loss should be a scalar (0-dim tensor)."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        loss, _, _ = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        assert loss.dim() == 0, f"{loss_name}: loss should be scalar, got shape {loss.shape}"

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_loss_is_finite(self, loss_name):
        """Loss should be finite (not NaN or Inf)."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        loss, _, _ = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        assert torch.isfinite(loss), f"{loss_name}: loss is not finite: {loss.item()}"

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_rewards_shape_matches_batch(self, loss_name):
        """Rewards should have shape (batch_size,)."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        _, c_rewards, r_rewards = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        assert c_rewards.shape == (BATCH_SIZE,), f"{loss_name}: chosen_rewards shape {c_rewards.shape}"
        assert r_rewards.shape == (BATCH_SIZE,), f"{loss_name}: rejected_rewards shape {r_rewards.shape}"

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_rewards_are_detached(self, loss_name):
        """Rewards should not require gradients (they're for metrics only)."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        _, c_rewards, r_rewards = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        assert not c_rewards.requires_grad, f"{loss_name}: chosen_rewards should be detached"
        assert not r_rewards.requires_grad, f"{loss_name}: rejected_rewards should be detached"


# ─── Gradient Tests ──────────────────────────────────────────────────────


class TestDPOLossGradients:
    """Test that gradients flow correctly through all losses."""

    @pytest.mark.parametrize("loss_name", ALL_LOSS_NAMES)
    def test_gradient_flows(self, loss_name):
        """Loss should produce non-zero gradients for chosen_logps."""
        fn = get_dpo_loss_fn(loss_name)
        chosen, rejected, ref_chosen, ref_rejected = _make_logps()
        loss, _, _ = fn(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        loss.backward()
        assert chosen.grad is not None, f"{loss_name}: no gradient for chosen_logps"
        assert not torch.all(chosen.grad == 0), f"{loss_name}: gradient is all zeros"


# ─── Preference Direction Tests ──────────────────────────────────────────


class TestDPOLossPreference:
    """Test that losses are lower when preference is correct."""

    @pytest.mark.parametrize(
        "loss_name",
        [
            "sigmoid",
            "ipo",
            "hinge",
        ],
    )
    def test_correct_preference_has_lower_loss(self, loss_name):
        """Loss should be lower when chosen is actually preferred."""
        fn = get_dpo_loss_fn(loss_name)

        # Correct preference: chosen >> rejected
        correct_logps = _make_logps(correct_preference=True)
        loss_correct, _, _ = fn(*correct_logps, beta=BETA)

        # Wrong preference: rejected >> chosen
        wrong_logps = _make_logps(correct_preference=False)
        loss_wrong, _, _ = fn(*wrong_logps, beta=BETA)

        assert loss_correct < loss_wrong, (
            f"{loss_name}: correct preference loss ({loss_correct.item():.4f}) "
            f"should be less than wrong ({loss_wrong.item():.4f})"
        )


# ─── Hyperparameter Tests ───────────────────────────────────────────────


class TestDPOLossHyperparameters:
    """Test hyperparameter effects."""

    def test_sigmoid_label_smoothing_increases_loss(self):
        """Label smoothing should increase loss for correct preferences."""
        fn = get_dpo_loss_fn("sigmoid")
        logps = _make_logps(correct_preference=True)

        loss_no_smooth, _, _ = fn(*logps, beta=BETA, label_smoothing=0.0)
        loss_with_smooth, _, _ = fn(*logps, beta=BETA, label_smoothing=0.1)

        assert loss_with_smooth > loss_no_smooth, "Label smoothing should increase loss"

    def test_reference_free_ignores_ref_model(self):
        """With reference_free=True, changing ref logps should not affect loss."""
        fn = get_dpo_loss_fn("sigmoid")
        chosen = torch.tensor([-1.0, -0.5], requires_grad=True)
        rejected = torch.tensor([-3.0, -2.5])
        ref1 = torch.tensor([-1.5, -1.0])
        ref2 = torch.tensor([-100.0, -100.0])  # Very different ref

        loss1, _, _ = fn(chosen, rejected, ref1, ref1, beta=BETA, reference_free=True)
        # Need fresh chosen for second call
        chosen2 = torch.tensor([-1.0, -0.5], requires_grad=True)
        loss2, _, _ = fn(chosen2, rejected, ref2, ref2, beta=BETA, reference_free=True)

        assert torch.allclose(loss1, loss2, atol=1e-6), (
            f"reference_free should ignore ref model. Loss1={loss1.item()}, Loss2={loss2.item()}"
        )

    def test_higher_beta_increases_sigmoid_sensitivity(self):
        """Higher beta should make sigmoid loss more sensitive to log-ratio differences."""
        fn = get_dpo_loss_fn("sigmoid")
        logps = _make_logps(correct_preference=True)

        loss_low_beta, _, _ = fn(*logps, beta=0.01)
        loss_high_beta, _, _ = fn(*logps, beta=1.0)

        # With correct preferences and higher beta, loss should be lower (model is "more right")
        assert loss_high_beta < loss_low_beta, (
            f"Higher beta should give lower loss for correct preferences. "
            f"Low beta: {loss_low_beta.item()}, High beta: {loss_high_beta.item()}"
        )


# ─── Exact values ────────────────────────────────────────────────────────


def _log_sigmoid(x: float) -> float:
    return -math.log1p(math.exp(-x))


class TestDPOLossExactValues:
    """One pair whose logits are (-1.0 + 1.5) - (-3.0 + 2.5) = 1.0, checked against each formula by hand."""

    PAIR = (
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-1.5]),
        torch.tensor([-2.5]),
    )

    @pytest.mark.parametrize(
        "loss_name, kwargs, expected",
        [
            ("sigmoid", {}, -_log_sigmoid(BETA * 1.0)),
            ("sigmoid", {"label_smoothing": 0.1}, -_log_sigmoid(0.1) * 0.9 - _log_sigmoid(-0.1) * 0.1),
            ("robust", {"label_smoothing": 0.1}, (-_log_sigmoid(0.1) * 0.9 + _log_sigmoid(-0.1) * 0.1) / 0.8),
            ("ipo", {}, (1.0 - 1 / (2 * BETA)) ** 2),
            ("hinge", {}, max(0.0, 1 - BETA * 1.0)),
        ],
    )
    def test_matches_formula(self, loss_name, kwargs, expected):
        loss, _, _ = get_dpo_loss_fn(loss_name)(*self.PAIR, beta=BETA, **kwargs)
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_rewards(self):
        _, chosen_rewards, rejected_rewards = get_dpo_loss_fn("sigmoid")(*self.PAIR, beta=BETA)
        assert chosen_rewards.item() == pytest.approx(BETA * 0.5)
        assert rejected_rewards.item() == pytest.approx(BETA * -0.5)

    def test_equal_policy_and_reference_gives_ln2(self):
        """At initialisation the policy equals the reference, so the standard loss is ln 2."""
        logps = torch.tensor([-2.0, -7.0])
        loss, _, _ = get_dpo_loss_fn("sigmoid")(logps, logps.flip(0), logps, logps.flip(0), beta=BETA)
        assert loss.item() == pytest.approx(math.log(2))


class TestDPOLossConfig:
    def test_defaults(self):
        config = DPOLossConfig()
        assert (config.loss_type, config.beta, config.label_smoothing, config.reference_free) == (
            "sigmoid",
            0.1,
            0.0,
            False,
        )

    @pytest.mark.parametrize("kwargs", [{"beta": 0.0}, {"label_smoothing": 0.5}, {"label_smoothing": -0.1}])
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            DPOLossConfig(**kwargs)
