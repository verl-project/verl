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

import math
import unittest

import pytest
import torch

import verl.trainer.distillation.losses as losses_module
from verl.trainer.distillation.losses import (
    DISTILLATION_LOSS_REGISTRY,
    DISTILLATION_SETTINGS_REGISTRY,
    DistillationLossSettings,
    clamp_log_probs,
    compute_distillation_loss_kl_estimator,
    compute_distillation_loss_topk,
    get_distillation_loss_fn,
    get_distillation_loss_settings,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    register_distillation_loss,
)
from verl.workers.config import DistillationConfig, OptimizerConfig


def mock_distillation_loss_fn(**kwargs):
    """Mock distillation loss function for testing."""
    return torch.tensor(0.0), {}


class TestDistillationLossSettings(unittest.TestCase):
    """Test the DistillationLossSettings dataclass."""

    def test_single_name_converted_to_list(self):
        """Test that a single name string is converted to a list."""
        settings = DistillationLossSettings(names="test_loss")
        self.assertEqual(settings.names, ["test_loss"])

    def test_list_of_names_preserved(self):
        """Test that a list of names is preserved."""
        settings = DistillationLossSettings(names=["loss1", "loss2"])
        self.assertEqual(settings.names, ["loss1", "loss2"])

    def test_use_topk_derived_from_student_topk(self):
        """Test that use_topk is True when use_student_topk is True."""
        settings = DistillationLossSettings(names="test", use_student_topk=True)
        self.assertTrue(settings.use_topk)

    def test_use_topk_derived_from_teacher_topk(self):
        """Test that use_topk is True when use_teacher_topk is True."""
        settings = DistillationLossSettings(names="test", use_teacher_topk=True)
        self.assertTrue(settings.use_topk)

    def test_use_topk_derived_from_both(self):
        """Test that use_topk is True when both student and teacher topk are True."""
        settings = DistillationLossSettings(names="test", use_student_topk=True, use_teacher_topk=True)
        self.assertTrue(settings.use_topk)

    def test_mutually_exclusive_full_and_topk(self):
        """Test that use_full and use_topk cannot both be True."""
        with self.assertRaises(ValueError) as cm:
            DistillationLossSettings(names="test", use_full=True, use_student_topk=True)
        self.assertIn("only one of", str(cm.exception).lower())

    def test_mutually_exclusive_full_and_estimator(self):
        """Test that use_full and use_estimator cannot both be True."""
        with self.assertRaises(ValueError) as cm:
            DistillationLossSettings(names="test", use_full=True, use_estimator=True)
        self.assertIn("only one of", str(cm.exception).lower())

    def test_mutually_exclusive_topk_and_estimator(self):
        """Test that use_topk and use_estimator cannot both be True."""
        with self.assertRaises(ValueError) as cm:
            DistillationLossSettings(names="test", use_student_topk=True, use_estimator=True)
        self.assertIn("only one of", str(cm.exception).lower())

    def test_all_false_is_valid(self):
        """Test that all flags being False is valid."""
        settings = DistillationLossSettings(names="test")
        self.assertFalse(settings.use_full)
        self.assertFalse(settings.use_topk)
        self.assertFalse(settings.use_estimator)


class TestRegisterDistillationLoss(unittest.TestCase):
    """Test the distillation loss registry functions."""

    def setUp(self):
        """Save the original registry state before each test."""
        self.original_loss_registry = DISTILLATION_LOSS_REGISTRY.copy()
        self.original_settings_registry = DISTILLATION_SETTINGS_REGISTRY.copy()

    def tearDown(self):
        """Restore the original registry state after each test."""
        losses_module.DISTILLATION_LOSS_REGISTRY.clear()
        losses_module.DISTILLATION_LOSS_REGISTRY.update(self.original_loss_registry)
        losses_module.DISTILLATION_SETTINGS_REGISTRY.clear()
        losses_module.DISTILLATION_SETTINGS_REGISTRY.update(self.original_settings_registry)

    def test_register_new_function(self):
        """Test registering a new distillation loss function."""
        settings = DistillationLossSettings(names="test_loss")

        @register_distillation_loss(settings)
        def test_fn(**kwargs):
            pass

        self.assertIn("test_loss", DISTILLATION_LOSS_REGISTRY)
        self.assertEqual(DISTILLATION_LOSS_REGISTRY["test_loss"], test_fn)
        self.assertIn("test_loss", DISTILLATION_SETTINGS_REGISTRY)
        self.assertEqual(DISTILLATION_SETTINGS_REGISTRY["test_loss"], settings)

    def test_register_with_multiple_names(self):
        """Test registering a function with multiple names."""
        settings = DistillationLossSettings(names=["alias1", "alias2"])

        @register_distillation_loss(settings)
        def test_fn(**kwargs):
            pass

        self.assertIn("alias1", DISTILLATION_LOSS_REGISTRY)
        self.assertIn("alias2", DISTILLATION_LOSS_REGISTRY)
        self.assertEqual(DISTILLATION_LOSS_REGISTRY["alias1"], test_fn)
        self.assertEqual(DISTILLATION_LOSS_REGISTRY["alias2"], test_fn)

    def test_duplicate_registration_raises_error(self):
        """Test that registering the same name twice raises ValueError."""
        settings1 = DistillationLossSettings(names="duplicate_test")
        settings2 = DistillationLossSettings(names="duplicate_test")

        @register_distillation_loss(settings1)
        def test_fn1(**kwargs):
            pass

        with self.assertRaises(ValueError) as cm:

            @register_distillation_loss(settings2)
            def test_fn2(**kwargs):
                pass

        self.assertIn("already registered", str(cm.exception))

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function."""
        settings = DistillationLossSettings(names="preserve_test")

        def test_fn(**kwargs):
            return "original"

        decorated = register_distillation_loss(settings)(test_fn)
        self.assertEqual(decorated(), "original")

    def test_get_distillation_loss_fn_valid_name(self):
        """Test retrieving a registered loss function."""
        # Test with a known registered loss
        loss_fn = get_distillation_loss_fn("kl")
        self.assertEqual(loss_fn, compute_distillation_loss_kl_estimator)

    def test_get_distillation_loss_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_distillation_loss_fn("invalid_loss_name")
        assert "Unsupported loss mode" in str(excinfo.value)

    def test_get_distillation_loss_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_distillation_loss_fn("KL")  # Should be lowercase

    def test_get_distillation_loss_settings_valid_name(self):
        """Test retrieving settings for a registered loss."""
        settings = get_distillation_loss_settings("kl")
        self.assertIsInstance(settings, DistillationLossSettings)
        self.assertTrue(settings.use_estimator)

    def test_get_distillation_loss_settings_invalid_name(self):
        """Test that invalid names raise ValueError for settings."""
        with pytest.raises(ValueError) as excinfo:
            get_distillation_loss_settings("invalid_loss_name")
        assert "Unsupported loss mode" in str(excinfo.value)


class TestClampLogProbs(unittest.TestCase):
    """Test the clamp_log_probs utility function."""

    def test_normal_values_unchanged(self):
        """Test that normal log prob values are not changed."""
        log_p = torch.tensor([-1.0, -2.0, -3.0])
        log_q = torch.tensor([-1.5, -2.5, -3.5])

        log_p_clamped, log_q_clamped = clamp_log_probs(log_p, log_q)

        torch.testing.assert_close(log_p_clamped, log_p)
        torch.testing.assert_close(log_q_clamped, log_q)

    def test_negative_inf_clamped(self):
        """Test that -inf values are clamped to min_log_prob."""
        log_p = torch.tensor([-1.0, float("-inf"), -3.0])
        log_q = torch.tensor([float("-inf"), -2.0, -3.0])
        eps = 1e-8

        log_p_clamped, log_q_clamped = clamp_log_probs(log_p, log_q, eps=eps)

        min_log_prob = math.log(eps)
        torch.testing.assert_close(log_p_clamped[1].item(), min_log_prob)
        torch.testing.assert_close(log_q_clamped[0].item(), min_log_prob)

    def test_very_small_values_clamped(self):
        """Test that very small values below eps are clamped."""
        eps = 1e-8
        min_log_prob = math.log(eps)
        very_small = min_log_prob - 10  # Much smaller than min

        log_p = torch.tensor([very_small, -1.0])
        log_q = torch.tensor([-1.0, very_small])

        log_p_clamped, log_q_clamped = clamp_log_probs(log_p, log_q, eps=eps)

        # Add offset to allow for small numerical errors in log calculations
        self.assertGreaterEqual(log_p_clamped[0].item(), min_log_prob - 1e-3)
        self.assertGreaterEqual(log_q_clamped[1].item(), min_log_prob - 1e-3)

    def test_custom_eps(self):
        """Test clamping with custom epsilon value."""
        log_p = torch.tensor([float("-inf")])
        log_q = torch.tensor([float("-inf")])
        eps = 1e-4

        log_p_clamped, log_q_clamped = clamp_log_probs(log_p, log_q, eps=eps)

        expected_min = math.log(eps)
        self.assertAlmostEqual(log_p_clamped[0].item(), expected_min, places=5)
        self.assertAlmostEqual(log_q_clamped[0].item(), expected_min, places=5)


class TestKullbackLeiblerDivergence(unittest.TestCase):
    """Test the kullback_leibler_divergence function."""

    def test_forward_kl_identical_distributions(self):
        """Test that forward KL divergence is 0 for identical distributions."""
        log_p = torch.tensor([[-1.0, -2.0, -0.5]])
        log_q = log_p.clone()

        kl = kullback_leibler_divergence(log_q, log_p, loss_mode="forward")

        self.assertAlmostEqual(kl.item(), 0.0, places=5)

    def test_reverse_kl_identical_distributions(self):
        """Test that reverse KL divergence is 0 for identical distributions."""
        log_p = torch.tensor([[-1.0, -2.0, -0.5]])
        log_q = log_p.clone()

        kl = kullback_leibler_divergence(log_q, log_p, loss_mode="reverse")

        self.assertAlmostEqual(kl.item(), 0.0, places=5)

    def test_forward_kl_non_negative(self):
        """Test that forward KL divergence is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            logits_p = torch.randn(1, 10)
            logits_q = torch.randn(1, 10)
            log_p = torch.log_softmax(logits_p, dim=-1)
            log_q = torch.log_softmax(logits_q, dim=-1)

            kl = kullback_leibler_divergence(log_q, log_p, loss_mode="forward")

            self.assertGreaterEqual(kl.item(), -1e-6)  # Allow small numerical error

    def test_reverse_kl_non_negative(self):
        """Test that reverse KL divergence is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            logits_p = torch.randn(1, 10)
            logits_q = torch.randn(1, 10)
            log_p = torch.log_softmax(logits_p, dim=-1)
            log_q = torch.log_softmax(logits_q, dim=-1)

            kl = kullback_leibler_divergence(log_q, log_p, loss_mode="reverse")

            self.assertGreaterEqual(kl.item(), -1e-6)

    def test_invalid_mode_raises_error(self):
        """Test that invalid loss mode raises ValueError."""
        log_p = torch.tensor([[-1.0, -2.0]])
        log_q = torch.tensor([[-1.5, -2.5]])

        with self.assertRaises(ValueError) as cm:
            kullback_leibler_divergence(log_q, log_p, loss_mode="invalid")
        self.assertIn("Unsupported loss mode", str(cm.exception))

    def test_batch_computation(self):
        """Test KL divergence computation over a batch."""
        batch_size = 4
        vocab_size = 100
        torch.manual_seed(42)

        logits_p = torch.randn(batch_size, vocab_size)
        logits_q = torch.randn(batch_size, vocab_size)
        log_p = torch.log_softmax(logits_p, dim=-1)
        log_q = torch.log_softmax(logits_q, dim=-1)

        kl = kullback_leibler_divergence(log_q, log_p, loss_mode="forward")

        self.assertEqual(kl.shape, (batch_size,))


class TestJensenShannonDivergence(unittest.TestCase):
    """Test the jensen_shannon_divergence function."""

    def test_jsd_identical_distributions(self):
        """Test that JSD is 0 for identical distributions."""
        log_p = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
        log_q = log_p.clone()

        jsd = jensen_shannon_divergence(log_q, log_p, beta=0.5)

        self.assertAlmostEqual(jsd.item(), 0.0, places=5)

    def test_jsd_symmetric_at_half_beta(self):
        """Test that JSD is symmetric when beta=0.5."""
        torch.manual_seed(42)
        logits_p = torch.randn(1, 10)
        logits_q = torch.randn(1, 10)
        log_p = torch.log_softmax(logits_p, dim=-1)
        log_q = torch.log_softmax(logits_q, dim=-1)

        jsd_pq = jensen_shannon_divergence(log_q, log_p, beta=0.5)
        jsd_qp = jensen_shannon_divergence(log_p, log_q, beta=0.5)

        torch.testing.assert_close(jsd_pq, jsd_qp, atol=1e-5, rtol=1e-5)

    def test_jsd_non_negative(self):
        """Test that JSD is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            logits_p = torch.randn(1, 10)
            logits_q = torch.randn(1, 10)
            log_p = torch.log_softmax(logits_p, dim=-1)
            log_q = torch.log_softmax(logits_q, dim=-1)

            for beta in [0.1, 0.5, 0.9]:
                jsd = jensen_shannon_divergence(log_q, log_p, beta=beta)
                self.assertGreaterEqual(jsd.item(), -1e-6)

    def test_jsd_bounded(self):
        """Test that JSD is bounded above by ln(2) for beta=0.5."""
        torch.manual_seed(42)
        for _ in range(10):
            logits_p = torch.randn(1, 10)
            logits_q = torch.randn(1, 10)
            log_p = torch.log_softmax(logits_p, dim=-1)
            log_q = torch.log_softmax(logits_q, dim=-1)

            jsd = jensen_shannon_divergence(log_q, log_p, beta=0.5)

            # JSD is bounded by ln(2) for beta=0.5
            self.assertLessEqual(jsd.item(), math.log(2) + 1e-5)

    def test_jsd_batch_computation(self):
        """Test JSD computation over a batch."""
        batch_size = 4
        vocab_size = 100
        torch.manual_seed(42)

        logits_p = torch.randn(batch_size, vocab_size)
        logits_q = torch.randn(batch_size, vocab_size)
        log_p = torch.log_softmax(logits_p, dim=-1)
        log_q = torch.log_softmax(logits_q, dim=-1)

        jsd = jensen_shannon_divergence(log_q, log_p, beta=0.5)

        self.assertEqual(jsd.shape, (batch_size,))


def _create_distillation_config(
    loss_mode: str = "k3",
    topk: int = 128,
    enabled: bool = True,
    jsd_beta: float = 0.5,
    loss_clamp: float = None,
) -> DistillationConfig:
    """Helper to create a DistillationConfig for testing."""
    optim = OptimizerConfig(lr=0.1)
    config = DistillationConfig(
        strategy="fsdp",
        loss_mode=loss_mode,
        topk=topk,
        enabled=enabled,
        jsd_beta=jsd_beta,
        loss_clamp=loss_clamp,
        optim=optim,
        use_dynamic_bsz=True,
        rollout_n=1,
    )
    # Set loss_settings as would happen at runtime
    config.loss_settings = get_distillation_loss_settings(loss_mode)
    for k, v in [("dp_size", 1), ("batch_num_tokens", None), ("global_batch_size", None), ("loss_scale_factor", None)]:
        config.global_batch_info[k] = v
    return config


@pytest.mark.parametrize("loss_mode", ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"])
def test_compute_distillation_loss_kl_estimator_modes(loss_mode: str):
    """Test KL estimator loss computation for all supported modes."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16

    teacher_log_probs = torch.randn(batch_size, seq_len)
    student_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    config = _create_distillation_config(loss_mode=loss_mode)

    loss, metrics = compute_distillation_loss_kl_estimator(
        teacher_log_probs=teacher_log_probs,
        student_log_probs=student_log_probs,
        response_mask=response_mask,
        config=config,
        loss_agg_mode="token-mean",
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert "distillation/loss_min" in metrics
    assert "distillation/loss_max" in metrics
    print(f"[KL Estimator] mode={loss_mode} loss={loss.item():.4f}")


@pytest.mark.parametrize(
    "loss_mode,use_student_topk,use_teacher_topk",
    [
        ("forward_kl_topk", False, True),
        ("reverse_kl_topk", True, False),
        ("jsd_topk", True, True),
    ],
)
def test_compute_distillation_loss_topk_modes(loss_mode: str, use_student_topk: bool, use_teacher_topk: bool):
    """Test top-k loss computation for all supported modes."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16
    topk = 32

    # Determine expected number of logprobs based on settings
    if use_student_topk and use_teacher_topk:
        num_logprobs = 2 * topk
    else:
        num_logprobs = topk

    # Create matching topk indices (required by the function)
    topk_indices = torch.randint(0, 1000, (batch_size, seq_len, num_logprobs))

    # Create log probabilities
    teacher_log_probs = torch.randn(batch_size, seq_len)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_topk_logprobs = torch.log_softmax(torch.randn(batch_size, seq_len, num_logprobs), dim=-1)
    student_topk_logprobs = torch.log_softmax(torch.randn(batch_size, seq_len, num_logprobs), dim=-1)

    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    config = _create_distillation_config(loss_mode=loss_mode, topk=topk)

    loss, metrics = compute_distillation_loss_topk(
        teacher_log_probs=teacher_log_probs,
        student_log_probs=student_log_probs,
        teacher_topk_logprobs=teacher_topk_logprobs,
        student_topk_logprobs=student_topk_logprobs,
        teacher_topk_indices=topk_indices,
        student_topk_indices=topk_indices,  # Must match teacher indices
        response_mask=response_mask,
        config=config,
        loss_agg_mode="token-mean",
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert "distillation/loss_min" in metrics
    assert "distillation/loss_max" in metrics
    assert "distillation/student_mass" in metrics
    assert "distillation/teacher_mass" in metrics
    print(f"[Top-K] mode={loss_mode} loss={loss.item():.4f}")


def test_compute_distillation_loss_topk_mismatched_indices_raises():
    """Test that mismatched teacher/student indices raise an error."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    topk = 16

    teacher_topk_indices = torch.randint(0, 100, (batch_size, seq_len, topk))
    student_topk_indices = torch.randint(0, 100, (batch_size, seq_len, topk))  # Different indices

    teacher_log_probs = torch.randn(batch_size, seq_len)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_topk_logprobs = torch.randn(batch_size, seq_len, topk)
    student_topk_logprobs = torch.randn(batch_size, seq_len, topk)
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    config = _create_distillation_config(loss_mode="forward_kl_topk", topk=topk)

    with pytest.raises(ValueError) as excinfo:
        compute_distillation_loss_topk(
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            student_topk_logprobs=student_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
            student_topk_indices=student_topk_indices,
            response_mask=response_mask,
            config=config,
        )
    assert "same" in str(excinfo.value).lower()


def test_compute_distillation_loss_topk_wrong_shape_raises():
    """Test that wrong logprob shapes raise an error."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    topk = 16
    wrong_topk = 8  # Wrong size

    topk_indices = torch.randint(0, 100, (batch_size, seq_len, topk))

    teacher_log_probs = torch.randn(batch_size, seq_len)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_topk_logprobs = torch.randn(batch_size, seq_len, wrong_topk)  # Wrong shape
    student_topk_logprobs = torch.randn(batch_size, seq_len, wrong_topk)
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    config = _create_distillation_config(loss_mode="forward_kl_topk", topk=topk)

    with pytest.raises(ValueError) as excinfo:
        compute_distillation_loss_topk(
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            student_topk_logprobs=student_topk_logprobs,
            teacher_topk_indices=topk_indices,
            student_topk_indices=topk_indices,
            response_mask=response_mask,
            config=config,
        )
    assert "shape" in str(excinfo.value).lower()


def test_compute_distillation_loss_with_loss_clamp():
    """Test that loss clamping is applied correctly."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16
    loss_clamp = 0.1

    # Create inputs that will produce high loss values
    teacher_log_probs = torch.zeros(batch_size, seq_len)
    student_log_probs = torch.ones(batch_size, seq_len) * -10  # Very different
    response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    config = _create_distillation_config(loss_mode="kl", loss_clamp=loss_clamp)

    loss, _ = compute_distillation_loss_kl_estimator(
        teacher_log_probs=teacher_log_probs,
        student_log_probs=student_log_probs,
        response_mask=response_mask,
        config=config,
    )

    # Loss should be bounded by the clamp value (approximately, due to mean aggregation)
    assert loss.item() <= loss_clamp + 1e-5


def test_registered_losses_exist():
    """Test that expected loss functions are registered."""
    # KL estimator losses
    kl_estimator_names = ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"]
    for name in kl_estimator_names:
        assert name in DISTILLATION_LOSS_REGISTRY, f"Expected '{name}' to be registered"
        assert name in DISTILLATION_SETTINGS_REGISTRY, f"Expected settings for '{name}' to be registered"

    # Top-k losses
    topk_names = ["forward_kl_topk", "reverse_kl_topk", "jsd_topk"]
    for name in topk_names:
        assert name in DISTILLATION_LOSS_REGISTRY, f"Expected '{name}' to be registered"
        assert name in DISTILLATION_SETTINGS_REGISTRY, f"Expected settings for '{name}' to be registered"


if __name__ == "__main__":
    unittest.main()
