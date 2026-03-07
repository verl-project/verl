# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Test that temperature scaling in dp_actor handles activation-offloaded views.

FSDP2 + activation_offload wraps squeeze() in a custom autograd Function.
The resulting view cannot be modified inplace — PyTorch raises RuntimeError
because div_() would override the custom backward. This test verifies that
the conditional inplace/non-inplace division in _forward_micro_batch works
correctly in both grad-enabled and no-grad contexts.
"""

import unittest

import torch


class _MockActivationOffload(torch.autograd.Function):
    """Simulates the custom autograd Function that activation_offload wraps around squeeze().

    When activation_offload is active, squeeze() returns a view through a custom
    autograd Function (SqueezeBackward1). Inplace ops on such views raise RuntimeError.
    """

    @staticmethod
    def forward(ctx, x):
        return x.squeeze(0)

    @staticmethod
    def backward(ctx, grad):
        return grad.unsqueeze(0)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Local copy of dp_actor._apply_temperature (avoids importing verl which requires ray)."""
    if torch.is_grad_enabled():
        return logits / temperature
    return logits.div_(temperature)


class TestInplaceDivActivationOffload(unittest.TestCase):
    """Verify temperature scaling handles activation-offloaded views correctly."""

    def test_inplace_div_fails_on_activation_offload_view(self):
        """Inplace div_ on an activation-offloaded view raises RuntimeError."""
        x = torch.randn(1, 4, requires_grad=True)
        view = _MockActivationOffload.apply(x)  # simulates activation offload squeeze
        with self.assertRaises(RuntimeError):
            view.div_(2.0)

    def test_non_inplace_div_works_on_activation_offload_view(self):
        """Non-inplace division on an activation-offloaded view works."""
        x = torch.randn(1, 4, requires_grad=True)
        view = _MockActivationOffload.apply(x)
        result = view / 2.0  # non-inplace
        self.assertEqual(result.shape, (4,))
        self.assertTrue(torch.allclose(result, x.squeeze(0) / 2.0))

    def test_conditional_div_with_grad_enabled(self):
        """With grad enabled (training), uses non-inplace division."""
        x = torch.randn(1, 4, requires_grad=True)
        view = _MockActivationOffload.apply(x)
        # Should not raise — uses non-inplace path
        result = _apply_temperature(view, 0.7)
        self.assertTrue(torch.allclose(result, x.squeeze(0) / 0.7))

    def test_conditional_div_with_no_grad(self):
        """With no_grad (compute_log_prob), uses inplace division."""
        x = torch.randn(1, 4)
        expected = x.squeeze(0) / 0.7
        logits = x.squeeze(0)  # normal squeeze, no custom autograd
        with torch.no_grad():
            result = _apply_temperature(logits, 0.7)
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_div_safe_on_offload_view_under_no_grad(self):
        """Under no_grad, inplace div_ on an activation-offloaded view does not raise."""
        x = torch.randn(1, 4, requires_grad=True)
        view = _MockActivationOffload.apply(x)
        with torch.no_grad():
            view.div_(2.0)  # should NOT raise — autograd tracking is disabled

    def test_gradient_flows_through_non_inplace_path(self):
        """Non-inplace division preserves gradient flow through activation-offloaded views."""
        x = torch.randn(1, 4, requires_grad=True)
        view = _MockActivationOffload.apply(x)
        result = _apply_temperature(view, 0.7)
        result.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
