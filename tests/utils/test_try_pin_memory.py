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

"""Unit tests for _try_pin_memory fallback in megatron_utils.

Tests that pin_memory() is attempted first and that a graceful
fallback to non-pinned CPU tensors occurs when pinning fails.
All tests run on CPU without GPU, distributed setup, or Megatron.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from verl.utils.megatron_utils import _try_pin_memory


class TestTryPinMemory:
    """Test the _try_pin_memory helper."""

    def test_returns_tensor_on_success(self):
        """When pin_memory succeeds, the returned tensor should be pinned."""
        t = torch.randn(4, 4)
        result = _try_pin_memory(t)
        # On machines without CUDA, pin_memory() may itself raise or
        # return the same tensor.  We just verify it does not crash and
        # returns a tensor with the same data.
        assert isinstance(result, torch.Tensor)
        assert result.shape == t.shape
        assert result.dtype == t.dtype
        assert torch.equal(result, t)

    def test_falls_back_on_failure(self):
        """When pin_memory raises, _try_pin_memory returns the original tensor."""
        t = torch.randn(8, 8)
        with patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("CUDA error: invalid argument")):
            result = _try_pin_memory(t)
        # Should be the exact same object (not a copy)
        assert result is t

    def test_falls_back_preserves_data(self):
        """Fallback tensor has identical contents to the input."""
        t = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        with patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("cudaErrorInvalidValue")):
            result = _try_pin_memory(t)
        assert torch.equal(result, t)
        assert result.dtype == torch.bfloat16

    def test_logs_warning_on_fallback(self):
        """A warning should be logged when falling back."""
        t = torch.randn(2, 2)
        with patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("out of pinned memory")):
            with patch("verl.utils.megatron_utils.logger") as mock_logger:
                _try_pin_memory(t)
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "pin_memory" in warning_msg

    def test_does_not_log_on_success(self):
        """No warning should be logged when pin_memory succeeds."""
        t = torch.randn(2, 2)
        # Patch pin_memory to succeed normally (return a new tensor)
        with patch.object(torch.Tensor, "pin_memory", return_value=torch.randn(2, 2)):
            with patch("verl.utils.megatron_utils.logger") as mock_logger:
                _try_pin_memory(t)
                mock_logger.warning.assert_not_called()

    def test_handles_empty_tensor(self):
        """Works correctly with zero-element tensors."""
        t = torch.empty(0, dtype=torch.float32)
        result = _try_pin_memory(t)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 0

    def test_handles_large_tensor_failure(self):
        """Simulates the real scenario: large tensor pin fails on resume."""
        # Create a tensor similar in spirit to the 1.4 GiB bf16 buffer in the issue
        t = torch.randn(1024, 1024, dtype=torch.bfloat16)
        with patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("CUDA error: invalid argument (cudaErrorInvalidValue)")):
            result = _try_pin_memory(t)
        assert result is t
        assert result.shape == (1024, 1024)
        assert result.dtype == torch.bfloat16
