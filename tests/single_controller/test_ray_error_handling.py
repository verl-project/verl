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

import pytest
from unittest.mock import MagicMock, patch

# Try to import ray, but don't fail if it's not available
try:
    import ray
except ImportError:
    ray = None

from verl.single_controller.ray.base import (
    detect_error_type,
    detect_oom_type,
    is_oom_error,
    aggregate_worker_errors,
    RayOOMError,
    RayCommunicationError
)


class TestRayErrorHandling:
    """Tests for Ray error handling utilities."""
    
    def test_detect_error_type(self):
        """Test error type detection for different error messages."""
        # Test device-side OOM
        assert detect_error_type("CUDA out of memory. Tried to allocate 20.00 MiB") == (True, "oom", "device")
        assert detect_error_type("NPU out of memory") == (True, "oom", "device")
        assert detect_error_type("torch.cuda.OutOfMemoryError: CUDA out of memory") == (True, "oom", "device")
        
        # Test host-side OOM
        assert detect_error_type("MemoryError: Unable to allocate 2.00 GiB") == (True, "oom", "host")
        assert detect_error_type("cannot allocate memory") == (True, "oom", "host")
        assert detect_error_type("malloc failed") == (True, "oom", "host")
        
        # Test general OOM
        assert detect_error_type("out of memory") == (True, "oom", "unknown")
        assert detect_error_type("OOM error") == (True, "oom", "unknown")
        
        # Test connection errors
        assert detect_error_type("Connection refused") == (True, "connection", "network")
        assert detect_error_type("timeout error") == (True, "connection", "network")
        assert detect_error_type("socket error") == (True, "connection", "network")
        
        # Test resource errors
        assert detect_error_type("No such file or directory") == (True, "resource", "filesystem")
        assert detect_error_type("Permission denied") == (True, "resource", "filesystem")
        assert detect_error_type("disk full") == (True, "resource", "filesystem")
        
        # Test unknown errors
        assert detect_error_type("ValueError: invalid value") == (False, "unknown", "unknown")
        assert detect_error_type("AssertionError: assertion failed") == (False, "unknown", "unknown")
    
    def test_detect_oom_type(self):
        """Test OOM type detection."""
        # Test device-side OOM
        assert detect_oom_type("CUDA out of memory") == (True, "device")
        
        # Test host-side OOM
        assert detect_oom_type("MemoryError") == (True, "host")
        
        # Test general OOM
        assert detect_oom_type("out of memory") == (True, "unknown")
        
        # Test non-OOM
        assert detect_oom_type("ValueError") == (False, "unknown")
    
    def test_is_oom_error(self):
        """Test OOM error detection."""
        assert is_oom_error("CUDA out of memory") is True
        assert is_oom_error("MemoryError") is True
        assert is_oom_error("out of memory") is True
        assert is_oom_error("ValueError") is False
    
    def test_aggregate_worker_errors(self):
        """Test error aggregation from multiple workers."""
        # Test no errors
        results = ["result1", "result2", "result3"]
        has_error, summary, details = aggregate_worker_errors(results)
        assert has_error is False
        assert summary == ""
        assert details == []
        
        # Test single error
        results = ["result1", ValueError("invalid value"), "result3"]
        has_error, summary, details = aggregate_worker_errors(results)
        assert has_error is True
        assert "1 workers (indices: [1]) encountered unknown error" in summary
        assert len(details) == 1
        assert details[0]["worker_index"] == 1
        
        # Test multiple errors of same type
        results = [
            "result1",
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory"),
            "result4"
        ]
        has_error, summary, details = aggregate_worker_errors(results)
        assert has_error is True
        assert "2 workers (indices: [1, 2]) encountered oom error (device)" in summary
        assert len(details) == 2
        
        # Test multiple errors of different types
        results = [
            RuntimeError("CUDA out of memory"),
            MemoryError("Unable to allocate"),
            ConnectionError("Connection refused"),
            ValueError("invalid value")
        ]
        has_error, summary, details = aggregate_worker_errors(results)
        assert has_error is True
        assert "1 workers (indices: [0]) encountered oom error (device)" in summary
        assert "1 workers (indices: [1]) encountered oom error (host)" in summary
        assert "1 workers (indices: [2]) encountered connection error (network)" in summary
        assert "1 workers (indices: [3]) encountered unknown error" in summary
        assert len(details) == 4
    
    def test_error_handling_integration(self):
        """Integration test for error handling logic without mocking RayWorkerGroup."""
        # Test the actual error aggregation logic directly
        # This tests the real error handling without mocking Ray-specific components
        
        from verl.single_controller.ray.base import RayOOMError, RayCommunicationError
        
        # Mock results that would come from ray.get() calls
        # Simulate a scenario where execute_all_sync would be called with these results
        
        # Test with OOM error
        mock_results = [
            "result1",
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory")
        ]
        
        # Test aggregation of these errors
        has_error, summary, details = aggregate_worker_errors(mock_results)
        assert has_error is True
        assert "2 workers (indices: [1, 2]) encountered oom error (device)" in summary
        assert len(details) == 2
        
        # Test with mixed errors
        mock_results = [
            RuntimeError("CUDA out of memory"),
            MemoryError("Unable to allocate"),
            "result3"
        ]
        
        has_error, summary, details = aggregate_worker_errors(mock_results)
        assert has_error is True
        assert "1 workers (indices: [0]) encountered oom error (device)" in summary
        assert "1 workers (indices: [1]) encountered oom error (host)" in summary
        assert len(details) == 2