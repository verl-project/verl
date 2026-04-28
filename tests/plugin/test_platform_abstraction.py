# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the platform abstraction layer.

We pre-populate ``sys.modules`` with lightweight stubs for the top-level
``verl`` and ``verl.plugin`` packages so that importing the platform
sub-package does **not** trigger the heavy dependency chain pulled in by
``verl/__init__.py`` (ray, tensordict, …).
"""

import os
import sys
from types import ModuleType

# ---------------------------------------------------------------------------
# Bootstrap: prevent verl.__init__ from executing (it imports ray, etc.)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _pkg, _rel in [("verl", "verl"), ("verl.plugin", os.path.join("verl", "plugin"))]:
    if _pkg not in sys.modules:
        _stub = ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_project_root, _rel)]
        _stub.__package__ = _pkg
        sys.modules[_pkg] = _stub

# ---------------------------------------------------------------------------

from unittest import mock  # noqa: E402

import pytest  # noqa: E402

from verl.plugin.platform import get_platform, set_platform  # noqa: E402
from verl.plugin.platform.platform_manager import (  # noqa: E402
    _create_platform,
    _detect_platform_name,
)


class TestPlatformDetection:
    """Test platform auto-detection logic."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_verl_platform_env_override_cuda(self):
        """Test VERL_PLATFORM environment variable override for CUDA."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cuda"}):
            name = _detect_platform_name()
            assert name == "cuda", f"Expected 'cuda', got '{name}'"

    def test_verl_platform_env_override_npu(self):
        """Test VERL_PLATFORM environment variable override for NPU."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "npu"}):
            name = _detect_platform_name()
            assert name == "npu", f"Expected 'npu', got '{name}'"

    def test_verl_platform_env_override_cpu_rejected(self):
        """Test that VERL_PLATFORM=cpu is rejected (CPU backend not supported)."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cpu"}):
            # "cpu" is not in _BUILTIN_PLATFORMS, so it falls back to auto-detection
            name = _detect_platform_name()
            assert name in ("cuda", "npu"), f"Expected 'cuda' or 'npu', got '{name}'"

    def test_verl_platform_invalid_value(self):
        """Test that invalid VERL_PLATFORM values fall back to auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            name = _detect_platform_name()
            assert name in ("cuda", "npu"), f"Got invalid platform name: {name}"

    def test_verl_platform_case_insensitive(self):
        """Test that VERL_PLATFORM is case-insensitive."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "CUDA"}):
            name = _detect_platform_name()
            assert name == "cuda", f"Expected 'cuda', got '{name}'"


class TestPlatformCreation:
    """Test platform creation and initialization."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_create_cuda_platform_raises_if_unavailable(self):
        """Test that CUDA platform raises RuntimeError if not available."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                _create_platform("cuda")

    def test_create_invalid_platform(self):
        """Test that invalid platform name raises ValueError."""
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformInterface:
    """Test platform interface methods."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        """Test that get_platform() returns a singleton."""
        platform1 = get_platform()
        platform2 = get_platform()
        assert platform1 is platform2, "get_platform() should return the same instance"


class TestEnvironmentVariableValidation:
    """Test VERL_PLATFORM environment variable validation."""

    def setup_method(self):
        """Reset platform singleton before each test."""
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_invalid_platform_falls_back(self):
        """Test that invalid VERL_PLATFORM values fall back to auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "opencl"}):
            name = _detect_platform_name()
            # Should fall back, not return "opencl"
            assert name != "opencl"
            assert name in ("cuda", "npu")

    def test_empty_platform_triggers_auto_detection(self):
        """Test that empty VERL_PLATFORM triggers auto-detection."""
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            name = _detect_platform_name()
            assert name in ("cuda", "npu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
