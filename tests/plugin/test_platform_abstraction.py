# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the platform abstraction layer.

We pre-populate ``sys.modules`` with lightweight stubs for the top-level
``verl`` and ``verl.plugin`` packages so that importing the platform
sub-package does **not** trigger the heavy dependency chain pulled in by
``verl/__init__.py`` (ray, tensordict, …).
"""

import os
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest import mock

import pytest

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

from verl.plugin.platform import get_platform, set_platform  # noqa: E402
from verl.plugin.platform.platform_base import PlatformBase  # noqa: E402
from verl.plugin.platform.platform_manager import (  # noqa: E402
    PlatformRegistry,
    _create_platform,
    _detect_platform_name,
)


def _make_mock_platform(name="mock_xpu"):
    """Return a minimal concrete PlatformBase subclass for testing."""

    class _Mock(PlatformBase):
        @property
        def device_name(self):
            return name

        @property
        def device_module(self):
            import torch

            return torch

        def is_available(self):
            return True

        def current_device(self):
            return 0

        def device_count(self):
            return 1

        def set_device(self, device_index):
            pass

        def synchronize(self, device_index=None):
            pass

        def manual_seed(self, seed):
            pass

        def manual_seed_all(self, seed):
            pass

        def set_allocator_settings(self, settings):
            pass

        def empty_cache(self):
            pass

        def get_device_capability(self, device_index=0):
            return (None, None)

        def communication_backend_name(self):
            return "mock_ccl"

        def visible_devices_envvar(self):
            return "MOCK_VISIBLE_DEVICES"

        @contextmanager
        def nvtx_range(self, msg):
            yield

        def profiler_start(self):
            pass

        def profiler_stop(self):
            pass

        def cudart(self):
            return None

    _Mock.__name__ = f"Mock_{name}"
    return _Mock


class TestPlatformDetection:
    """Test platform auto-detection logic."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_env_override_cuda(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cuda"}):
            assert _detect_platform_name() == "cuda"

    def test_env_override_npu(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "npu"}):
            assert _detect_platform_name() == "npu"

    def test_invalid_value_falls_back(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            assert _detect_platform_name() in ("cuda", "npu")

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "CUDA"}):
            assert _detect_platform_name() == "cuda"

    def test_empty_triggers_auto_detection(self):
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            assert _detect_platform_name() in ("cuda", "npu")


class TestPlatformCreation:
    """Test platform creation."""

    def test_cuda_raises_if_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                _create_platform("cuda")

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformSingleton:
    """Test singleton and external injection."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_set_platform_external_injection(self):
        """Simulates VERL_USE_EXTERNAL_MODULES plugin calling set_platform()."""
        custom = _make_mock_platform("mock_xpu")()
        set_platform(custom)
        assert get_platform() is custom
        assert get_platform().device_name == "mock_xpu"


class TestPlatformRegistry:
    """Test PlatformRegistry dynamic registration."""

    def setup_method(self):
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_builtin_platforms_registered(self):
        names = PlatformRegistry.registered_names()
        assert "cuda" in names
        assert "npu" in names

    def test_register_custom_platform(self):
        """External plugin registers a new platform via @PlatformRegistry.register()."""
        MockCls = _make_mock_platform("mock_test")
        PlatformRegistry.register(platform="mock_test")(MockCls)

        assert "mock_test" in PlatformRegistry.registered_names()
        assert PlatformRegistry.get("mock_test") is MockCls
        assert _create_platform("mock_test").device_name == "mock_test"

        del PlatformRegistry._platforms["mock_test"]

    def test_register_override(self):
        """Re-registering the same name overrides the previous class (last writer wins)."""
        original = PlatformRegistry.get("cuda")
        FakeCls = _make_mock_platform("cuda")
        PlatformRegistry.register(platform="cuda")(FakeCls)
        assert PlatformRegistry.get("cuda") is FakeCls
        PlatformRegistry._platforms["cuda"] = original

    def test_unregistered_platform_raises(self):
        with pytest.raises(ValueError):
            _create_platform("nonexistent_platform")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
