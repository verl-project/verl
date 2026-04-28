# Copyright (c) 2026 BAAI. All rights reserved.
"""Singleton platform manager with auto-detection and environment override.

The platform is resolved **once** on first call to :func:`get_platform` and
cached for the rest of the process lifetime.  Users can override auto-detection
by either:

* Setting the ``VERL_PLATFORM`` environment variable (e.g. ``cuda``, ``npu``).
* Calling :func:`set_platform` before any other code touches the platform.
"""

import logging
import os

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)

_current_platform: PlatformBase | None = None

# Built-in platform names that are auto-detected.  Third-party backends can
# be added via ``set_platform()`` without modifying this list.
_BUILTIN_PLATFORMS = ("cuda", "npu")


def _detect_platform_name() -> str:
    """Probe the environment and return the best platform name."""

    # 1. Explicit user override via environment variable
    env_name = os.environ.get("VERL_PLATFORM", "").strip().lower()
    if env_name:
        if env_name not in _BUILTIN_PLATFORMS:
            logger.warning(
                "Invalid VERL_PLATFORM='%s', must be one of %s. Falling back to auto-detection.",
                env_name,
                _BUILTIN_PLATFORMS,
            )
            env_name = ""
        else:
            logger.info("Platform override from VERL_PLATFORM=%s", env_name)
            return env_name

    # 2. Auto-detect CUDA
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except (ImportError, RuntimeError):
        pass

    # 3. Auto-detect Ascend NPU
    try:
        import torch

        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            return "npu"
    except (ImportError, RuntimeError):
        pass

    # 4. No accelerator found
    raise RuntimeError(
        "No supported accelerator detected (checked CUDA and NPU). "
        "Set VERL_PLATFORM to one of %s if detection is incorrect." % (_BUILTIN_PLATFORMS,)
    )


def _create_platform(name: str) -> PlatformBase:
    """Instantiate the concrete platform for *name*."""
    if name == "cuda":
        from .platform_cuda import PlatformCUDA

        platform = PlatformCUDA()
        if not platform.is_available():
            raise RuntimeError("CUDA platform specified but not available.")
        return platform

    if name == "npu":
        from .platform_npu import PlatformNPU

        platform = PlatformNPU()
        if not platform.is_available():
            raise RuntimeError("NPU platform specified but not available.")
        return platform

    raise ValueError(
        f"Unknown platform '{name}'.  Built-in platforms are {_BUILTIN_PLATFORMS}.  "
        "Use set_platform() to register a custom PlatformBase instance."
    )


def get_platform() -> PlatformBase:
    """Return the current platform singleton (auto-detected on first call)."""
    global _current_platform
    if _current_platform is None:
        name = _detect_platform_name()
        _current_platform = _create_platform(name)
        logger.info("verl platform initialised: %s", _current_platform.device_name)
    return _current_platform


def set_platform(platform: PlatformBase) -> None:
    """Override the platform singleton.

    Must be called **before** any code calls :func:`get_platform`,
    otherwise a warning is emitted and the platform is replaced.
    """
    global _current_platform
    if _current_platform is not None:
        logger.warning(
            "Replacing already-initialised platform '%s' with '%s'",
            _current_platform.device_name,
            platform.device_name,
        )
    _current_platform = platform
