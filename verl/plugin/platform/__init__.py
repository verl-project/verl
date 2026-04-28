# Copyright (c) 2026 BAAI. All rights reserved.
"""Unified platform abstraction for multi-chip support.

Usage::

    from verl.plugin.platform import get_platform

    platform = get_platform()          # auto-detected singleton
    platform.manual_seed(42)
    with platform.nvtx_range("train"):
        ...

Set ``VERL_PLATFORM=cuda`` (or ``npu``) to override auto-detection.
"""

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry, get_platform, set_platform

__all__ = ["PlatformBase", "PlatformRegistry", "get_platform", "set_platform"]
