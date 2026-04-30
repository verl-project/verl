# Copyright (c) 2026 BAAI. All rights reserved.
"""Singleton platform manager with registry, auto-detection, and environment override.

The platform is resolved **once** on first call to :func:`get_platform` and
cached for the rest of the process lifetime.  Users can override auto-detection
by either:

* Setting the ``VERL_PLATFORM`` environment variable (e.g. ``cuda``, ``npu``).
* Calling :func:`set_platform` before any other code touches the platform.

New hardware backends are added via :meth:`PlatformRegistry.register`::

    @PlatformRegistry.register(platform="musa")
    class PlatformMUSA(PlatformBase):
        ...

External plugins loaded by ``VERL_USE_EXTERNAL_MODULES`` can register their
own platform classes without modifying the verl source tree.
"""

import logging
import os

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)

_current_platform: PlatformBase | None = None


class PlatformRegistry:
    """Registry that maps platform name strings to concrete ``PlatformBase`` subclasses.

    Built-in platforms (``cuda``, ``npu``) are registered at import time.
    External plugins can register additional platforms via the
    :meth:`register` decorator.
    """

    _platforms: dict[str, type[PlatformBase]] = {}

    @classmethod
    def register(cls, platform: str):
        """Class decorator that registers a ``PlatformBase`` subclass.

        Usage::

            @PlatformRegistry.register(platform="cuda")
            class PlatformCUDA(PlatformBase):
                ...
        """

        def decorator(platform_cls: type[PlatformBase]) -> type[PlatformBase]:
            assert issubclass(platform_cls, PlatformBase), f"{platform_cls.__name__} must be a subclass of PlatformBase"
            name = platform.strip().lower()
            if name in cls._platforms:
                logger.info(
                    "PlatformRegistry: overriding '%s' (%s -> %s)",
                    name,
                    cls._platforms[name].__name__,
                    platform_cls.__name__,
                )
            cls._platforms[name] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[PlatformBase] | None:
        """Look up a registered platform class by name (returns ``None`` if not found)."""
        return cls._platforms.get(name.strip().lower())

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Return a tuple of all registered platform names."""
        return tuple(cls._platforms.keys())


def _detect_platform_name() -> str:
    """Probe the environment and return the best platform name."""

    registered = PlatformRegistry.registered_names()

    # 1. Explicit user override via environment variable
    env_name = os.environ.get("VERL_PLATFORM", "").strip().lower()
    if env_name:
        if PlatformRegistry.get(env_name) is None:
            logger.warning(
                "Invalid VERL_PLATFORM='%s', registered platforms are %s. Falling back to auto-detection.",
                env_name,
                registered,
            )
        else:
            logger.info("Platform override from VERL_PLATFORM=%s", env_name)
            return env_name

    # 2. Auto-detect: try each registered platform in registration order
    for name in registered:
        platform_cls = PlatformRegistry.get(name)
        if platform_cls is None:
            continue
        try:
            if platform_cls().is_available():
                return name
        except Exception:
            continue

    # 3. No accelerator found – fall back to CPU with a warning
    logger.warning(
        "No supported accelerator detected. Registered platforms: %s. Falling back to 'cuda' (CPU-only mode).",
        registered,
    )
    return "cuda"


def _create_platform(name: str) -> PlatformBase:
    """Instantiate the concrete platform for *name*."""
    platform_cls = PlatformRegistry.get(name)
    if platform_cls is None:
        raise ValueError(
            f"Unknown platform '{name}'. Registered: {PlatformRegistry.registered_names()}. "
            "Use @PlatformRegistry.register() to add a new platform."
        )
    platform = platform_cls()
    if not platform.is_available():
        logger.warning("Platform '%s' (%s) is registered but not available.", name, platform_cls.__name__)
    return platform


def get_platform() -> PlatformBase:
    """Return the current platform singleton (auto-detected on first call)."""
    global _current_platform
    if _current_platform is None:
        name = _detect_platform_name()
        _current_platform = _create_platform(name)
        logger.info("verl platform initialised: %s", _current_platform.device_name)
    return _current_platform


def set_platform(platform: PlatformBase) -> None:
    """Override the platform singleton with an already-instantiated platform.

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


# ---------------------------------------------------------------------------
# Register built-in platforms.  Imported here so that the @register decorator
# fires at module load time.  The imports are at the bottom to avoid circular
# references (platform_cuda/npu import PlatformBase from platform_base, not
# from this module).
# ---------------------------------------------------------------------------
from .platform_cuda import PlatformCUDA  # noqa: E402, F401
from .platform_npu import PlatformNPU  # noqa: E402, F401
