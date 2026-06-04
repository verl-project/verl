# Copyright (c) 2026 BAAI. All rights reserved.
"""Singleton platform manager with registry and auto-detection.

The platform is resolved **once** on first call to :func:`get_platform` and
cached for the rest of the process lifetime.

New hardware backends are added via :meth:`PlatformRegistry.register`::

    @PlatformRegistry.register(platform="metax")
    class PlatformMetaX(PlatformBase):
        ...

External plugins loaded by ``VERL_USE_EXTERNAL_MODULES`` or discovered via
setuptools entry_points can register their own platform classes without
modifying the verl source tree.
"""

import logging
import os

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_current_platform: PlatformBase | None = None


class PlatformRegistry:
    """Registry that maps platform names to concrete ``PlatformBase`` subclasses.

    Built-in platforms (``cuda``, ``npu``) are registered at import time.
    External plugins can register additional platforms via the
    :meth:`register` decorator.

    Each platform name (e.g. ``cuda``, ``npu``, ``metax``, ``xpu``, ``mlu``)
    is already vendor-specific, so no separate vendor key is needed.
    """

    _platforms: dict[str, type[PlatformBase]] = {}

    @classmethod
    def register(cls, platform: str):
        """Class decorator that registers a ``PlatformBase`` subclass.

        Usage::

            @PlatformRegistry.register(platform="nvidia")
            class PlatformCUDA(PlatformBase):
                ...

            @PlatformRegistry.register(platform="metax")
            class PlatformMetaX(PlatformBase):
                ...
        """

        def decorator(platform_cls: type[PlatformBase]) -> type[PlatformBase]:
            assert issubclass(platform_cls, PlatformBase), f"{platform_cls.__name__} must be a subclass of PlatformBase"
            name = platform.strip().lower()
            if name in cls._platforms:
                logger.info(
                    "PlatformRegistry: overriding %s (%s -> %s)",
                    name,
                    cls._platforms[name].__name__,
                    platform_cls.__name__,
                )
            cls._platforms[name] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[PlatformBase] | None:
        """Look up a registered platform class by name."""
        return cls._platforms.get(name.strip().lower())

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Return a tuple of all registered platform names."""
        return tuple(cls._platforms.keys())


def _detect_platform_name() -> str:
    """Probe the environment and return the best platform name.

    Detection order:
    1. ``VERL_PLATFORM`` environment variable (explicit override)
    2. Probe registered platforms via ``is_platform_available(use_smi_check=True)``
    3. Fall back to ``'nvidia'``
    """

    env_platform = os.environ.get("VERL_PLATFORM", "").strip().lower()
    if env_platform:
        logger.info("Platform override from VERL_PLATFORM: %s", env_platform)
        return env_platform

    names = PlatformRegistry.registered_names()
    logger.debug("Registered platforms: %s", names)

    for name in names:
        platform_cls = PlatformRegistry.get(name)
        if platform_cls is None:
            continue
        try:
            instance = platform_cls()
            if instance.is_platform_available(use_smi_check=True):
                logger.debug("Auto-detected platform: %s", name)
                return name
        except Exception as e:
            logger.debug("Platform '%s' detection failed: %s", name, e)
            continue

    logger.warning(
        "No supported accelerator detected. Registered: %s. Falling back to 'nvidia'.",
        names,
    )
    return "nvidia"


def _create_platform(name: str) -> PlatformBase:
    """Instantiate the concrete platform for *name*."""
    platform_cls = PlatformRegistry.get(name)
    if platform_cls is None:
        raise ValueError(
            f"Unknown platform '{name}'. "
            f"Registered: {PlatformRegistry.registered_names()}. "
            "Use @PlatformRegistry.register() to add a new platform."
        )
    platform = platform_cls()
    if not platform.is_available():
        logger.warning(
            "Platform '%s' (%s) is registered but not available. "
            "This may be due to this ray actor being a CPU-only actor.",
            name,
            platform_cls.__name__,
        )
    return platform


def get_platform() -> PlatformBase:
    """Return the current platform singleton (auto-detected on first call)."""
    global _current_platform
    if _current_platform is None:
        name = _detect_platform_name()
        _current_platform = _create_platform(name)
        logger.debug("verl platform initialised: %s", _current_platform.device_name)
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
