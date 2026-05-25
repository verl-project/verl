# Copyright (c) 2026 BAAI. All rights reserved.
"""Singleton platform manager with registry and auto-detection.

The platform is resolved **once** on first call to :func:`get_platform` and
cached for the rest of the process lifetime.

New hardware backends are added via :meth:`PlatformRegistry.register`::

    @PlatformRegistry.register(platform="cuda", vendor="metax")
    class PlatformMetaX(PlatformBase):
        ...

External plugins loaded by ``VERL_USE_EXTERNAL_MODULES`` can register their
own platform classes without modifying the verl source tree.
"""

import logging

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)

_current_platform: PlatformBase | None = None


class PlatformRegistry:
    """Registry that maps (device_name, vendor) pairs to concrete ``PlatformBase`` subclasses.

    Built-in platforms (``cuda``, ``npu``) are registered at import time.
    External plugins can register additional platforms via the
    :meth:`register` decorator.

    Registration key is ``(platform, vendor)``. When ``vendor`` is None the
    class acts as the default for that platform name. During lookup, a
    vendor-specific match is preferred; if not found, the default (vendor=None)
    entry is used.
    """

    _platforms: dict[tuple[str, str | None], type[PlatformBase]] = {}

    @classmethod
    def register(cls, platform: str, vendor: str | None = None):
        """Class decorator that registers a ``PlatformBase`` subclass.

        Usage::

            @PlatformRegistry.register(platform="cuda")
            class PlatformCUDA(PlatformBase):
                ...

            @PlatformRegistry.register(platform="cuda", vendor="metax")
            class PlatformMetaX(PlatformBase):
                ...
        """

        def decorator(platform_cls: type[PlatformBase]) -> type[PlatformBase]:
            assert issubclass(platform_cls, PlatformBase), f"{platform_cls.__name__} must be a subclass of PlatformBase"
            name = platform.strip().lower()
            vendor_key = vendor.strip().lower() if vendor else None
            key = (name, vendor_key)
            if key in cls._platforms:
                logger.info(
                    "PlatformRegistry: overriding (%s, vendor=%s) (%s -> %s)",
                    name,
                    vendor_key,
                    cls._platforms[key].__name__,
                    platform_cls.__name__,
                )
            cls._platforms[key] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get(cls, name: str, vendor: str | None = None) -> type[PlatformBase] | None:
        """Look up a registered platform class by name and optional vendor.

        Tries ``(name, vendor)`` first, then falls back to ``(name, None)``.
        """
        name = name.strip().lower()
        vendor_key = vendor.strip().lower() if vendor else None

        # Vendor-specific match
        if vendor_key:
            result = cls._platforms.get((name, vendor_key))
            if result is not None:
                return result

        # Default (vendor=None) fallback
        return cls._platforms.get((name, None))

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Return a tuple of all registered platform device names (deduplicated)."""
        seen = dict.fromkeys(name for name, _ in cls._platforms.keys())
        return tuple(seen)

    @classmethod
    def registered_entries(cls) -> tuple[tuple[str, str | None], ...]:
        """Return all registered (platform, vendor) pairs."""
        return tuple(cls._platforms.keys())


def _detect_platform_name() -> tuple[str, str | None]:
    """Probe the environment and return the best (platform_name, vendor) pair.

    Detection order:
    1. Try vendor-specific entries first (more specific match wins)
    2. Fall back to default (vendor=None) entries
    3. If nothing is available, fall back to ('cuda', None)
    """

    entries = PlatformRegistry.registered_entries()
    logger.info("Registered platform entries: %s", entries)

    # Try vendor-specific entries first (vendor is not None)
    for name, vendor in entries:
        if vendor is None:
            continue
        platform_cls = PlatformRegistry.get(name, vendor)
        if platform_cls is None:
            continue
        try:
            instance = platform_cls()
            if instance.is_available(use_smi_check=True):
                logger.info("Auto-detected platform: %s (vendor=%s)", name, vendor)
                return name, vendor
        except Exception:
            continue

    # Try default entries (vendor=None)
    for name, vendor in entries:
        if vendor is not None:
            continue
        platform_cls = PlatformRegistry.get(name)
        if platform_cls is None:
            continue
        try:
            instance = platform_cls()
            if instance.is_available(use_smi_check=True):
                logger.info("Auto-detected platform: %s", name)
                return name, None
        except Exception:
            continue

    # No accelerator found
    logger.warning(
        "No supported accelerator detected. Registered: %s. Falling back to 'cuda'.",
        entries,
    )
    return "cuda", None


def _create_platform(name: str, vendor: str | None = None) -> PlatformBase:
    """Instantiate the concrete platform for *name* and optional *vendor*."""
    platform_cls = PlatformRegistry.get(name, vendor)
    if platform_cls is None:
        raise ValueError(
            f"Unknown platform '{name}' (vendor={vendor}). "
            f"Registered: {PlatformRegistry.registered_entries()}. "
            "Use @PlatformRegistry.register() to add a new platform."
        )
    platform = platform_cls()
    if not platform.is_available():
        logger.warning(
            "Platform '%s' (vendor=%s, %s) is registered but not available.", name, vendor, platform_cls.__name__
        )
    return platform


def get_platform() -> PlatformBase:
    """Return the current platform singleton (auto-detected on first call)."""
    global _current_platform
    if _current_platform is None:
        name, vendor = _detect_platform_name()
        _current_platform = _create_platform(name, vendor)
        logger.info(
            "verl platform initialised: %s (vendor=%s)", _current_platform.device_name, _current_platform.vendor
        )
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
