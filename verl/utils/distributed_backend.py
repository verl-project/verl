"""
Distributed backend selection module.

This module allows users to choose between Ray and YuanRong (ray_adapter) backends
via the DISTRIBUTED_BACKEND environment variable.

Usage:
    Set DISTRIBUTED_BACKEND=yr or DISTRIBUTED_BACKEND=yuanrong to use ray_adapter
    Set DISTRIBUTED_BACKEND=ray or leave unset to use ray (default)

    Import this module at the very beginning of entry points:
        import verl.utils.distributed_backend  # Must be before any other import ray
"""
import os
import sys

_BACKEND = os.getenv("DISTRIBUTED_BACKEND", "ray").lower()

if _BACKEND in ("yr", "yuanrong"):
    try:
        import ray_adapter as _ray_module
    except ImportError:
        raise ImportError(
            f"DISTRIBUTED_BACKEND is set to '{_BACKEND}' but ray_adapter is not installed. "
            "Please install ray_adapter or set DISTRIBUTED_BACKEND=ray to use the default Ray backend."
        )
else:
    import ray as _ray_module

# Inject the selected module into sys.modules so that all subsequent
# 'import ray' statements will use the selected backend
sys.modules['ray'] = _ray_module