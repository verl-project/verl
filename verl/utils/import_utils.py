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
"""
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

import importlib
import importlib.util
import os
import warnings
from functools import cache, wraps
from typing import Optional


@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None


@cache
def is_vllm_available():
    try:
        vllm_spec = importlib.util.find_spec("vllm")
    except ModuleNotFoundError:
        vllm_spec = None
    return vllm_spec is not None


@cache
def is_sglang_available():
    try:
        sglang_spec = importlib.util.find_spec("sglang")
    except ModuleNotFoundError:
        sglang_spec = None
    return sglang_spec is not None


@cache
def is_nvtx_available():
    try:
        nvtx_spec = importlib.util.find_spec("nvtx")
    except ModuleNotFoundError:
        nvtx_spec = None
    return nvtx_spec is not None


@cache
def is_trl_available():
    try:
        trl_spec = importlib.util.find_spec("trl")
    except ModuleNotFoundError:
        trl_spec = None
    return trl_spec is not None


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, list):
        external_libs = [external_libs]
    import importlib

    for external_lib in external_libs:
        importlib.import_module(external_lib)


PKG_PATH_PREFIX = "pkg://"
FILE_PATH_PREFIX = "file://"


def load_module(module_path: str, module_name: Optional[str] = None) -> object:
    """Load a module from a path.

    Args:
        module_path (str):
            The path to the module. Either
                - `pkg_path`, e.g.,
                    - "pkg://verl.utils.dataset.rl_dataset"
                    - "pkg://verl/utils/dataset/rl_dataset"
                - or `file_path` (absolute or relative), e.g.,
                    - "file://verl/utils/dataset/rl_dataset.py"
                    - "/path/to/verl/utils/dataset/rl_dataset.py"
        module_name (str, optional):
            The name of the module to added to ``sys.modules``. If not provided, the module will not be added,
                thus will not be cached and directly ``import``able.
    """
    if not module_path:
        return None

    if module_path.startswith(PKG_PATH_PREFIX):
        module_name = module_path[len(PKG_PATH_PREFIX) :].replace("/", ".")
        module = importlib.import_module(module_name)

    else:
        if module_path.startswith(FILE_PATH_PREFIX):
            module_path = module_path[len(FILE_PATH_PREFIX) :]

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Custom module file not found: {module_path=}")

        # Use the provided module_name for the spec, or derive a unique name to avoid collisions.
        spec_name = module_name or f"custom_module_{hash(os.path.abspath(module_path))}"
        spec = importlib.util.spec_from_file_location(spec_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path=}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from {module_path=}") from e

        if module_name is not None:
            import sys

            # Avoid overwriting an existing module with a different object.
            if module_name in sys.modules and sys.modules[module_name] is not module:
                raise RuntimeError(
                    f"Module name '{module_name}' already in `sys.modules` and points to a different module."
                )
            sys.modules[module_name] = module

    return module


def _get_qualified_name(func):
    """Get full qualified name including module and class (if any)."""
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"


def deprecated(replacement: str = ""):
    """Decorator to mark functions or classes as deprecated."""

    def decorator(obj):
        qualified_name = _get_qualified_name(obj)

        if isinstance(obj, type):
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                msg = f"Warning: Class '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return obj

        else:

            @wraps(obj)
            def wrapped(*args, **kwargs):
                msg = f"Warning: Function '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapped

    return decorator


def load_extern_object(module_path: str, object_name: str) -> object:
    """Load an object from a module path.

    Args:
        module_path (str): See :func:`load_module`.
        object_name (str):
            The name of the object to load with ``getattr(module, object_name)``.
    """
    module = load_module(module_path)

    if not hasattr(module, object_name):
        raise AttributeError(f"Object not found in module: {object_name=}, {module_path=}.")

    return getattr(module, object_name)


def load_class_from_fqn(fqn: str, description: str = "class") -> type:
    """Load a class from its fully qualified name.

    Args:
        fqn: Fully qualified class name (e.g., 'mypackage.module.ClassName').
        description: Description for error messages (e.g., 'AgentLoopManager').

    Returns:
        The loaded class.

    Raises:
        ValueError: If fqn format is invalid (missing dot separator).
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.

    Example:
        >>> cls = load_class_from_fqn("verl.experimental.agent_loop.AgentLoopManager")
        >>> instance = cls(config=config, ...)
    """
    if "." not in fqn:
        raise ValueError(
            f"Invalid {description} '{fqn}'. Expected fully qualified class name (e.g., 'mypackage.module.ClassName')."
        )
    try:
        module_path, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}' for {description}: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}") from e


@deprecated(replacement="load_module(file_path); getattr(module, type_name)")
def load_extern_type(file_path: str, type_name: str) -> type:
    """DEPRECATED. Directly use `load_extern_object` instead."""
    return load_extern_object(file_path, type_name)
