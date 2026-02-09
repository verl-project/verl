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
Contain small python utility functions
"""

import importlib
import multiprocessing
import os
import queue  # Import the queue module for exception type hint
import signal
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional

from verl.utils.metric import Metric


# --- Top-level helper for multiprocessing timeout ---
# This function MUST be defined at the top level to be pickleable
def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: tuple, kwargs: dict[str, Any]):
    """
    Internal wrapper function executed in the child process.
    Calls the original target function and puts the result or exception into the queue.
    """
    try:
        result = target_func(*args, **kwargs)
        mp_queue.put((True, result))  # Indicate success and put result
    except Exception as e:
        # Ensure the exception is pickleable for the queue
        try:
            import pickle

            pickle.dumps(e)  # Test if the exception is pickleable
            mp_queue.put((False, e))  # Indicate failure and put exception
        except (pickle.PicklingError, TypeError):
            # Fallback if the original exception cannot be pickled
            mp_queue.put((False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}")))


# Renamed the function from timeout to timeout_limit
def timeout_limit(seconds: float, use_signals: bool = False):
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).
    """

    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            # Issue deprecation warning if use_signals is explicitly True
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                # Use setitimer for float seconds support, alarm only supports integers
                signal.setitimer(signal.ITIMER_REAL, seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    # Reset timer and handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result

            return wrapper_signal
        else:
            # --- Multiprocessing based timeout (existing logic) ---
            @wraps(func)
            def wrapper_mp(*args, **kwargs):
                q = multiprocessing.Queue(maxsize=1)
                process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))
                process.start()
                process.join(timeout=seconds)

                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)  # Give it a moment to terminate
                    if process.is_alive():
                        print(f"Warning: Process {process.pid} did not terminate gracefully after timeout.")
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!")

                try:
                    success, result_or_exc = q.get(timeout=0.1)  # Small timeout for queue read
                    if success:
                        return result_or_exc
                    else:
                        raise result_or_exc  # Reraise exception from child
                except queue.Empty as err:
                    exitcode = process.exitcode
                    if exitcode is not None and exitcode != 0:
                        raise RuntimeError(
                            f"Child process exited with error (exitcode: {exitcode}) before returning result."
                        ) from err
                    else:
                        # Should have timed out if queue is empty after join unless process died unexpectedly
                        # Update function name in error message if needed (optional but good practice)
                        raise TimeoutError(
                            f"Operation timed out or process finished unexpectedly without result "
                            f"(exitcode: {exitcode})."
                        ) from err
                finally:
                    q.close()
                    q.join_thread()

            return wrapper_mp

    return decorator


def union_two_dict(dict1: dict, dict2: dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def rename_dict(data: dict, prefix: str = "") -> dict:
    """Add a prefix to all the keys in the data dict if it's name is not started with prefix

    Args:
        data: a dictionary
        prefix: prefix

    Returns:
        dictionary with modified name

    """
    new_data = {}
    for key, val in data.items():
        new_key = f"{prefix}{key}" if not key.startswith(prefix) else key
        new_data[new_key] = val
    return new_data


def append_to_dict(data: dict, new_data: dict, prefix: str = ""):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    for key, val in new_data.items():
        new_key = f"{prefix}{key}" if not key.startswith(prefix) else key
        if new_key not in data:
            data[new_key] = val.init_list() if isinstance(val, Metric) else []
        if isinstance(val, list):
            data[new_key].extend(val)
        else:
            data[new_key].append(val)


class NestedNamespace(SimpleNamespace):
    """A nested version of SimpleNamespace that recursively converts dictionaries to namespaces.

    This class allows for dot notation access to nested dictionary structures by recursively
    converting dictionaries to NestedNamespace objects.

    Example:
        config_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = NestedNamespace(config_dict)
        # Access with: config.a, config.b.c, config.b.d

    Args:
        dictionary: The dictionary to convert to a nested namespace.
        **kwargs: Additional attributes to set on the namespace.
    """

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


class DynamicEnumMeta(type):
    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._registry.values())

    def __contains__(cls, item: Any) -> bool:
        # allow `name in EnumClass` or `member in EnumClass`
        if isinstance(item, str):
            return item in cls._registry
        return item in cls._registry.values()

    def __getitem__(cls, name: str) -> Any:
        return cls._registry[name]

    def __reduce_ex__(cls, protocol):
        # Always load the existing module and grab the class
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)

    def names(cls):
        return list(cls._registry.keys())

    def values(cls):
        return list(cls._registry.values())


class DynamicEnum(metaclass=DynamicEnumMeta):
    _registry: dict[str, "DynamicEnum"] = {}
    _next_value: int = 0

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __reduce_ex__(self, protocol):
        """
        Unpickle via: getattr(import_module(module).Dispatch, 'ONE_TO_ALL')
        so the existing class is reused instead of re-executed.
        """
        module = importlib.import_module(self.__class__.__module__)
        enum_cls = getattr(module, self.__class__.__name__)
        return getattr, (enum_cls, self.name)

    @classmethod
    def register(cls, name: str) -> "DynamicEnum":
        key = name.upper()
        if key in cls._registry:
            raise ValueError(f"{key} already registered")
        member = cls(key, cls._next_value)
        cls._registry[key] = member
        setattr(cls, key, member)
        cls._next_value += 1
        return member

    @classmethod
    def remove(cls, name: str):
        key = name.upper()
        member = cls._registry.pop(key)
        delattr(cls, key)
        return member

    @classmethod
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:
        return cls._registry.get(name.upper())


@contextmanager
def temp_env_var(key: str, value: str):
    """Context manager for temporarily setting an environment variable.

    This context manager ensures that environment variables are properly set and restored,
    even if an exception occurs during the execution of the code block.

    Args:
        key: Environment variable name to set
        value: Value to set the environment variable to

    Yields:
        None

    Example:
        >>> with temp_env_var("MY_VAR", "test_value"):
        ...     # MY_VAR is set to "test_value"
        ...     do_something()
        ... # MY_VAR is restored to its original value or removed if it didn't exist
    """
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, ListConfig | DictConfig):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, list | tuple):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj
