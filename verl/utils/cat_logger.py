# Copyright 2025 Meituan Ltd. and/or its affiliates
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
Generic Cat Logger utility for Cat(raptor) monitoring operations.

This module provides a unified interface for Cat monitoring across the verl codebase.
It abstracts away the complexity of Cat initialization checks and provides
convenient methods for transaction management.

Features:
    - init_cat(): Initialize Cat monitoring system based on config (call once per process)
    - CatLogger: Main class for transaction logging (use where monitoring is needed)
    - Global logger instance: Convenient access to a global logger
    - Context manager support: Automatic transaction lifecycle management
    - Flexible API: Works with or without explicit transaction variables

Typical workflow:
    1. In process initialization: Call init_cat() once(per process) to set up Cat system
    2. In monitoring locations: Create CatLogger instances and log transactions

Example 1 - Process initialization (call once):
    >>> # In your process startup code
    >>> from verl.utils.cat_logger import init_cat
    >>>
    >>> def initialize_process(config):
    ...     # Initialize Cat system once at process startup
    ...     if init_cat(config):
    ...         logger.info("Cat monitoring initialized")

Example 2 - Monitoring in worker/function (use where needed):
    >>> # In your worker or function that needs monitoring
    >>> from verl.utils.cat_logger import CatLogger
    >>>
    >>> def process_data(data, config):
    ...     # Create CatLogger instance for this operation
    ...     cat_logger = CatLogger(config=config)
    ...     cat_logger.new_transaction(t_type="data_process")
    ...     try:
    ...         result = expensive_operation(data)
    ...         cat_logger.set_status(exception=False)
    ...         return result
    ...     except Exception as e:
    ...         cat_logger.set_status(exception=True)
    ...         raise
    ...     finally:
    ...         cat_logger.complete()

Example 3 - Using global logger (after initialization):
    >>> # After init_cat() and init_global_cat_logger() are called
    >>> from verl.utils.cat_logger import get_global_cat_logger
    >>>
    >>> def train_step(batch, config):
    ...     cat_logger = get_global_cat_logger()
    ...     cat_logger.new_transaction(t_type="train_step")
    ...     try:
    ...         loss = model(batch)
    ...         cat_logger.set_status(exception=False)
    ...     except Exception as e:
    ...         cat_logger.set_status(exception=True)
    ...         raise
    ...     finally:
    ...         cat_logger.complete()

Example 4 - Using context manager:
    >>> from verl.utils.cat_logger import CatLogger
    >>>
    >>> def http_request(url, config):
    ...     cat_logger = CatLogger(config=config)
    ...     with cat_logger.transaction(t_type="http_request"):
    ...         response = requests.get(url)
    ...         return response.json()
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

try:
    from pycat import Cat, CatStatusEnum
except ImportError:
    Cat = None
    CatStatusEnum = None

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


def init_cat(config: Optional[Any] = None) -> bool:
    """
    Initialize Cat monitoring system.

    This is a generic initialization method that handles Cat setup based on config.
    It should be called early in the application lifecycle before any Cat logging occurs.

    Args:
        config: Configuration object containing cat settings. Should have:
                - cat.enable: bool - Whether to enable Cat monitoring
                - cat.app_key: str - App key for Cat service
                - cat.env_file_path: str - Path to environment file
                - cat.cat_routers_config_file: str - Path to routers config
                - cat.cat_routers_cache_file: str - Path to routers cache

    Returns:
        bool: True if Cat was successfully initialized, False otherwise.

    Example:
        >>> from verl.utils.cat_logger import init_cat
        >>> if init_cat(config):
        ...     logger.info("Cat monitoring initialized")
    """
    # Check if Cat is enabled in config
    if config is None:
        logger.warning("Cat initialization skipped: config is None")
        return False

    if not hasattr(config, "cat"):
        logger.warning("Cat initialization skipped: config.cat not found")
        return False

    if not hasattr(config.cat, "enable"):
        logger.warning("Cat initialization skipped: config.cat.enable not found")
        return False

    if not config.cat.enable:
        logger.info("Cat monitoring is disabled in config")
        return False

    # Try to initialize Cat
    try:
        if Cat is None:
            logger.warning("Cat initialization failed: pycat library not available")
            logger.warning("Install it with: pip install python-cat")
            return False

        # Check if already initialized
        if Cat.is_inited():
            logger.info("Cat is already initialized")
            return True

        # Get configuration parameters
        app_key = getattr(config.cat, "app_key", None)
        env_file_path = getattr(config.cat, "env_file_path", None)
        cat_routers_config_file = getattr(config.cat, "cat_routers_config_file", None)
        cat_routers_cache_file = getattr(config.cat, "cat_routers_cache_file", None)

        logger.info(f"Initializing CAT monitoring with app_key: {app_key}")

        # Initialize Cat with provided configuration
        Cat.init_cat(
            app_key=app_key,
            app_env_file=env_file_path,
            cat_routers_config_file=cat_routers_config_file,
            cat_routers_cache_file=cat_routers_cache_file,
            disable_falcon=True,
        )

        logger.info("Cat monitoring initialized successfully.")
        print("[CatLogger] Cat monitoring initialized successfully.")
        return True

    except ImportError:
        logger.warning("Cat initialization failed: pycat library not found")
        logger.warning("Install it with: pip install python-cat")
        return False

    except Exception as e:
        logger.error(f"Failed to initialize Cat monitoring: {e}")
        logger.error("Please check CAT configuration and network connectivity.")
        return False


class CatLogger:
    """
    Generic Cat Logger for monitoring and logging operations.

    This class provides a unified interface for Cat logging that can be used
    throughout the verl codebase. It handles initialization checks and provides
    convenient methods for transaction management.

    Attributes:
        config: Configuration object or dict containing cat settings. Should have
                    a `cat` key/attribute with `enable` and `cat_uuid` fields.
                    Supports both dict-style (config['cat']['enable']) and
                    object-style (config.cat.enable) access.
        enabled: Whether Cat logging is enabled (determined by config and Cat availability)
        cat_uuid: Cat UUID extracted from config, used as default transaction name
        t: Current transaction object. Can be None if Cat is disabled or not initialized.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the Cat Logger.

        Args:
            config: Optional configuration object or dict containing cat settings.
                        Supports both dict-style ({'cat': {'enable': True, 'cat_uuid': '...'}})
                        and object-style (config.cat.enable) access.
                        If None or if Cat is not available, logging will be disabled.
        """
        self.config = config
        self.enabled = self._check_cat_enabled()
        self.cat_uuid = self._extract_cat_uuid()
        self.t: Optional[Any] = None

    def _check_cat_enabled(self) -> bool:
        """
        Check if Cat logging is enabled and available.

        Returns:
            bool: True if Cat is available and enabled in config, False otherwise.
        """
        if Cat is None:
            return False

        if self.config is None:
            return False

        # Support both dict and object-style config access
        if isinstance(self.config, dict):
            cat_config = self.config.get("cat")
            if cat_config is None:
                return False

            if isinstance(cat_config, dict):
                return cat_config.get("enable", False)
            else:
                return getattr(cat_config, "enable", False)
        else:
            if not hasattr(self.config, "cat"):
                return False

            if not hasattr(self.config.cat, "enable"):
                return False

            return self.config.cat.enable

    def _extract_cat_uuid(self) -> Optional[str]:
        """
        Extract cat_uuid from config.

        Returns:
            str or None: The cat_uuid if available, None otherwise.
        """
        if self.config is None:
            return None

        # Support both dict and object-style config access
        if isinstance(self.config, dict):
            cat_config = self.config.get("cat")
            if cat_config is None:
                return None

            if isinstance(cat_config, dict):
                return cat_config.get("cat_uuid")
            else:
                return getattr(cat_config, "cat_uuid", None)
        else:
            if not hasattr(self.config, "cat"):
                return None

            return getattr(self.config.cat, "cat_uuid", None)

    def new_transaction(self, t_type: str = "operation", t_name: Optional[str] = None) -> Optional[Any]:
        """
        Create a new cat transaction for monitoring and logging.

        This method creates a new transaction and stores it in self.t.
        The transaction should be completed using the `complete()` method.

        Args:
            t_type: Type of transaction (e.g., "mcp_op", "http_op").
                   Defaults to "operation".
            t_name: Optional name for the transaction. If None, uses the cat_uuid
                   from config. Defaults to None.

        Returns:
            Transaction object if Cat is enabled, None otherwise.
            Also stores the transaction in self.t for later use.

        Example:
            >>> logger = CatLogger(config)
            >>> logger.new_transaction(t_type="mcp_op", t_name="batch_1")
            >>> try:
            ...     # Do work
            ...     logger.set_status(exception=False)
            ... except Exception as e:
            ...     logger.set_status(exception=True)
            ... finally:
            ...     logger.complete()
        """
        if not self.enabled:
            self.t = None
            return None

        try:
            if not Cat.is_inited():
                logger.warning("Cat is not initialized! Cannot create new transaction.")
                self.t = None
                return None

            # Use cat_uuid as default transaction name if not provided
            if t_name is None:
                t_name = self.cat_uuid

            self.t = Cat.new_transaction(t_type, t_name)
            return self.t

        except Exception as e:
            logger.warning(f"Failed to create Cat transaction: {e}")
            self.t = None
            return None

    def set_status(self, exception: bool = False, t: Optional[Any] = None) -> None:
        """
        Set the status of a cat transaction.

        This method updates the status of a transaction to indicate success
        or exception. Should be called before `complete()`.

        Args:
            exception: If True, sets status to EXCEPTION. If False, sets to SUCCESS.
                      Defaults to False.
            t: Optional transaction object. If not provided, uses self.t.
               If None, this method does nothing.

        Example:
            >>> logger = CatLogger(config)
            >>> logger.new_transaction(t_type="operation")
            >>> try:
            ...     result = do_something()
            ...     logger.set_status(exception=False)
            ... except Exception as e:
            ...     logger.set_status(exception=True)
            ...     raise
        """
        # Use provided t or fall back to self.t
        transaction = t if t is not None else self.t

        if not self.enabled or transaction is None:
            return

        try:
            if not Cat.is_inited():
                logger.warning("Cat is not initialized! Cannot set transaction status.")
                return

            status = CatStatusEnum.EXCEPTION if exception else CatStatusEnum.SUCCESS
            transaction.set_status(status)

        except Exception as e:
            logger.warning(f"Failed to set Cat transaction status: {e}")

    def complete(self, t: Optional[Any] = None) -> None:
        """
        Complete a cat transaction.

        This method marks a transaction as complete. Should be called in a
        finally block to ensure completion even if an exception occurs.

        Args:
            t: Optional transaction object. If not provided, uses self.t.
               If None, this method does nothing.

        Example:
            >>> logger = CatLogger(config)
            >>> logger.new_transaction(t_type="operation")
            >>> try:
            ...     result = do_something()
            ... finally:
            ...     logger.complete()
        """
        # Use provided t or fall back to self.t
        transaction = t if t is not None else self.t

        if not self.enabled or transaction is None:
            return

        try:
            transaction.complete()

        except Exception as e:
            logger.warning(f"Failed to complete Cat transaction: {e}")

    @contextmanager
    def transaction(self, t_type: str = "operation", t_name: Optional[str] = None):
        """
        Context manager for managing cat transactions automatically.

        This method provides a clean and Pythonic way to manage transactions.
        The transaction is automatically completed when exiting the context,
        and exceptions are automatically logged.

        Args:
            t_type: Type of transaction (e.g., "mcp_op", "inference", "training").
                   Defaults to "operation".
            t_name: Optional name for the transaction. If None, uses the cat_uuid
                   from config. Defaults to None.

        Yields:
            None (the transaction is managed internally)

        Example:
            >>> logger = CatLogger(config)
            >>> with logger.transaction(t_type="mcp_connect"):
            ...     # Do some work
            ...     result = connect_to_server()
            ...     # Exception is automatically logged if raised
        """
        t = self.new_transaction(t_type=t_type, t_name=t_name)
        exception_occurred = False
        try:
            yield
        except Exception:
            exception_occurred = True
            self.set_status(t, exception=True)
            raise
        else:
            if not exception_occurred:
                self.set_status(t, exception=False)
        finally:
            self.complete(t)

    def log_operation(self, t_type: str = "operation", t_name: Optional[str] = None, exception: bool = False) -> None:
        """
        Log a simple operation in one call (convenience method).

        This is a convenience method for logging operations that don't need
        explicit status management. It creates a transaction, sets status,
        and completes it all in one call.

        Args:
            t_type: Type of transaction (e.g., "mcp_op", "inference").
                   Defaults to "operation".
            t_name: Optional name for the transaction.
            exception: If True, logs as exception. If False, logs as success.
                      Defaults to False.

        Example:
            >>> logger = CatLogger(config)
            >>> logger.log_operation(t_type="inference", exception=False)
        """
        t = self.new_transaction(t_type=t_type, t_name=t_name)
        self.set_status(t, exception=exception)
        self.complete(t)


# Global cat logger instance for convenience
_global_cat_logger: Optional[CatLogger] = None


def init_global_cat_logger(config: Optional[Any] = None) -> CatLogger:
    """
    Initialize the global Cat Logger instance.

    Args:
        config: Configuration object containing cat settings.

    Returns:
        The global CatLogger instance.

    Example:
        >>> from verl.utils.cat_logger import init_global_cat_logger, get_global_cat_logger
        >>> init_global_cat_logger(config)
        >>> logger = get_global_cat_logger()
    """
    global _global_cat_logger
    _global_cat_logger = CatLogger(config=config)
    return _global_cat_logger


def get_global_cat_logger() -> CatLogger:
    """
    Get the global Cat Logger instance.

    Returns:
        The global CatLogger instance. If not initialized, returns a new
        CatLogger with no config.

    Example:
        >>> from verl.utils.cat_logger import get_global_cat_logger, init_global_cat_logger
        >>> init_global_cat_logger(config)
        >>> logger = get_global_cat_logger()
        >>> with logger.transaction(t_type="operation"):
        ...     # Do work
        ...     pass
    """
    global _global_cat_logger
    if _global_cat_logger is None:
        _global_cat_logger = CatLogger()
    return _global_cat_logger


@contextmanager
def cat_transaction(t_type: str = "operation", t_name: Optional[str] = None, config: Optional[Any] = None):
    """
    Standalone context manager for cat transactions.

    This function provides a convenient way to use cat logging without
    needing to instantiate a CatLogger object. It uses the global logger
    or creates a temporary one if needed.

    Args:
        t_type: Type of transaction (e.g., "mcp_op", "inference", "training").
               Defaults to "operation".
        t_name: Optional name for the transaction.
        config: Optional configuration object. If provided, creates a new
                    logger with this config. Otherwise uses global logger.

    Yields:
        None

    Example:
        >>> from verl.utils.cat_logger import cat_transaction
        >>> with cat_transaction(t_type="mcp_connect", t_name="server_1"):
        ...     # Do work
        ...     result = connect_to_server()
    """
    if config is not None:
        # Use temporary logger with provided config
        logger_instance = CatLogger(config=config)
    else:
        # Use global logger
        logger_instance = get_global_cat_logger()

    with logger_instance.transaction(t_type=t_type, t_name=t_name):
        yield
