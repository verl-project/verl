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
import importlib
import logging
import os
import sys
from typing import Any, Callable, Optional

from omegaconf import DictConfig

from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Type alias for filter function signature
FilterFunction = Callable[..., Any]


class FilterManager:
    """Manager for loading and applying custom filter functions to agent loop outputs."""

    def __init__(self, config: DictConfig):
        """Initialize FilterManager.

        Args:
            config (DictConfig): Configuration containing filter settings.
                Expected keys:
                - filter_function_path: Path to Python file containing filter function
                - filter_function_name: Name of filter function in the module
        """
        self.config = config
        self._filter_func: Optional[FilterFunction] = None
        self._load_filter_function()

    def _load_filter_function(self) -> None:
        """Load custom filter function from configured path.

        Attempts to load a filter function from the configured module path.

        Raises:
            RuntimeError: If module loading fails or function not found.
        """

        filter_path = self.config.filter.get(
            "filter_function_path", "verl/experimental/fully_async_policy/utils/filter/dummy_filter.py"
        )
        filter_name = self.config.filter.get("filter_function_name", "dummy_filter")

        logger.info(f"Loading filter function '{filter_name}' from {filter_path}")

        try:
            self._filter_func = self._load_module_function(filter_path, filter_name)
        except Exception as e:
            logger.error(f"Failed to load filter function: {e}")
            raise

    def _load_module_function(self, module_path: str, function_name: str) -> FilterFunction:
        """Load a specific function from a Python module.

        Args:
            module_path (str): Path to the Python file.
            function_name (str): Name of the function to load.

        Returns:
            FilterFunction: The loaded filter function.

        Raises:
            RuntimeError: If module cannot be loaded or function not found.
        """
        module_key = "custom_filter_module"

        # Check if module already loaded
        module = sys.modules.get(module_key)
        if module is not None:
            logger.debug(f"Using cached module: {module_key}")
        else:
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_key, module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec from {module_path}")

            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules[module_key] = module
                spec.loader.exec_module(module)
                logger.info(f"Successfully loaded module from {module_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to execute module from {module_path}: {e}") from e

        # Verify function exists
        if not hasattr(module, function_name):
            available_functions = [name for name in dir(module) if not name.startswith("_")]
            raise RuntimeError(
                f"Filter function '{function_name}' not found in {module_path}. "
                f"Available functions: {available_functions}"
            )

        filter_func = getattr(module, function_name)
        logger.info(f"Successfully loaded filter function '{function_name}' from {module.__file__}")
        return filter_func

    def apply(self, output: DataProto) -> DataProto:
        """Apply filter function to agent loop output.

        Args:
            output(DataProto): The data to be filtered, the output includes all rollout.n trajectory.

        Returns:
            DataProto: Filtered output.
        """

        if self._filter_func is None:
            return output

        original_count = len(output)
        try:
            filtered_outputs = self._filter_func(output, self.config)
            filtered_count = len(filtered_outputs)
            logger.info(
                f"Filter applied: {original_count} inputs → {filtered_count} outputs "
                f"({100 * (1 - filtered_count / original_count):.1f}% removed)"
            )
            return filtered_outputs
        except Exception as e:
            logger.exception(f"Error applying filter function: {e}")
            raise

    def is_enabled(self) -> bool:
        """Check if filtering is enabled.

        Returns:
            bool: True if a filter function is loaded, False otherwise.
        """
        return self._filter_func is not None
