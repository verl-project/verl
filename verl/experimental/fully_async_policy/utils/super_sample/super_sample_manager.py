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

SuperSampleFn = Callable[..., Any]


class SuperSampleManager:
    """Manager for loading and applying custom super sample functions."""

    def __init__(self, config: DictConfig):
        """Initialize SuperSampleManager.

        Args:
            config (DictConfig): Configuration containing super sample settings.
                Expected keys:
                - super_sample_function_path: Path to Python file containing super sample function
                - super_sample_function_name: Name of super sample function in the module
        """
        self.config = config
        self._super_sample_func: Optional[SuperSampleFn] = None
        self._load_super_sample_function()

    def _load_super_sample_function(self) -> None:
        """Load custom super sample function from configured path.

        Attempts to load a super sample function from the configured module path.
        Raises:
            RuntimeError: If module loading fails or function not found.
        """

        super_sample_path = self.config.super_sample.get(
            "super_sample_function_path",
            "verl/experimental/fully_async_policy/utils/super_sample/default_super_sample.py",
        )

        super_sample_function_name = self.config.super_sample.get(
            "super_sample_function_name", "default_super_sample_func"
        )
        logger.info(f"Loading super sample function '{super_sample_function_name}' from {super_sample_path}")

        try:
            self._super_sample_func = self._load_module_function(super_sample_path, super_sample_function_name)
        except Exception as e:
            logger.error(f"Failed to load super sample function: {e}")
            raise

    def _load_module_function(self, module_path: str, function_name: str) -> SuperSampleFn:
        """Load a specific function from a Python module.

        Args:
            module_path (str): Path to the Python file.
            function_name (str): Name of the function to load.

        Returns:
            SuperSampleFn: The loaded super sample function.

        Raises:
            RuntimeError: If module cannot be loaded or function not found.
        """
        module_key = "custom_super_sample_module"

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
                f"Super sample function '{function_name}' not found in {module_path}. "
                f"Available functions: {available_functions}"
            )

        super_sample_func = getattr(module, function_name)
        logger.info(f"Successfully loaded super sample function '{function_name}' from {module.__file__}")
        return super_sample_func

    def apply(self, output: DataProto) -> DataProto:
        """Apply super sample function."""
        if self._super_sample_func is None:
            return output
        try:
            super_sampled_outputs = self._super_sample_func(output, self.config)
            return super_sampled_outputs
        except Exception as e:
            logger.exception(f"Error applying super sample function: {e}")
            raise

    def is_enabled(self) -> bool:
        """Check if super sample is enabled.

        Returns:
            bool: True if a super sample function is loaded, False otherwise.
        """
        return self._super_sample_func is not None
