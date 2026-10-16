"""
Bootcamp Registry for InternBootcamp Environment

This module provides a registry system for dynamically discovering and managing
InternBootcamp tasks without having to manually import each one.
"""

import importlib
import inspect
import logging
import random
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class BootcampRegistry:
    """Registry for InternBootcamp tasks with dynamic discovery."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._discovered = False

    def discover_bootcamps(self) -> None:
        """Dynamically discover all available bootcamp classes from InternBootcamp."""
        if self._discovered:
            return

        try:
            # Import the internbootcamp.bootcamp module
            bootcamp_module = importlib.import_module("internbootcamp.bootcamp")

            # Get all attributes from the module
            for name in dir(bootcamp_module):
                if name.endswith("bootcamp") and not name.startswith("_"):
                    try:
                        obj = getattr(bootcamp_module, name)
                        # Check if it's a class and has the required methods
                        if (
                            inspect.isclass(obj)
                            and hasattr(obj, "case_generator")
                            and hasattr(obj, "prompt_func")
                            and hasattr(obj, "verify_score")
                        ):
                            self._registry[name] = obj
                            logger.debug(f"Registered bootcamp: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to register {name}: {e}")

            self._discovered = True
            logger.info(f"Discovered {len(self._registry)} bootcamp tasks")

        except ImportError as e:
            logger.error(f"Failed to import internbootcamp.bootcamp: {e}")
            raise

    def get_bootcamp_class(self, name: str) -> Type:
        """Get a bootcamp class by name."""
        if not self._discovered:
            self.discover_bootcamps()

        if name not in self._registry:
            available = self.list_available_bootcamps()
            raise ValueError(
                f"Unknown bootcamp: {name}. "
                f"Available bootcamps: {', '.join(available[:10])}..."
                f" ({len(available)} total)"
            )

        return self._registry[name]

    def create_bootcamp_instance(self, name: str, **params) -> Any:
        """Create an instance of a bootcamp with given parameters."""
        bootcamp_class = self.get_bootcamp_class(name)

        # Get the __init__ signature to see what parameters are accepted
        try:
            sig = inspect.signature(bootcamp_class.__init__)
            valid_params = {}

            # Filter out parameters that the bootcamp doesn't accept
            for param_name, param_value in params.items():
                if param_name in sig.parameters:
                    valid_params[param_name] = param_value
                else:
                    logger.warning(
                        f"Parameter '{param_name}' not accepted by {name}, ignoring"
                    )

            return bootcamp_class(**valid_params)

        except Exception as e:
            logger.error(f"Failed to create instance of {name}: {e}")
            # Try with no parameters as fallback
            try:
                return bootcamp_class()
            except Exception as e:
                raise e

    def list_available_bootcamps(self) -> List[str]:
        """List all available bootcamp names."""
        if not self._discovered:
            self.discover_bootcamps()
        return sorted(list(self._registry.keys()))

    def get_bootcamp_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific bootcamp."""
        bootcamp_class = self.get_bootcamp_class(name)

        info = {
            "name": name,
            "class": bootcamp_class,
            "docstring": inspect.getdoc(bootcamp_class) or "No documentation available",
            "parameters": {},
        }

        # Get __init__ parameters
        try:
            sig = inspect.signature(bootcamp_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name not in ["self"]:
                    param_info = {
                        "default": (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else None
                        ),
                    }
                    info["parameters"][param_name] = param_info
        except Exception as e:
            logger.warning(f"Could not inspect parameters for {name}: {e}")

        return info


class RandomTask:
    """Special bootcamp that randomly selects from available bootcamps on each call."""

    def __init__(self, **params):
        self.registry = BootcampRegistry()
        self.registry.discover_bootcamps()
        self.available_bootcamps = self.registry.list_available_bootcamps()
        # Remove base classes and template classes from the list
        self.available_bootcamps = [
            name
            for name in self.available_bootcamps
            if not any(x in name.lower() for x in ["base", "template", "{puzzlename}"])
        ]
        self.params = params
        self.current_bootcamp = None
        self.current_bootcamp_name = None
        logger.info(
            f"RandomTask initialized with {len(self.available_bootcamps)} available bootcamps"
        )

    def case_generator(self) -> object:
        """Generate a case by randomly selecting a bootcamp."""
        # Select a random bootcamp
        self.current_bootcamp_name = random.choice(self.available_bootcamps)
        self.current_bootcamp = self.registry.create_bootcamp_instance(
            self.current_bootcamp_name, **self.params
        )

        # Generate case from the selected bootcamp
        case = self.current_bootcamp.case_generator()

        # Add bootcamp name to the case for tracking
        if isinstance(case, dict):
            case["_bootcamp_name"] = self.current_bootcamp_name
        else:
            # If case is not a dict, wrap it
            case = {"data": case, "_bootcamp_name": self.current_bootcamp_name}

        return case

    def prompt_func(self, identity) -> str:
        """Generate prompt using the current bootcamp."""
        # Extract the bootcamp name if stored
        bootcamp_name = identity.get("_bootcamp_name", self.current_bootcamp_name)

        # If we need to recreate the bootcamp (e.g., during scoring)
        if not self.current_bootcamp or self.current_bootcamp_name != bootcamp_name:
            self.current_bootcamp_name = bootcamp_name
            self.current_bootcamp = self.registry.create_bootcamp_instance(
                bootcamp_name, **self.params
            )

        # Remove the bootcamp name before passing to prompt_func
        identity_copy = dict(identity)
        identity_copy.pop("_bootcamp_name", None)
        if "data" in identity_copy and len(identity_copy) == 1:
            identity_copy = identity_copy["data"]

        return self.current_bootcamp.prompt_func(identity_copy)

    @classmethod
    def extract_output(cls, output):
        """This should not be called directly for RandomTask."""
        raise NotImplementedError(
            "RandomTask does not implement extract_output directly"
        )

    @classmethod
    def _verify_correction(cls, solution, identity):
        """This should not be called directly for RandomTask."""
        raise NotImplementedError(
            "RandomTask does not implement _verify_correction directly"
        )

    def verify_score(
        self,
        model_output,
        identity,
        format_score=0,
        short_penalty=True,
        short_threshold=100,
        format_penalty=True,
    ) -> float:
        """Verify score using the appropriate bootcamp."""
        # Extract the bootcamp name
        bootcamp_name = identity.get("_bootcamp_name", self.current_bootcamp_name)

        # If we need to recreate the bootcamp
        if not self.current_bootcamp or self.current_bootcamp_name != bootcamp_name:
            self.current_bootcamp_name = bootcamp_name
            self.current_bootcamp = self.registry.create_bootcamp_instance(
                bootcamp_name, **self.params
            )

        # Remove the bootcamp name before passing to verify_score
        identity_copy = dict(identity)
        identity_copy.pop("_bootcamp_name", None)
        if "data" in identity_copy and len(identity_copy) == 1:
            identity_copy = identity_copy["data"]

        # Call the bootcamp's verify_score method
        return self.current_bootcamp.verify_score(
            model_output,
            identity_copy,
            format_score,
            short_penalty,
            short_threshold,
            format_penalty,
        )


# Global registry instance
bootcamp_registry = BootcampRegistry()


def get_available_bootcamps() -> List[str]:
    """Get a list of all available bootcamp names."""
    return bootcamp_registry.list_available_bootcamps()


def create_bootcamp(name: str, **params) -> Any:
    """Create a bootcamp instance by name with parameters."""
    # Special handling for RandomTask
    if name == "RandomTask":
        return RandomTask(**params)
    return bootcamp_registry.create_bootcamp_instance(name, **params)
