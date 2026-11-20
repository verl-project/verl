import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Type, Union

from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


class RewardRegistry:
    """Registry for reward functions with factory pattern"""

    def __init__(self):
        self._registry: Dict[str, Type[RewardFunction]] = {}
        self._reward_fns_dir = Path(__file__).parent

    def register(self, cls=None, name=None):
        """
        Register a reward function class.

        Can be used as a decorator:

        @registry.register
        class MyReward(RewardFunction):
            ...

        or with a custom name:

        @registry.register(name="custom_name")
        class MyReward(RewardFunction):
            ...

        Args:
            cls: The reward function class to register
            name: Optional custom name to register the class under

        Returns:
            The registered class (for decorator use)
        """

        def _register(cls):
            # Validate that it's a subclass of RewardFunction
            if not inspect.isclass(cls) or not issubclass(cls, RewardFunction):
                raise TypeError(
                    f"Class {cls.__name__} is not a subclass of RewardFunction"
                )

            registered_name = name or cls.__name__.lower()
            if registered_name.endswith("reward"):
                # Convert ClassNameReward to class_name
                registered_name = registered_name[:-6].lower()

            self._registry[registered_name] = cls
            logger.debug(f"Registered reward function: {registered_name}")
            return cls

        if cls is None:
            return _register
        return _register(cls)

    def register_function(self, name, function):
        """
        Register a legacy function-based reward function.
        This is temporary for backward compatibility.

        Args:
            name: Name to register the function under
            function: The reward function to register
        """
        self._registry[name] = function

    def create(self, name_or_config: Union[str, Dict], **kwargs) -> RewardFunction:
        """
        Create a reward function from name or config dict.

        Args:
            name_or_config: Either a string name of a registered reward function,
                           or a dict with 'type' key and optional parameters
            **kwargs: Default parameters that can be overridden by config

        Returns:
            Instantiated RewardFunction object
        """
        if isinstance(name_or_config, str):
            # Simple case: just a name
            reward_type = name_or_config
            reward_params = kwargs
        else:
            # Dict case with config
            reward_config = name_or_config.copy()
            reward_type = reward_config.pop("type")

            # Handle params dictionary if present
            if "params" in reward_config:
                params = reward_config.pop("params")
                reward_config.update(params)

            # Start with kwargs as defaults, override with config
            reward_params = {**kwargs}
            reward_params.update(reward_config)

        # Make sure the reward function is loaded
        if reward_type not in self._registry:
            self._load_reward_function(reward_type)

        reward_class = self._registry[reward_type]

        # Handle legacy function-based reward functions
        if not inspect.isclass(reward_class):
            # This is a function not a class - handle legacy case
            return LegacyFunctionWrapper(reward_class, **reward_params)

        # Create instance of the reward function class
        return reward_class(**reward_params)

    def get(self, name: str) -> Union[Type[RewardFunction], Callable]:
        """
        Get a reward function class by name.

        This is for backward compatibility with the old registry interface.
        New code should use create() instead.

        Args:
            name: The name of the reward function to get

        Returns:
            The reward function class or function
        """
        if name not in self._registry:
            self._load_reward_function(name)
        return self._registry[name]

    def _load_reward_function(self, name: str) -> None:
        """
        Load a reward function from a file.

        This supports both new class-based and legacy function-based reward functions.
        Files can be named either "name.py" or "name_reward.py".

        Args:
            name: The name of the reward function to load

        Raises:
            ImportError: If the reward function file is not found or can't be loaded
        """
        try:
            # Try different file name patterns
            base_name = name
            if name.endswith("_reward"):
                base_name = name[:-7]  # Remove "_reward" suffix

            module_paths = [
                self._reward_fns_dir / f"{base_name}.py",
                self._reward_fns_dir / f"{base_name}_reward.py",
                self._reward_fns_dir / f"{name}.py",
            ]

            module_path = None
            for path in module_paths:
                if path.exists():
                    module_path = path
                    break

            if module_path is None:
                raise ImportError(
                    f"No reward function file found for {name} (tried {', '.join(str(p) for p in module_paths)})"
                )

            # Generate a unique module name to avoid import conflicts
            module_name = f"atroposlib.envs.reward_fns.{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # First try to find a class that inherits from RewardFunction
            for obj_name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, RewardFunction)
                    and obj is not RewardFunction
                ):
                    # Register the class with the requested name
                    # This ensures it's accessible by the name the test expects
                    self.register_function(name, obj)
                    return

            # If no class found, look for functions with matching name patterns
            func_patterns = [f"{base_name}", f"{base_name}_reward", "format_reward"]
            for func_name in func_patterns:
                if hasattr(module, func_name):
                    self.register_function(name, getattr(module, func_name))
                    return

            raise AttributeError(f"No reward function found in {module_path}")

        except Exception as e:
            raise ImportError(f"Failed to load reward function {name}: {str(e)}")

    def list_registered(self) -> List[str]:
        """Return list of all registered reward function names"""
        return list(self._registry.keys())

    def load_required_functions(self, config) -> Set[str]:
        """
        Load all reward functions required by a config.

        This is for backward compatibility with the old registry interface.

        Args:
            config: The config object to load reward functions from

        Returns:
            A set of all loaded reward function names
        """
        required_funcs = set()

        if hasattr(config, "datasets"):
            for dataset in config.datasets:
                for field in ["dataset_reward_funcs", "reward_funcs"]:
                    if hasattr(dataset, field) and getattr(dataset, field):
                        required_funcs.update(getattr(dataset, field))

                if hasattr(dataset, "types") and dataset.types:
                    for type_config in dataset.types:
                        if "reward_funcs" in type_config:
                            required_funcs.update(type_config["reward_funcs"])

        # Also check for reward_functions and reward_funcs at the top level
        for field in ["reward_functions", "reward_funcs"]:
            if hasattr(config, field) and getattr(config, field):
                field_value = getattr(config, field)
                for item in field_value:
                    if isinstance(item, str):
                        required_funcs.add(item)
                    elif isinstance(item, dict) and "type" in item:
                        required_funcs.add(item["type"])

        for func_name in required_funcs:
            self.get(func_name)

        return required_funcs


class LegacyFunctionWrapper(RewardFunction):
    """Wrapper for legacy function-based reward functions to fit the new class-based interface"""

    def __init__(self, func: Callable, weight: float = 1.0, **kwargs):
        """
        Initialize with a legacy reward function.

        Args:
            func: The legacy reward function to wrap
            weight: The weight for this reward function
            **kwargs: Additional configuration parameters
        """
        super().__init__(weight=weight, **kwargs)
        self.func = func
        self._func_name = func.__name__

    @property
    def name(self) -> str:
        """Get the name of the wrapped function"""
        return self._func_name

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """Call the wrapped function"""
        result = self.func(completions, **kwargs)

        # Convert to list if it's a single value
        if not isinstance(result, list):
            result = [result] * len(completions)

        return result


# Global registry instance
registry = RewardRegistry()
