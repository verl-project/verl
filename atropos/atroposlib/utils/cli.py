import sys
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, create_model


def get_prefixed_pydantic_model(base_model: BaseModel, prefix: str) -> BaseModel:
    fields = {}
    for name, field in base_model.model_fields.items():
        new_name = f"{prefix}{name}"

        # Preserve field metadata, especially description
        field_kwargs = {}
        if hasattr(field, "description") and field.description is not None:
            field_kwargs["description"] = field.description

        # Handle both default and default_factory
        if field.default_factory is not None:
            field_kwargs["default_factory"] = field.default_factory
        else:
            field_kwargs["default"] = field.default

        fields[new_name] = (
            field.annotation,
            Field(**field_kwargs),
        )

    return create_model(f"{prefix.capitalize()}{base_model.__name__}", **fields)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later dictionaries taking precedence over earlier ones.
    If multiple dictionaries contain a dictionary at the same key, the dictionaries
    are recursively merged.
    Args:
        *dicts: An arbitrary number of dictionaries to merge.
    Returns:
        A merged dictionary with values from later dictionaries taking precedence.
    """
    if not dicts:
        return {}

    result = dicts[0].copy()
    for current_dict in dicts[1:]:
        for key, value in current_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

    return result


def extract_namespace(model: Any, prefix: str) -> Dict[str, Any]:
    """Extract fields from a model or dict that start with a given prefix, stripping the prefix from the keys.
    Args:
        model: A Pydantic model or any object with `.model_dump()` or `.to_dict()`, or a plain dict.
        prefix: Prefix to filter by (e.g., 'env.', 'server.').
    Returns:
        A dict of key-value pairs where keys had the prefix, with prefix stripped.
    """
    try:
        data = model.model_dump()
    except AttributeError:
        try:
            data = model.to_dict()
        except AttributeError:
            if isinstance(model, dict):
                data = model
            else:
                raise TypeError(
                    "Model must have model_dump(), to_dict(), or be a dict."
                )

    result = {
        key[len(prefix) :]: value
        for key, value in data.items()
        if key.startswith(prefix)
    }
    return result


def adjust_model_defaults(
    model_class: Type[BaseModel],
    defaults: dict[str, Any] | BaseModel,
) -> Type[BaseModel]:
    """Return a subclass of *model_class* with some field defaults replaced.
    *Fields listed in *defaults* get their default value replaced (other
    metadata is *not* preserved, by design).
    *All other fields are inherited verbatim from *model_class* – we do **not**
      re-declare them, which avoids Pydantic's "field overridden by a
      non-annotated attribute" error.
    Args:
        model_class: Original Pydantic model.
        defaults:   Mapping ``field_name -> new_default`` or a Pydantic model instance.
    Returns:
        A fresh subclass called ``<OriginalName>WithDefaults``.
    """
    overridden: Dict[str, tuple] = {}

    # Convert Pydantic model to dict if needed
    if isinstance(defaults, BaseModel):
        defaults_dict = defaults.model_dump()
    else:
        defaults_dict = defaults

    for name, field_info in model_class.model_fields.items():
        if name in defaults_dict:
            # Re-declare the field with its original annotation and new default
            # Preserve the field's description and other metadata
            field_kwargs = {}
            if (
                hasattr(field_info, "description")
                and field_info.description is not None
            ):
                field_kwargs["description"] = field_info.description

            # If the original field had a default_factory and we're not explicitly overriding with a value,
            # preserve the default_factory
            if field_info.default_factory is not None and defaults_dict[name] is None:
                field_kwargs["default_factory"] = field_info.default_factory
            else:
                field_kwargs["default"] = defaults_dict[name]

            overridden[name] = (
                field_info.annotation,
                Field(**field_kwargs),
            )

    new_name = f"{model_class.__name__}WithDefaults"
    return create_model(new_name, __base__=model_class, **overridden)


def get_double_dash_flags() -> Dict[str, Any]:
    """
    Parses command-line arguments starting with '--' directly from sys.argv
    into a dictionary.
    - Handles flags like '--key=value'.
    - Handles flags like '--key value' (if 'value' doesn't start with '-').
    - Treats standalone flags like '--verbose' as boolean True.
    This function automatically reads sys.argv, ignoring the first element
    (the script name), and processes the remaining arguments. It takes no parameters.
    Returns:
        A dictionary where keys are the flag names (without '--') and
        values are the parsed values (string or True).
    Example:
        # Assuming the script was run like:
        # python your_script.py --user=admin --port 8080 --verbose --input-file data.csv -o out.txt last_arg
        # Inside your_script.py:
        # flags_dict = get_double_dash_flags_as_dict()
        # print(flags_dict)
        # Output would be:
        # {'user': 'admin', 'port': '8080', 'verbose': True, 'input-file': 'data.csv'}
    """
    args = sys.argv[1:]
    flags_dict: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]

        if not arg.startswith("--"):
            # Skip arguments that don't start with '--'
            i += 1
            continue

        # Remove '--' prefix
        key_part = arg[2:]
        key = ""
        value_str = (
            None  # Variable to hold the string value before potential conversion
        )

        # Check for '--key=value' format
        if "=" in key_part:
            key, value_str = key_part.split("=", 1)
            if not key:  # Ensure key is not empty (e.g. --=value)
                i += 1
                continue  # Skip if key is empty

            # Process value: Convert "None" string to None object
            if value_str == "None":
                flags_dict[key] = None
            else:
                flags_dict[key] = value_str
            i += 1
        # Check if next argument exists and is a value (doesn't start with '-')
        elif i + 1 < len(args) and not args[i + 1].startswith("-"):
            key = key_part
            value_str = args[i + 1]

            # Process value: Convert "None" string to None object
            if value_str == "None":
                flags_dict[key] = None
            else:
                flags_dict[key] = value_str
            # Skip the next argument since we've consumed it as a value
            i += 2
        # Otherwise, treat as a boolean flag
        else:
            key = key_part
            if key:  # Ensure key is not empty (e.g. just '--')
                flags_dict[key] = True
            i += 1

    return flags_dict


if __name__ == "__main__":
    from pydantic_cli import Cmd, run_and_exit
    from trajectoryhandler.envs.base import BaseEnvConfig

    try:
        from rich import print
    except ImportError:
        pass

    # Create a command class
    class EnvConfigCmd(get_prefixed_pydantic_model(BaseEnvConfig, "env."), Cmd):
        def run(self):
            print("Model:")
            print(self)
            print("Extract namespace:")
            print(extract_namespace(self, "env."))
            print("double dash", extract_namespace(get_double_dash_flags(), "env."))

    # Run the CLI
    run_and_exit(EnvConfigCmd)
