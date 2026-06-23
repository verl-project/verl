import importlib.util
import os
import types


class ConfigDict(dict):

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


class Config:

    @staticmethod
    def fromfile(file_path):
        config_dict = ConfigDict()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        # Load the configuration file as a module
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Function to convert nested dictionaries to ConfigDict recursively
        def convert_to_config_dict(d):
            if isinstance(d, dict):

                config_dict = ConfigDict()
                for key, value in d.items():
                    if isinstance(value, dict):
                        config_dict[key] = convert_to_config_dict(value)
                    else:
                        config_dict[key] = value
                return config_dict
            else:
                return d

        # Retrieve all attributes (variables) from the module
        for attribute_name in dir(config_module):
            if not attribute_name.startswith("__"):
                config_dict[attribute_name] = convert_to_config_dict(
                    getattr(config_module, attribute_name)
                )
        for key, value in list(config_dict.items()):
            if isinstance(value, (types.FunctionType, types.ModuleType)):
                config_dict.pop(key)
        return config_dict
