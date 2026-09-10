import json
import os

class ConfigReader:
    """
    A class to read and optionally save configuration data from/to a JSON file.
    """

    def __init__(self, config_path = "libs/futoshiki/config.json"):
        """
        Initializes the ConfigReader with the path to the JSON configuration file.

        :param config_path: Path to the config.json file
        """
        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self):
        """
        Loads the configuration data from the JSON file.

        :return: A dictionary containing the configuration data
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            try:
                with open('libs/futoshiki/config.json', 'r', encoding='utf-8') as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"The configuration file '{self.config_path}' was not found.")
            raise FileNotFoundError(f"The configuration file '{self.config_path}' was not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from the file '{self.config_path}': {e}")

    def get(self, key, default=None):
        """
        Retrieves the value for a given key from the configuration data.

        :param key: The key to look up in the configuration
        :param default: The default value to return if the key is not found
        :return: The value associated with the key, or the default value if key is not found
        """
        return self.config_data.get(key, default)