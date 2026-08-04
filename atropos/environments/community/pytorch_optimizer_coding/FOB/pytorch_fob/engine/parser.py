import re
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


class YAMLParser:
    def __init__(self) -> None:
        pass

    def parse_yaml(self, file: Path) -> Any:
        """
        Opens and parses a YAML file.
        """
        with open(file, "r", encoding="utf8") as f:
            return yaml.safe_load(f)

    def parse_yamls_and_extra_args(
        self,
        default_yaml: Path,
        custom_yaml: Optional[Path],
        additional_args: Iterable[str] = tuple(),
    ) -> dict:
        """assumes that there is a dict in the yaml"""
        config_to_use = self.parse_yaml(default_yaml)
        if custom_yaml is not None:
            user_yaml = self.parse_yaml(custom_yaml)
            # merge in place
            self.merge_dicts_hierarchical(lo=config_to_use, hi=user_yaml)
        self.parse_args_into_searchspace(config_to_use, additional_args)
        return config_to_use

    def parse_args_into_searchspace(
        self, searchspace: dict[str, Any], args: Iterable[str]
    ):
        """
        Overwrites args given in the form of 'this.that=something'. Also supports lists: 'this.that[0]=something'
        """
        for arg in args:
            self._parse_arg_into_searchspace(searchspace, arg)

    def _parse_arg_into_searchspace(self, searchspace: dict[str, Any], arg: str):
        keys, value = arg.split("=")
        keys = keys.split(".")
        keys_with_list_indices = []
        for key in keys:
            match = re.search(r"^(.*?)\[(\-?\d+)\]$", key)
            if match:
                keys_with_list_indices.append(match.group(1))
                keys_with_list_indices.append(int(match.group(2)))
            else:
                keys_with_list_indices.append(key)
        target = searchspace
        for key in keys_with_list_indices[:-1]:
            if isinstance(target, dict) and key not in target:
                target[key] = {}
            target = target[key]
        target[keys_with_list_indices[-1]] = yaml.safe_load(value)

    def merge_dicts_hierarchical(self, lo: dict, hi: dict):
        """
        Overwrites values in `lo` with values from `hi` if they are present in both/
        """
        for k, v in hi.items():
            if isinstance(v, dict) and isinstance(lo.get(k, None), dict):
                self.merge_dicts_hierarchical(lo[k], v)
            else:
                lo[k] = v
