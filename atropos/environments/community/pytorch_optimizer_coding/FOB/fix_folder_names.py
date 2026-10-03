"""
This tool fixes folder names which are incorrect due to changes in the default.yaml
"""

import argparse
import sys
from pathlib import Path
from pprint import pprint

import yaml
from pytorch_fob.engine.engine import Engine


def deep_diff(dict1, dict2):
    diff = {}

    # Check keys in dict1 but not in dict2
    for key in dict1:
        if key not in dict2:
            if dict1[key] is not None:
                diff[key] = {"old_value": dict1[key], "new_value": None}
        elif dict1[key] != dict2[key]:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = deep_diff(dict1[key], dict2[key])
                if nested_diff:
                    diff[key] = nested_diff
            else:
                diff[key] = {"old_value": dict1[key], "new_value": dict2[key]}

    # Check keys in dict2 but not in dict1
    for key in dict2:
        if key not in dict1:
            if dict2[key] is not None:
                diff[key] = {"old_value": None, "new_value": dict2[key]}

    return diff


def fix_recursive(path: Path, dry_run: bool, ignore_config_diff: bool):
    for file in path.iterdir():
        if file.name == "config.yaml":
            e = Engine()
            e.parse_experiment_from_file(file)
            runs = list(e.runs())
            if len(runs) != 1:
                print("Error config.yaml is invalid:", file, file=sys.stderr)
                sys.exit(1)
            r = runs[0]
            target = r.run_dir.name
            actual = file.parent.name
            if actual == target:
                continue
            target = file.parent.parent / target
            actual = file.parent
            print(
                "folder name is wrong and needs fixing:\ncurrent:",
                actual,
                "\ncalculated:",
                target,
            )
            computed_config = r.export_config_dict()
            with open(file, "r", encoding="utf8") as f:
                actual_config = yaml.safe_load(f)
            clean_computed_config = {
                "engine": {"devices": computed_config["engine"]["devices"]},
                "task": computed_config["task"],
                "optimizer": computed_config["optimizer"],
            }
            clean_actual_config = {
                "engine": {"devices": actual_config["engine"]["devices"]},
                "task": actual_config["task"],
                "optimizer": actual_config["optimizer"],
            }
            diff = deep_diff(clean_actual_config, clean_computed_config)
            if diff and (not ignore_config_diff):
                print("warning config dict differs!:")
                pprint(diff)
                print("skipping folder!")
                continue
            if not dry_run:
                print("renaming...")
                if target.exists():
                    print("target path already exists, skipping...")
                    continue
                actual.rename(target)

        elif file.is_dir():
            fix_recursive(file, dry_run, ignore_config_diff)


def main(args: argparse.Namespace):
    base_folder: Path = args.base_folder
    ignore_config_diff = args.ignore_config_diff
    if ignore_config_diff:
        res = input(
            "WARNING: ignoring the config dict diffs can be dangerous do you know what you are doing? [y/n]"
        )
        if res.strip().lower() != "y":
            return
    fix_recursive(base_folder, args.dry_run, ignore_config_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folder name fix tool")
    parser.add_argument(
        "base_folder", type=Path, help="Folder with experiments, will check recursively"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just print what would be changed, do not change any files",
    )
    parser.add_argument(
        "--ignore_config_diff",
        action="store_true",
        help="Ignores config dict difference WARNING: experiment could be totally different!\
                              Use with care!",
    )
    args = parser.parse_args()
    main(args)
