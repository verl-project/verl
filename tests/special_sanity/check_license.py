# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

license_head_bytedance = "Copyright 2024 Bytedance Ltd. and/or its affiliates"
license_head_bytedance_25 = "Copyright 2025 Bytedance Ltd. and/or its affiliates"
license_head_bytedance_26 = "Copyright 2026 Bytedance Ltd. and/or its affiliates"
# Add custom license headers below
license_head_prime = "Copyright 2024 PRIME team and/or its affiliates"
license_head_individual = "Copyright 2025 Individual Contributor:"
license_head_sglang = "Copyright 2023-2024 SGLang Team"
license_head_modelbest = "Copyright 2025 ModelBest Inc. and/or its affiliates"
license_head_amazon = "Copyright 2025 Amazon.com Inc and/or its affiliates"
license_head_facebook = "Copyright (c) 2016-     Facebook, Inc"
license_head_meituan = "Copyright 2025 Meituan Ltd. and/or its affiliates"
license_head_huawei = "Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved."
license_headers = [
    license_head_bytedance,
    license_head_bytedance_25,
    license_head_bytedance_26,
    license_head_prime,
    license_head_individual,
    license_head_sglang,
    license_head_modelbest,
    license_head_amazon,
    license_head_facebook,
    license_head_meituan,
    license_head_huawei,
]


def get_py_files(path_arg: Path) -> Iterable[Path]:
    """get py files under a dir. if already py file return it

    Args:
        path_arg (Path): path to scan for py files

    Returns:
        Iterable[Path]: list of py files
    """
    if path_arg.is_dir():
        return path_arg.glob("**/*.py")
    elif path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="List of directories to check for license headers",
    )
    args = parser.parse_args()

    # Collect all Python files from specified directories
    pathlist = set(path for path_arg in args.directories for path in get_py_files(path_arg))

    for path in pathlist:
        # because path is object not string
        path_in_str = str(path.absolute())
        print(path_in_str)
        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            has_license = False
            for lh in license_headers:
                if lh in file_content:
                    has_license = True
                    break
            assert has_license, f"file {path_in_str} does not contain license"
