# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Prepare a separate checkpoint view using the current training model code."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("target", type=Path)
parser.add_argument("--code", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve(strict=True)
target = args.target
if target.resolve() == source:
    raise ValueError("Model view must be separate from the source checkpoint")
target.mkdir(parents=True, exist_ok=True)
for src in source.iterdir():
    dst = target / src.name
    if src.is_file() and src.suffix == ".safetensors":
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"Unexpected weight link: {dst}")
        elif not dst.exists():
            dst.symlink_to(src.resolve())
    elif src.is_file() and src.suffix in {".json", ".py", ".jinja", ".txt", ".model", ".tiktoken"}:
        if dst.is_symlink():
            raise ValueError(f"Metadata must not be a symlink: {dst}")
        shutil.copy2(src, dst)
for src in args.code.glob("*.py"):
    dst = target / src.name
    if dst.is_symlink():
        raise ValueError(f"Model code must not be a symlink: {dst}")
    shutil.copy2(src, dst)

# HF's local remote-code loader caches immediate flat relative modules. Export
# the native package's implementation as that module, without an extra shim.
dst = target / "kimi_k3.py"
if dst.is_symlink():
    raise ValueError(f"Model code must not be a symlink: {dst}")
shutil.copy2(args.code / "kimi_k3/checkpoint.py", dst)
config_path = target / "config.json"
config = json.loads(config_path.read_text())
config["load_checkpoint_after_shard"] = True
config_path.write_text(json.dumps(config, indent=2) + "\n")
