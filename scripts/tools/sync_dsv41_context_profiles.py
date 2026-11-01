#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Add the 1.5M/2M/4M DSV4.1 context profiles to a shared TOML file.

The script copies the existing ``sft-geo3k-cp32-ctx1m`` section and changes
only its sequence-length field and both parquet paths. It is idempotent and
keeps the profile file's surrounding comments and formatting. The generated
parquet is used for both train and validation because the context probe is a
throughput/memory stress dataset, not a held-out quality set.

Example::

    python scripts/tools/sync_dsv41_context_profiles.py \
        --profile /shared/dsv41_profiles.toml \
        --data-dir /shared/data/ctxprobe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

TARGETS = (
    ("sft-geo3k-cp32-ctx1_5m", 1_500_000, "trainreal1_5m.parquet"),
    ("sft-geo3k-cp32-ctx2m", 2_000_000, "trainreal2m.parquet"),
    ("sft-geo3k-cp32-ctx4m", 4_000_000, "trainreal4m.parquet"),
)
BASE_PROFILE = "sft-geo3k-cp32-ctx1m"


def _section(text: str, name: str) -> re.Match[str] | None:
    pattern = rf"(?ms)^\[{re.escape(name)}\][^\n]*\n.*?(?=^\[|\Z)"
    return re.search(pattern, text)


def _replace_assignment(block: str, key: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*.*$"
    replacement = f"{key} = {value}"
    block, count = re.subn(pattern, replacement, block, count=1)
    if count != 1:
        raise ValueError(f"profile section is missing exactly one {key!r} assignment")
    return block


def _render_block(base_block: str, name: str, seq_len: int, parquet: Path) -> str:
    block = re.sub(r"^\[[^\]]+\]", f"[{name}]", base_block, count=1, flags=re.MULTILINE)
    block = _replace_assignment(block, "seq_len", str(seq_len))
    quoted_path = json.dumps(str(parquet))
    block = _replace_assignment(block, "train_files", quoted_path)
    block = _replace_assignment(block, "val_files", quoted_path)
    return block.rstrip() + "\n\n"


def sync_profile(profile: Path, data_dir: Path, *, check_data: bool = True) -> None:
    text = profile.read_text()
    base = _section(text, BASE_PROFILE)
    if base is None:
        raise ValueError(f"profile does not contain [{BASE_PROFILE}]")

    rendered = {
        name: _render_block(base.group(0), name, seq_len, data_dir / filename) for name, seq_len, filename in TARGETS
    }
    if check_data:
        missing = [path for path in (data_dir / filename for _, _, filename in TARGETS) if not path.is_file()]
        if missing:
            paths = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"generated parquet is missing:\n{paths}")

    updated = text
    for name, _, _ in TARGETS:
        existing = _section(updated, name)
        if existing is None:
            updated = updated.rstrip() + "\n\n" + rendered[name]
        else:
            updated = updated[: existing.start()] + rendered[name] + updated[existing.end() :]

    if updated == text:
        print(f"profile already synchronized: {profile}")
        return

    profile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=profile.parent, delete=False) as handle:
        handle.write(updated)
        temporary = Path(handle.name)
    os.replace(temporary, profile)
    print(f"updated {profile} with {', '.join(name for name, _, _ in TARGETS)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Shared dsv41_profiles.toml")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing trainreal1_5m.parquet, trainreal2m.parquet and trainreal4m.parquet",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Render profile paths without requiring the generated parquet files yet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sync_profile(args.profile, args.data_dir, check_data=not args.allow_missing_data)
