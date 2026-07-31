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

import json
from pathlib import Path, PurePosixPath

_CONFIG_NAME = "config.json"
_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _safe_shard_path(target_dir: Path, shard_name: str) -> Path | None:
    shard_path = PurePosixPath(shard_name)
    if shard_path.is_absolute() or not shard_path.parts or ".." in shard_path.parts:
        return None
    return target_dir.joinpath(*shard_path.parts)


def validate_hf_model_output(target_dir: str | Path) -> None:
    """Raise when a model merger output is not a complete Hugging Face model."""
    target_dir = Path(target_dir)
    errors = []

    config_path = target_dir / _CONFIG_NAME
    if not _is_nonempty_file(config_path):
        errors.append(f"missing or empty {_CONFIG_NAME}")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                errors.append(f"{_CONFIG_NAME} must contain a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {_CONFIG_NAME}: {exc}")

    has_complete_weights = any(_is_nonempty_file(target_dir / name) for name in _WEIGHT_NAMES)
    index_errors = []
    for index_name in _WEIGHT_INDEX_NAMES:
        index_path = target_dir / index_name
        if not index_path.exists():
            continue

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            index_errors.append(f"invalid {index_name}: {exc}")
            continue

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            index_errors.append(f"{index_name} has no non-empty weight_map")
            continue

        shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in shard_names):
            index_errors.append(f"{index_name} contains invalid shard names")
            continue

        shard_names = set(shard_names)
        shard_paths = {name: _safe_shard_path(target_dir, name) for name in shard_names}
        unsafe_shards = sorted(name for name, path in shard_paths.items() if path is None)
        if unsafe_shards:
            index_errors.append(f"{index_name} contains unsafe shard paths: {unsafe_shards}")
            continue

        missing_shards = sorted(name for name, path in shard_paths.items() if not _is_nonempty_file(path))
        if missing_shards:
            index_errors.append(f"{index_name} references missing or empty shards: {missing_shards}")
            continue

        has_complete_weights = True

    if not has_complete_weights:
        errors.extend(index_errors)
        expected = ", ".join((*_WEIGHT_NAMES, *_WEIGHT_INDEX_NAMES))
        errors.append(f"no complete model weights; expected one of: {expected}")

    if errors:
        details = "; ".join(errors)
        raise RuntimeError(f"Incomplete Hugging Face model output at {target_dir}: {details}")
