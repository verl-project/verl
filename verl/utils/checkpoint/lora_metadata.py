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

import json
import os
import warnings
from typing import Any, Optional

LORA_TRAIN_META_FILENAME = "lora_train_meta.json"


def _get_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_lora_train_meta(model_config: Any) -> Optional[dict[str, object]]:
    """Build LoRA rank/alpha metadata from an HF or Megatron model config."""
    lora_config = _get_value(model_config, "lora", {}) or {}

    flat_rank = _to_int(_get_value(model_config, "lora_rank", 0))
    nested_rank = _to_int(_get_value(lora_config, "rank", 0))
    flat_adapter_path = _get_value(model_config, "lora_adapter_path", None)
    nested_adapter_path = _get_value(lora_config, "adapter_path", None)

    if flat_rank <= 0 and nested_rank <= 0 and flat_adapter_path is None and nested_adapter_path is None:
        return None

    if flat_rank > 0 or flat_adapter_path is not None:
        lora_rank = flat_rank
        raw_lora_alpha = _get_value(model_config, "lora_alpha", 0)
    else:
        lora_rank = nested_rank
        raw_lora_alpha = _get_value(lora_config, "alpha", 0)

    task_type = _get_value(model_config, "task_type", None) or "CAUSAL_LM"

    return {
        "r": lora_rank,
        "lora_alpha": _to_int(raw_lora_alpha),
        "task_type": str(task_type),
    }


def load_lora_train_meta(local_dir: str | os.PathLike | None) -> Optional[dict[str, object]]:
    if not local_dir:
        return None

    meta_path = os.path.join(str(local_dir), LORA_TRAIN_META_FILENAME)
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            lora_meta = json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read LoRA metadata from {meta_path}: {e}", stacklevel=2)
        return None

    result = {}
    if "r" in lora_meta:
        try:
            result["r"] = int(lora_meta["r"])
        except (TypeError, ValueError):
            warnings.warn(f"Invalid LoRA rank in {meta_path}: {lora_meta['r']}", stacklevel=2)

    if "lora_alpha" in lora_meta:
        try:
            result["lora_alpha"] = int(lora_meta["lora_alpha"])
        except (TypeError, ValueError):
            warnings.warn(f"Invalid lora_alpha in {meta_path}: {lora_meta['lora_alpha']}", stacklevel=2)

    if "task_type" in lora_meta:
        task_type = lora_meta["task_type"]
        if task_type is None:
            pass
        elif isinstance(task_type, str):
            result["task_type"] = task_type
        else:
            warnings.warn(f"Invalid task_type in {meta_path}: {task_type}", stacklevel=2)

    return result if len(result) > 0 else None


def save_lora_train_meta(
    model_config: Any,
    local_dir: str | os.PathLike,
    hdfs_dir: str | os.PathLike | None = None,
) -> Optional[str]:
    """Persist LoRA rank/alpha metadata beside a checkpoint and optionally copy it to HDFS."""
    lora_train_meta = get_lora_train_meta(model_config)
    if lora_train_meta is None:
        return None

    os.makedirs(local_dir, exist_ok=True)
    meta_path = os.path.join(str(local_dir), LORA_TRAIN_META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(lora_train_meta, f, ensure_ascii=False, indent=4)

    if hdfs_dir is not None:
        from verl.utils import hdfs_io

        hdfs_io.makedirs(str(hdfs_dir), exist_ok=True)
        hdfs_io.copy(meta_path, os.path.join(str(hdfs_dir), LORA_TRAIN_META_FILENAME))

    return meta_path
