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

"""Opt-in identities for the built-in, local text RLHF dataset and v1 resume.

The manifest detects changed inputs; it does not restore sampling decisions or
promise deterministic training. No raw rows, vocabulary or absolute paths are saved.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
MANIFEST_NAME = "data_manifest.json"
_VERSION = 1


def resume_data_check(config) -> str:
    """Read and validate the opt-in data.resume_check policy."""
    policy = config.get("resume_check", "off")
    if policy not in ("off", "warn", "error"):
        raise ValueError("data.resume_check must be one of: off, warn, error")
    return policy


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _stat(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def snapshot_sources(paths) -> list[dict]:
    """Hash local sources in bounded memory before constructing the dataset."""
    result = []
    for source in paths:
        path = Path(source)
        if "://" in str(source) or not path.is_file() or path.suffix not in (".parquet", ".json", ".jsonl"):
            raise ValueError("Resume identity currently requires local parquet/JSON/JSONL files")
        before = _stat(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if _stat(path) != before:
            raise ValueError("Data source changed while computing its resume identity")
        result.append({"path": str(path), "stat": before, "format": path.suffix, "sha256": digest.hexdigest()})
    return result


def build_rlhf_signature(dataset, sources: list[dict]) -> dict:
    """Describe the loaded rows, input bytes and effective text preprocessing."""
    from omegaconf import OmegaConf
    from transformers import PreTrainedTokenizerFast

    if dataset.processor is not None or not isinstance(dataset.tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Resume identity currently requires text data and a fast tokenizer")
    if dataset.chat_template_func is not None:
        raise ValueError("Custom chat_template_func is not supported by resume identity")
    if (dataset.tool_config_path or dataset.function_tool_path) and dataset.tool_schemas is None:
        raise ValueError("Cannot verify resume identity after tool schema initialization failed")
    tokenizer = dataset.tokenizer
    tokenizer_state = {
        "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "chat_template": tokenizer.chat_template,
        "special_tokens": {
            key: [str(token) for token in value] if isinstance(value, list) else str(value)
            for key, value in tokenizer.special_tokens_map.items()
        },
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
    }
    rows_hash = hashlib.sha256()
    for row in dataset.dataframe:
        if any(row.get(key) for key in (dataset.image_key, dataset.video_key, dataset.audio_key)):
            raise ValueError("Resume identity currently supports text-only rows")
        if any(not isinstance(message.get("content"), str) for message in row[dataset.prompt_key]):
            raise ValueError("Resume identity currently requires plain-text message content")
        rows_hash.update(_json_bytes(row))
        rows_hash.update(b"\n")
    processing = {
        name: getattr(dataset, name)
        for name in (
            "prompt_key",
            "max_prompt_length",
            "truncation",
            "return_raw_chat",
            "return_full_prompt",
            "filter_overlong_prompts",
            "max_samples",
            "shuffle",
            "seed",
            "need_tools_kwargs",
            "return_multi_modal_inputs",
        )
    }
    processing["return_raw_input_ids"] = dataset.config.get("return_raw_input_ids", False)
    kwargs = dataset.apply_chat_template_kwargs
    kwargs = OmegaConf.to_container(kwargs, resolve=True) if OmegaConf.is_config(kwargs) else kwargs
    processing["apply_chat_template_kwargs_sha256"] = hashlib.sha256(_json_bytes(kwargs)).hexdigest()
    processing["tool_schemas_sha256"] = hashlib.sha256(_json_bytes(dataset.tool_schemas)).hexdigest()
    for source in sources:
        if _stat(source["path"]) != source["stat"]:
            raise ValueError("Data source changed while constructing the dataset")
    return {
        "files": [{key: item[key] for key in ("format", "sha256")} for item in sources],
        "rows": {"count": len(dataset.dataframe), "sha256": rows_hash.hexdigest()},
        "tokenizer_sha256": hashlib.sha256(_json_bytes(tokenizer_state)).hexdigest(),
        "processing": processing,
    }


def get_data_manifest(dataloader, policy: str) -> dict | None:
    """Capture a supported dataset identity, or explicitly report it as unavailable."""
    if policy == "off":
        return None
    try:
        signature = dataloader.dataset.get_resume_signature()
        if not isinstance(signature, dict) or not signature:
            raise ValueError("Dataset did not provide a resume signature")
        # Validate serialization before any model checkpoint write is started.
        manifest = {"schema_version": _VERSION, "dataset": signature}
        return json.loads(_json_bytes(manifest))
    except (AttributeError, NotImplementedError, TypeError, ValueError) as error:
        message = f"Data resume identity is unavailable: {error}"
        if policy == "error":
            raise ValueError(message) from error
        logger.warning(message)
        return {"schema_version": _VERSION, "dataset": None}


def write_data_manifest(checkpoint_dir, manifest: dict | None) -> None:
    """Publish a local identity atomically, or invalidate an older identity with None."""
    directory = Path(checkpoint_dir)
    if manifest is None:
        (directory / MANIFEST_NAME).unlink(missing_ok=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".data_manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / MANIFEST_NAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_data_manifest(checkpoint_dir, dataloader, policy: str) -> None:
    """Check identity before worker or dataloader restore; off leaves legacy behavior intact."""
    if policy == "off":
        return
    current = get_data_manifest(dataloader, policy)
    path = Path(checkpoint_dir) / MANIFEST_NAME
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(saved, dict)
            or type(saved.get("schema_version")) is not int
            or saved["schema_version"] != _VERSION
        ):
            raise ValueError("Unsupported data manifest schema version")
        if not isinstance(saved.get("dataset"), dict) or current["dataset"] is None:
            raise ValueError("Checkpoint or current dataset has no verifiable data identity")
        if saved["dataset"] != current["dataset"]:
            fields = sorted(
                key
                for key in saved["dataset"].keys() | current["dataset"].keys()
                if key not in saved["dataset"]
                or key not in current["dataset"]
                or saved["dataset"][key] != current["dataset"][key]
            )
            raise ValueError(f"Data identity changed in: {', '.join(fields)}")
    except (OSError, ValueError, TypeError) as error:
        message = f"Cannot verify checkpoint data identity: {error}"
        if policy == "error":
            raise ValueError(message) from error
        logger.warning(message)
