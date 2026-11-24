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
"""Helpers for frozen Engram / PLE n-gram embedding weight export and rollout safetensors loading."""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

_ENGRAM_TABLE_SCOPES = (
    # Qwen4-Exp PLE n-gram embedding shards and runtime/FP8 scale variants.
    re.compile(
        r"^(?P<scope>(?:.+\.)?ngram_embedding(?:\.shard_\d+)?)\.(?:weight|weight_scale|scale|weight_scale_inv)$"
    ),
    # DeepSeek-V4.1 Engram embedding lookup table and block scale.
    re.compile(
        r"^(?P<scope>(?:.+\.)?engram(?:\.\d+)?\.(?:embed|embed_tokens))\.(?:weight|scale|weight_scale|weight_scale_inv)$"
    ),
    # Standalone n-gram embedding table collection (e.g. ngram_embeddings.tables.{i}.weight).
    re.compile(
        r"^(?P<scope>(?:.+\.)?(?:engram|ngram_embedding|ngram_embeddings)\.tables\.\d+)\.(?:weight|scale|weight_scale|weight_scale_inv)$"
    ),
)

_PLE_HASH_BUFFER_PATTERN = re.compile(
    r"(?:^|\.)ple\.ple_embedding\.(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)$"
)

_LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_NATURAL_SPLIT_PATTERN = re.compile(r"(\d+)")
_MID_WRAPPER_SEGMENTS = ("_orig_mod.", "_fsdp_wrapped_module.")


def _strip_wrapper_prefixes(name: str) -> str:
    changed = True
    while changed:
        changed = False
        if name.startswith("module."):
            name = name.removeprefix("module.")
            changed = True
        for seg in _MID_WRAPPER_SEGMENTS:
            if name.startswith(seg):
                name = name[len(seg) :]
                changed = True
            elif f".{seg}" in name:
                name = name.replace(f".{seg}", ".")
                changed = True
    return name


def _engram_table_scope(name: str) -> str | None:
    clean_name = _strip_wrapper_prefixes(name)
    for pattern in _ENGRAM_TABLE_SCOPES:
        match = pattern.match(clean_name)
        if match is not None:
            return match.group("scope")
    return None


def canonical_engram_scope(name: str) -> str | None:
    """Return a normalized Engram / PLE table scope across runtime and HF checkpoint key layouts."""
    scope = _engram_table_scope(name)
    if scope is None:
        return None
    scope = scope.replace(".engram.embed_tokens", ".engram.embed")
    scope = scope.replace("language_model.model.", "model.language_model.")
    if scope.startswith("model."):
        scope = scope.removeprefix("model.")
    return scope


def is_engram_table_param(name: str) -> bool:
    """Return whether ``name`` identifies an Engram or PLE n-gram lookup table weight/scale."""
    return _engram_table_scope(name) is not None


def is_engram_safetensors_key(name: str) -> bool:
    """Return whether a checkpoint ``safetensors`` key belongs to an Engram / PLE lookup table."""
    if is_engram_table_param(name):
        return True
    clean_name = _strip_wrapper_prefixes(name)
    return _PLE_HASH_BUFFER_PATTERN.search(clean_name) is not None


def collect_frozen_engram_keys(
    module: torch.nn.Module,
    state_dict_keys: Iterable[str] | None = None,
    *,
    key_converter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> set[str]:
    """Collect ``state_dict`` keys belonging to frozen Engram / PLE lookup tables.

    ``module.state_dict()`` detaches tensors, which clears ``requires_grad`` on
    every returned tensor. Inspect ``module.named_parameters()`` up front so
    callers can skip frozen Engram/PLE ``DTensor`` shards *before* triggering
    ``full_tensor()`` all-gather collectives. Companion scale buffers under the
    same frozen table scope are included as well.
    """
    raw_params = dict(module.named_parameters())
    converted_names: dict[str, str]
    if key_converter is not None and raw_params:
        placeholders = {name: None for name in raw_params}
        converted_map = key_converter(placeholders)
        if len(converted_map) == len(raw_params):
            converted_names = dict(zip(raw_params.keys(), converted_map.keys(), strict=True))
        else:
            converted_names = {}
            for name in raw_params:
                converted = key_converter({name: None})
                if converted:
                    converted_names[name] = next(iter(converted))
    else:
        converted_names = {name: name for name in raw_params}

    frozen_keys: set[str] = set()
    frozen_scopes: set[str] = set()
    trainable_scopes: set[str] = set()

    for raw_name, param in raw_params.items():
        mapped_name = converted_names.get(raw_name, raw_name)
        for candidate in (raw_name, mapped_name, _strip_wrapper_prefixes(mapped_name)):
            scope = canonical_engram_scope(candidate)
            if scope is not None:
                break
        else:
            continue

        if param.requires_grad:
            trainable_scopes.add(scope)
        else:
            frozen_keys.add(raw_name)
            frozen_keys.add(mapped_name)
            frozen_keys.add(_strip_wrapper_prefixes(raw_name))
            frozen_keys.add(_strip_wrapper_prefixes(mapped_name))
            frozen_scopes.add(scope)

    pure_frozen_scopes = frozen_scopes - trainable_scopes
    if state_dict_keys is not None and pure_frozen_scopes:
        for key in state_dict_keys:
            clean_key = _strip_wrapper_prefixes(key)
            if key in frozen_keys or clean_key in frozen_keys:
                frozen_keys.add(key)
                continue
            scope = canonical_engram_scope(clean_key)
            if scope is not None and scope in pure_frozen_scopes:
                frozen_keys.add(key)
                frozen_keys.add(clean_key)

    return frozen_keys


def find_model_engram_parameters(
    models: Iterable[torch.nn.Module],
) -> list[tuple[torch.nn.Module, str, torch.nn.Parameter]]:
    """Return ``(model, qualname, param)`` for all Engram / PLE table parameters on ``models``."""
    found: list[tuple[torch.nn.Module, str, torch.nn.Parameter]] = []
    for model in models:
        if model is None:
            continue
        for name, param in model.named_parameters():
            if is_engram_table_param(name):
                found.append((model, name, param))
    return found


def extract_engram_layer_ids(
    engram_params: Iterable[tuple[torch.nn.Module, str, torch.nn.Parameter]],
) -> set[int] | None:
    """Return local layer indices owning Engram/PLE params, or ``None`` if any name is unindexed."""
    layer_ids: set[int] = set()
    saw_any = False
    for _model, name, _param in engram_params:
        saw_any = True
        match = _LAYER_ID_PATTERN.search(name)
        if match is None:
            return None
        layer_ids.add(int(match.group(1)))
    return layer_ids if saw_any else set()


def has_unpreserved_device_engram_params(
    models_with_config: Iterable[tuple[torch.nn.Module, Any]],
) -> bool:
    """Return True if any Engram/PLE table lives on an accelerator and is not preserved across level-2 sleep."""
    for model, model_config in models_with_config:
        if model is None:
            continue
        preserve_patterns = tuple(getattr(model_config, "sleep_preserve_parameter_names", None) or ())
        for name, param in model.named_parameters():
            if not is_engram_table_param(name):
                continue
            if param.is_cpu:
                continue
            if preserve_patterns and any(fnmatch.fnmatchcase(name, pat) for pat in preserve_patterns):
                continue
            return True
    return False


def _natural_sort_key(path: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in _NATURAL_SPLIT_PATTERN.split(os.path.basename(path))]


def _matches_local_layer(key: str, local_layer_ids: set[int] | None) -> bool:
    if local_layer_ids is None:
        return True
    match = _LAYER_ID_PATTERN.search(key)
    if match is None:
        return True
    return int(match.group(1)) in local_layer_ids


def _is_excluded_engram_scope(key: str, exclude_scopes: set[str] | None) -> bool:
    if not exclude_scopes:
        return False
    scope = canonical_engram_scope(key)
    return scope is not None and scope in exclude_scopes


def _resolve_engram_safetensors_plan(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    download_dir: str | None = None,
    local_layer_ids: set[int] | None = None,
    exclude_scopes: set[str] | None = None,
) -> list[tuple[str, list[str] | None]]:
    """Resolve ``(safetensors_file_path, keys_or_none)`` containing Engram / PLE tensors."""
    index_filename = "model.safetensors.index.json"

    if os.path.isdir(model_name_or_path):
        index_path = os.path.join(model_name_or_path, index_filename)
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map: dict[str, str] = json.load(f).get("weight_map", {})
            keys_by_rel_file: dict[str, list[str]] = {}
            for key, rel_file in weight_map.items():
                if (
                    is_engram_safetensors_key(key)
                    and _matches_local_layer(key, local_layer_ids)
                    and not _is_excluded_engram_scope(key, exclude_scopes)
                ):
                    keys_by_rel_file.setdefault(rel_file, []).append(key)
            return [
                (os.path.join(model_name_or_path, rel_file), keys)
                for rel_file, keys in sorted(keys_by_rel_file.items(), key=lambda item: _natural_sort_key(item[0]))
            ]

        st_files = sorted(glob.glob(os.path.join(model_name_or_path, "*.safetensors")), key=_natural_sort_key)
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_name_or_path!r} for Engram weight loading")
        return [(st_file, None) for st_file in st_files]

    # Remote HuggingFace Hub model ID: fetch index first when available so only Engram shards are downloaded.
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        index_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename=index_filename,
            revision=revision,
            cache_dir=download_dir,
        )
    except EntryNotFoundError:
        folder = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=download_dir,
            allow_patterns=["*.safetensors"],
        )
        st_files = sorted(glob.glob(os.path.join(folder, "*.safetensors")), key=_natural_sort_key)
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in repo {model_name_or_path!r}") from None
        return [(st_file, None) for st_file in st_files]

    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    keys_by_rel_file = {}
    for key, rel_file in weight_map.items():
        if (
            is_engram_safetensors_key(key)
            and _matches_local_layer(key, local_layer_ids)
            and not _is_excluded_engram_scope(key, exclude_scopes)
        ):
            keys_by_rel_file.setdefault(rel_file, []).append(key)

    plan: list[tuple[str, list[str] | None]] = []
    for rel_file, keys in sorted(keys_by_rel_file.items(), key=lambda item: _natural_sort_key(item[0])):
        local_shard = hf_hub_download(
            repo_id=model_name_or_path,
            filename=rel_file,
            revision=revision,
            cache_dir=download_dir,
        )
        plan.append((local_shard, keys))
    return plan


def iter_engram_safetensors_batches(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    download_dir: str | None = None,
    local_layer_ids: set[int] | None = None,
    exclude_scopes: Iterable[str] | None = None,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Stream Engram / PLE tensors from ``safetensors`` one shard file at a time.

    Each yielded batch stays inside its ``safe_open`` context while the caller
    copies the rank-local slice into the rollout model, so at most one shard
    file's mmap is active at any point in time.
    """
    from safetensors import safe_open

    normalized_exclude: set[str] | None = None
    if exclude_scopes:
        normalized_exclude = set()
        for scope_or_key in exclude_scopes:
            normalized_exclude.add(canonical_engram_scope(scope_or_key) or scope_or_key)

    plan = _resolve_engram_safetensors_plan(
        model_name_or_path,
        revision=revision,
        download_dir=download_dir,
        local_layer_ids=local_layer_ids,
        exclude_scopes=normalized_exclude,
    )
    for st_file, expected_keys in plan:
        with safe_open(st_file, framework="pt", device="cpu") as sf:
            if expected_keys is None:
                candidate_keys = [
                    key
                    for key in sf.keys()
                    if is_engram_safetensors_key(key)
                    and _matches_local_layer(key, local_layer_ids)
                    and not _is_excluded_engram_scope(key, normalized_exclude)
                ]
            else:
                available = set(sf.keys())
                candidate_keys = [key for key in expected_keys if key in available]
            if not candidate_keys:
                continue
            yield [(key, sf.get_tensor(key)) for key in candidate_keys]
