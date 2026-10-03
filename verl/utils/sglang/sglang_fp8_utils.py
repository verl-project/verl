# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import os
import re
from collections.abc import Iterable
from typing import Any

from verl.utils.fp8_utils import FP8QuantizerHelper


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    get_value = getattr(config, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return getattr(config, key, default)


def _normalize_ignored_layers(ignored_layers: Any) -> list[str]:
    if ignored_layers is None:
        return []
    if isinstance(ignored_layers, str):
        ignored_layers = ignored_layers.split(",")
    elif not isinstance(ignored_layers, Iterable):
        ignored_layers = [ignored_layers]

    normalized = []
    for layer in ignored_layers:
        layer_name = str(layer).strip()
        if layer_name:
            normalized.append(layer_name)
    return normalized


def _dedupe_layers(ignored_layers: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for layer in ignored_layers:
        layer_lower = layer.lower()
        if layer_lower in seen:
            continue
        seen.add(layer_lower)
        deduped.append(layer)
    return deduped


def _get_ignored_layers_from_env() -> list[str]:
    return _normalize_ignored_layers(os.getenv("SGLANG_FP8_IGNORED_LAYERS"))


def get_sglang_fp8_ignored_layers(quant_config: Any = None) -> list[str]:
    ignored_layers = []
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "ignored_layers")))
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "modules_to_not_convert")))
    ignored_layers.extend(_get_ignored_layers_from_env())
    return _dedupe_layers(ignored_layers)


def _matches_ignored_layer(param_name: str, ignored_layer: str) -> bool:
    ignored_layer = ignored_layer.strip()
    if not ignored_layer:
        return False

    name = param_name.strip(".")
    module_name = name[: -len(".weight")] if name.lower().endswith(".weight") else name
    if ignored_layer.startswith("re:"):
        pattern = ignored_layer[3:]
        return any(re.match(pattern, candidate) for candidate in (name, module_name))

    ignored_layer = ignored_layer.lower().strip(".")
    name = name.lower()
    module_name = module_name.lower()
    for candidate in (name, module_name):
        if candidate == ignored_layer:
            return True
        if candidate.startswith(f"{ignored_layer}."):
            return True
        if candidate.endswith(f".{ignored_layer}"):
            return True
        if f".{ignored_layer}." in f".{candidate}.":
            return True
    return False


def build_sglang_fp8_quant_config(hf_config: Any = None, ignored_layers: Any = None) -> dict[str, Any]:
    """Build SGLang block-wise FP8 config shared by server init and weight sync."""
    fp8_quant_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }

    hf_quant_config = _get_config_value(hf_config, "quantization_config")
    merged_ignored_layers = get_sglang_fp8_ignored_layers(hf_quant_config)
    merged_ignored_layers.extend(_normalize_ignored_layers(ignored_layers))
    merged_ignored_layers = _dedupe_layers(merged_ignored_layers)
    if merged_ignored_layers:
        fp8_quant_config["ignored_layers"] = merged_ignored_layers

    return fp8_quant_config


class SGLangFP8QuantizerHelper(FP8QuantizerHelper):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.ignored_layers = get_sglang_fp8_ignored_layers(quant_config)

    def should_quantize_param(self, param_name):
        for ignored_layer in self.ignored_layers:
            if _matches_ignored_layer(param_name, ignored_layer):
                return False
        return super().should_quantize_param(param_name)
