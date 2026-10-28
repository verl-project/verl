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

"""Layer selection shared by Megatron low-precision training recipes."""

from typing import Any, Mapping


def hf_config_get(hf_config: Any, name: str, default: Any = None) -> Any:
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


def text_config(hf_config: Any) -> Any:
    """The decoder config; multimodal HF configs nest it under ``text_config``."""
    config = hf_config_get(hf_config, "text_config")
    return hf_config if config is None else config


def num_decoder_layers(hf_config: Any) -> int:
    return int(hf_config_get(text_config(hf_config), "num_hidden_layers") or 0)


def bf16_layers_at_start_end(override_transformer_config: Mapping[str, Any]) -> tuple[int, int]:
    """Decoder layers Megatron-Core keeps in BF16 at the start and end (``first_last_layers_bf16``)."""
    if not override_transformer_config.get("first_last_layers_bf16", False):
        return 0, 0
    # Megatron-Core defaults
    start = override_transformer_config.get("num_layers_at_start_in_bf16", 1)
    end = override_transformer_config.get("num_layers_at_end_in_bf16", 1)
    if start < 0 or end < 0:
        raise ValueError(f"BF16 layer carve-out must be non-negative, got {start}/{end}")
    return start, end


def bf16_layer_indices(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> list[int]:
    """Decoder layers kept in BF16 by the first/last carve-out."""
    if num_layers_at_start < 0 or num_layers_at_end < 0:
        raise ValueError(f"BF16 layer carve-out must be non-negative, got {num_layers_at_start}/{num_layers_at_end}")
    if num_layers_at_start + num_layers_at_end >= num_layers:
        raise ValueError(
            f"BF16 layer carve-out {num_layers_at_start}/{num_layers_at_end} leaves no quantized layer of {num_layers}"
        )
    return list(range(num_layers_at_start)) + list(range(num_layers - num_layers_at_end, num_layers))
