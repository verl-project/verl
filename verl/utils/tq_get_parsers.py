# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""``data_parser`` callables for ``TransferQueue`` gets.

They run on the SimpleStorage units, which unpickle them by reference, so they must stay
importable without heavy dependencies and must not modify the stored values they receive.
"""

from typing import Any

PIXEL_KEYS = ("pixel_values", "pixel_values_videos")


def slice_for_rank(
    data: dict[str, list],
    args: Any,
    *,
    drop_multi_modal_keys: tuple[str, ...] = (),
    route_layers: list[int] | None = None,
) -> dict[str, list]:
    """Keep only the part of each sample one model-parallel rank reads.

    ``drop_multi_modal_keys`` are removed from every ``multi_modal_inputs`` dict, and
    ``routed_experts`` of shape ``(seq_len, num_layers, topk)`` keeps only ``route_layers``.
    """
    if drop_multi_modal_keys and "multi_modal_inputs" in data:
        data["multi_modal_inputs"] = [
            {k: v for k, v in inputs.items() if k not in drop_multi_modal_keys} if isinstance(inputs, dict) else inputs
            for inputs in data["multi_modal_inputs"]
        ]
    if route_layers is not None and "routed_experts" in data:
        # A pipeline stage owns consecutive layers, which a basic slice copies ~20x faster than
        # the fancy indexing the general case needs.
        span = (
            (route_layers[0], route_layers[-1] + 1)
            if route_layers and route_layers == list(range(route_layers[0], route_layers[-1] + 1))
            else None
        )
        sliced = []
        for routes in data["routed_experts"]:
            if routes is None:
                sliced.append(routes)
                continue
            if route_layers and max(route_layers) >= routes.shape[1]:
                raise ValueError(f"route layers {route_layers} exceed routed_experts layers {routes.shape[1]}")
            if span is None:
                sliced.append(routes[:, route_layers])
            else:
                # A basic slice is a view, which would ship the whole tensor, so copy it.
                part = routes[:, span[0] : span[1]]
                sliced.append(part.contiguous() if hasattr(part, "contiguous") else part.copy())
        data["routed_experts"] = sliced
    return data


def select_multi_modal_rows(data: dict[str, list], args: list | None) -> dict[str, list]:
    """Return, for each sample, rows ``[start, end)`` of the ``multi_modal_inputs`` keys it asks for.

    Each ``args`` entry is a list of ``[key, start, end]`` triples; the result holds only those keys.
    """
    data["multi_modal_inputs"] = [
        {key: inputs[key][start:end] for key, start, end in ranges}
        for inputs, ranges in zip(data["multi_modal_inputs"], args, strict=True)
    ]
    return data
