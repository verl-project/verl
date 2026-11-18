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

Importing this module on a storage unit nonetheless runs ``verl/__init__.py``, which costs
~30 s because it reaches ``transformers``. That lands on the first get that carries a parser,
once per storage unit process; nothing here can avoid it short of living outside the package.
"""

from typing import Any

PIXEL_KEYS = ("pixel_values", "pixel_values_videos")


def zigzag_cp_rows(seq_len: int, cp_rank: int, cp_size: int, align_size: int) -> tuple[int, range, range]:
    """The rows of one sequence a CP rank holds under the zigzag THD layout.

    Returns the rank's row count and the two source row ranges it takes, clipped to the real
    tokens: ``preprocess_thd_engine`` pads every sequence to ``align_size`` and hands rank ``r``
    one chunk from each end, so the rows past ``seq_len`` are padding and stay zero.
    """
    padded = -(-seq_len // align_size) * align_size
    chunk = padded // cp_size
    half = chunk // 2
    head = range(half * cp_rank, min(half * (cp_rank + 1), seq_len))
    tail = range(padded - half * (cp_rank + 1), min(padded - half * cp_rank, seq_len))
    return chunk, head, tail


def _zeros_like_rows(tensor, rows: int):
    """``rows`` zeroed rows shaped like ``tensor`` past its first dimension."""
    if hasattr(tensor, "new_zeros"):
        return tensor.new_zeros((rows, *tensor.shape[1:]))
    import numpy as np

    return np.zeros((rows, *tensor.shape[1:]), dtype=tensor.dtype)


def slice_for_rank(
    data: dict[str, list],
    args: Any,
    *,
    drop_multi_modal_keys: tuple[str, ...] = (),
    route_layers: list[int] | None = None,
    cp_rows: tuple[int, int, int] | None = None,
) -> dict[str, list]:
    """Keep only the part of each sample one model-parallel rank reads.

    ``drop_multi_modal_keys`` are removed from every ``multi_modal_inputs`` dict, and
    ``routed_experts`` of shape ``(seq_len, num_layers, topk)`` keeps only ``route_layers``
    and, given ``cp_rows`` of ``(cp_rank, cp_size, align_size)``, only the rows of that CP rank.
    """
    if drop_multi_modal_keys and "multi_modal_inputs" in data:
        data["multi_modal_inputs"] = [
            {k: v for k, v in inputs.items() if k not in drop_multi_modal_keys} if isinstance(inputs, dict) else inputs
            for inputs in data["multi_modal_inputs"]
        ]
    if (route_layers is not None or cp_rows is not None) and "routed_experts" in data:
        # A pipeline stage owns consecutive layers, which a basic slice copies ~20x faster than
        # the fancy indexing the general case needs.
        span = (
            (route_layers[0], route_layers[-1] + 1)
            if route_layers and route_layers == list(range(route_layers[0], route_layers[-1] + 1))
            else None
        )
        # R3 rollout can miss the route of a final generated token, which ``align_r3_router_replay_data``
        # pads back on the worker; the CP split runs on that aligned length, so take it from the inputs.
        if cp_rows is not None and "input_ids" not in data:
            raise ValueError("slicing routed_experts per CP rank needs input_ids to give the aligned length")
        lengths = [len(ids) for ids in data["input_ids"]] if cp_rows is not None else [0] * len(data["routed_experts"])
        sliced = []
        for routes, seq_len in zip(data["routed_experts"], lengths, strict=True):
            if routes is None:
                sliced.append(routes)
                continue
            if route_layers and max(route_layers) >= routes.shape[1]:
                raise ValueError(f"route layers {route_layers} exceed routed_experts layers {routes.shape[1]}")
            if route_layers is None:
                part = routes
            elif span is None:
                part = routes[:, route_layers]
            else:
                # A basic slice is a view, which would ship the whole tensor, so copy it.
                part = routes[:, span[0] : span[1]]
            if cp_rows is None:
                sliced.append(part.contiguous() if hasattr(part, "contiguous") else part.copy())
                continue
            if not 0 <= seq_len - routes.shape[0] <= 1:
                raise ValueError(f"sample has {routes.shape[0]} route rows for {seq_len} tokens")
            chunk, head, tail = zigzag_cp_rows(seq_len, *cp_rows)
            # The rank's own rows, padded like the packing does so they concatenate into its buffer.
            local = _zeros_like_rows(part, chunk)
            head = range(head.start, min(head.stop, routes.shape[0]))
            tail = range(tail.start, min(tail.stop, routes.shape[0]))
            local[: len(head)] = part[head.start : head.stop]
            local[chunk // 2 : chunk // 2 + len(tail)] = part[tail.start : tail.stop]
            sliced.append(local)
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
