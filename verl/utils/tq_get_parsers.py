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
Importing one still runs ``verl/__init__.py`` and through it ``transformers``, a few seconds
that a storage unit pays once, on the first get that carries a parser.
"""

from typing import Any


def zigzag_cp_rows(seq_len: int, cp_rank: int, cp_size: int, align_size: int) -> tuple[int, range, range]:
    """Rows of one sequence a CP rank holds under the zigzag THD layout: its row count and two ranges.

    ``preprocess_thd_engine`` pads the sequence to ``align_size`` and hands rank ``r`` one chunk
    from each end, so the returned ranges may reach past ``seq_len`` into padding.
    """
    padded = -(-seq_len // align_size) * align_size
    chunk = padded // cp_size
    half = chunk // 2
    return (
        chunk,
        range(half * cp_rank, half * (cp_rank + 1)),
        range(padded - half * (cp_rank + 1), padded - half * cp_rank),
    )


def slice_for_rank(
    data: dict[str, list],
    args: Any,
    *,
    route_layers: list[int] | None = None,
    cp_rows: tuple[int, int, int] | None = None,
) -> dict[str, list]:
    """Keep only the part of each sample one model-parallel rank reads.

    ``routed_experts`` of shape ``(seq_len, num_layers, topk)`` keeps only ``route_layers``, and,
    given ``cp_rows`` of ``(cp_rank, cp_size, align_size)``, only the rows of that CP rank.
    """
    if route_layers is None and cp_rows is None or "routed_experts" not in data:
        return data

    # A pipeline stage owns consecutive layers, which a basic slice copies ~20x faster than fancy indexing.
    if route_layers is None:
        layers = slice(None)
    elif route_layers and route_layers == list(range(route_layers[0], route_layers[-1] + 1)):
        layers = slice(route_layers[0], route_layers[-1] + 1)
    else:
        layers = route_layers
    # R3 rollout can miss the last token's route, so the padded length comes from input_ids instead.
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
        part = routes[:, layers]
        if cp_rows is None:
            # A basic slice is a view, which would ship the whole tensor.
            sliced.append(part.contiguous())
            continue
        if not 0 <= seq_len - routes.shape[0] <= 1:
            raise ValueError(f"sample has {routes.shape[0]} route rows for {seq_len} tokens")
        chunk, head, tail = zigzag_cp_rows(seq_len, *cp_rows)
        # The rank's own rows, zero-padded like the packing so they concatenate into its buffer.
        local = part.new_zeros((chunk, *part.shape[1:]))
        for offset, rows in ((0, head), (chunk // 2, tail)):
            stop = min(rows.stop, routes.shape[0])
            local[offset : offset + max(stop - rows.start, 0)] = part[rows.start : stop]
        sliced.append(local)
    data["routed_experts"] = sliced
    return data
