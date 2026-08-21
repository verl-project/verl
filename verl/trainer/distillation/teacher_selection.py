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

from typing import Any

import torch


def align_teacher_log_prob_rows(
    teacher_log_prob_rows: Any,
    input_sequence_lengths: Any,
    *,
    teacher_key: str,
) -> list[Any]:
    """Normalize teacher outputs to the full-sequence layout consumed by the loss."""
    rows = list(teacher_log_prob_rows)
    sequence_lengths = [int(length) for length in input_sequence_lengths]
    if len(rows) != len(sequence_lengths):
        raise RuntimeError(
            f"Teacher {teacher_key!r} returned {len(rows)} rows for {len(sequence_lengths)} input sequences."
        )

    aligned_rows = []
    for row, sequence_length in zip(rows, sequence_lengths, strict=True):
        if row.shape[0] == sequence_length - 1:
            # Some next-token APIs omit the final, unused prediction position.
            row = torch.cat([row, row.new_zeros((1, *row.shape[1:]))], dim=0)
        elif row.shape[0] != sequence_length:
            raise RuntimeError(
                f"Teacher {teacher_key!r} returned sequence length {row.shape[0]} "
                f"for an input of length {sequence_length}."
            )
        aligned_rows.append(row.float().clone())
    return aligned_rows


def select_teacher_log_prob_rows(
    teacher_results: list[Any],
    route_to_teacher_idx: dict[str, int],
    route_values: Any,
    *,
    batch_size: int,
    routing_field: str,
) -> list[Any]:
    """Select one teacher result per row, bypassing routing for a single teacher."""
    if not teacher_results:
        raise ValueError("At least one teacher result is required.")

    if len(teacher_results) == 1:
        return [teacher_results[0][row] for row in range(batch_size)]

    if route_values is None:
        raise ValueError(
            f"Multi-teacher selection requires routing field {routing_field!r}, but the batch does not contain it."
        )
    if len(route_values) != batch_size:
        raise RuntimeError(
            f"Routing field {routing_field!r} has {len(route_values)} values for {batch_size} trajectories."
        )

    selected = []
    for row, route_value in enumerate(route_values):
        if hasattr(route_value, "item"):
            route_value = route_value.item()
        try:
            teacher_idx = route_to_teacher_idx[route_value]
        except KeyError as exc:
            raise ValueError(
                f"No fused teacher matches {routing_field}={route_value!r}; "
                f"configured route keys are {sorted(route_to_teacher_idx)}."
            ) from exc
        selected.append(teacher_results[teacher_idx][row])
    return selected
