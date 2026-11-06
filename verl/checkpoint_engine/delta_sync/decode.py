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
"""Device- and backend-agnostic decoding of DeltaFlush payloads.

This is the consumer-free half of the wire decode: it validates one received
flush (spec JSON, encoding consistency, per-parameter slice bounds, checksum)
and returns plain per-parameter patches. Backend-specific consumers then
convert the patches into whatever their loader expects (the vLLM CUDA path
turns them into ``CheckpointWeightPatch`` via vLLM's checkpoint patch API;
the NPU HCCL receiver and tests apply them through ``index_copy_`` /
``model.load_weights``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .encode import checksum


@dataclass
class DeltaPatch:
    """One decoded parameter patch in final checkpoint/HF coordinates.

    ``positions`` is an int32 flat-index tensor into the flattened parameter
    (``None`` for dense/full-coverage flushes); ``values`` holds the
    corresponding elements in the parameter's dtype.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    values: torch.Tensor
    positions: torch.Tensor | None  # int32 flat positions; None for dense


def decode_delta_wire(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> tuple[str, list[DeltaPatch]]:
    """Validate and decode one DeltaFlush into ``(encoding, [DeltaPatch])``.

    Raises ``ValueError`` on malformed payloads and ``RuntimeError`` on a
    checksum mismatch (wire corruption).
    """
    tensors = dict(named_tensors)
    try:
        spec_tensor = tensors["__delta_spec__"]
        values = tensors["__values__"]
        spec = json.loads(bytes(spec_tensor.detach().cpu().numpy().tobytes()).decode())
    except KeyError as exc:
        raise ValueError(f"DeltaFlush is missing {exc.args[0]}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("DeltaFlush contains an invalid JSON spec") from exc

    encoding = spec["encoding"]
    if encoding not in {"dense", "indices"}:
        raise ValueError(f"Unsupported DeltaFlush encoding: {encoding!r}")
    positions = tensors.get("__positions__")
    if positions is None:
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)
    if encoding == "dense" and positions.numel():
        raise ValueError("Dense DeltaFlush must not contain positions")
    if encoding == "indices" and "__positions__" not in tensors:
        raise ValueError("Indices DeltaFlush requires __positions__")

    got_checksum = checksum(positions, values)
    expected_checksum = int(spec["checksum"])
    if got_checksum != expected_checksum:
        raise RuntimeError(f"Delta checksum mismatch: got {got_checksum}, expected {expected_checksum}")

    patches: list[DeltaPatch] = []
    for param_spec in spec["params"]:
        name = param_spec["name"]
        dtype = getattr(torch, param_spec["dtype"])
        patch_values = values[param_spec["val_start"] : param_spec["val_end"]]
        if encoding == "dense":
            patch_positions = None
        else:
            if param_spec["pos_width"] != 4:
                raise ValueError(f"{name}: indices encoding requires pos_width=4")
            pos_bytes = positions[param_spec["pos_start"] : param_spec["pos_end"]]
            if pos_bytes.numel() != patch_values.numel() * 4:
                raise ValueError(f"{name}: position and value slices have inconsistent lengths")
            patch_positions = pos_bytes.view(torch.int32)

        patches.append(
            DeltaPatch(
                name=name,
                shape=tuple(param_spec["shape"]),
                dtype=dtype,
                values=patch_values,
                positions=patch_positions,
            )
        )
    return encoding, patches
