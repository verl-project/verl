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

"""Temporary optimization patch for exact segmented TorchStore tensor reads.

The pinned LocalClient assembles whole tensors/rectangles. Remove this shim when
its native batch API preserves sparse segments, aliases, and failure cleanup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchstore.client import LocalClient
    from torchstore.transport.types import Request

    from verl.single_controller.monarch.object_store.store import TensorRowRange


async def fetch_torchstore_tensor_rows(
    client: LocalClient, requests: list[Request], row_ranges: dict[str, TensorRowRange]
) -> dict[str, Any]:
    """Fetch exact row segments together and assemble one envelope per key.

    Native transports retain every segment, but the pinned client assumes
    whole tensors or fully covered rectangles. Assemble only segmented keys
    here, using every source-slice axis; ordinary reads retain native assembly.
    """
    from torchstore.transport import create_transport_buffer
    from torchstore.utils import get_target_tensor_shape_and_offset

    if not requests:
        return {}
    unique_requests = {request.key: request for request in requests}
    multiple = {key: rows for key, rows in row_ranges.items() if key in unique_requests and len(rows.segments) > 1}
    locations = await client._locate_volumes(list(unique_requests))
    buffers = {
        volume: create_transport_buffer(client.strategy.get_storage_volume(volume))
        for volume in {volume for by_volume in locations.values() for volume in by_volume}
    }
    by_volume, whole = client._build_volume_requests(requests, locations, buffers)

    async def fetch_volume(volume: str, children: list[Request]) -> list[tuple[Request, Any]]:
        values = await buffers[volume].get_from_storage_volume(children)
        return list(zip(children, values, strict=True))

    tasks = [asyncio.create_task(fetch_volume(volume, children)) for volume, children in by_volume.items()]
    try:
        batches = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    pairs = [pair for batch in batches for pair in batch]
    # OBJECT decoding ignores row selections, including objects reducing to a
    # tensor. Keep those keys on the native whole-object branch.
    object_keys = {request.key for request, _value in pairs if request.tensor_slice is None}
    multiple = {key: rows for key, rows in multiple.items() if key not in object_keys}
    grouped = {key: [] for key in multiple}
    for request, value in pairs:
        if request.key in grouped:
            grouped[request.key].append((request, value))
    for key, rows in multiple.items():
        matched = 0
        for start, stop in rows.segments:
            parts = [
                (request, value)
                for request, value in grouped[key]
                if start <= request.tensor_slice.offsets[0]
                and request.tensor_slice.offsets[0] + request.tensor_slice.local_shape[0] <= stop
            ]
            matched += len(parts)
            message = f"Incomplete tensor rows for key {key!r}: [{start}, {stop})"
            if not parts or any(
                tuple(value.shape) != tuple(request.tensor_slice.local_shape) for request, value in parts
            ):
                raise RuntimeError(message)
            origin = (start,) + (0,) * (len(rows.shape) - 1)
            expected = (stop - start, *rows.shape[1:])
            if any(
                offset < lower or offset + size > lower + extent
                for request, _value in parts
                for offset, size, lower, extent in zip(
                    request.tensor_slice.offsets, request.tensor_slice.local_shape, origin, expected, strict=True
                )
            ):
                raise RuntimeError(message)
            try:
                shape, offset = get_target_tensor_shape_and_offset(
                    [value.shape for _request, value in parts],
                    [request.tensor_slice.offsets for request, _value in parts],
                )
            except AssertionError as error:
                raise RuntimeError(message) from error
            if tuple(shape) != expected or tuple(offset) != origin:
                raise RuntimeError(message)
        if matched != len(grouped[key]):
            raise RuntimeError(f"Tensor parts exceed requested row segments for key {key!r}")
    result = client._assemble_results(
        [request for key, request in unique_requests.items() if key not in multiple],
        [(request, value) for request, value in pairs if request.key not in multiple],
        whole,
    )
    for request, value in pairs:
        rows = multiple.get(request.key)
        if rows is None:
            continue
        if request.key not in result:
            result[request.key] = value.new_zeros((rows.stop - rows.start, *rows.shape[1:]))
        source = request.tensor_slice
        assert source is not None
        origins = (rows.start,) + (0,) * (len(rows.shape) - 1)
        indices = tuple(
            slice(offset - origin, offset - origin + size)
            for offset, origin, size in zip(source.offsets, origins, source.local_shape, strict=True)
        )
        result[request.key][indices].copy_(value)
    return result
