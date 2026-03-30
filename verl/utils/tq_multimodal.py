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
"""Utilities for storing / retrieving multi-modal image data via TransferQueue.

The key design goal is **minimal copy**:

*  Store path: ``PIL.Image`` → ``np.asarray`` (zero-copy view of PIL buffer)
   → pre-allocated contiguous buffer ``copy_`` (single memcpy, no intermediate
   allocation) → TQ ``async_kv_put`` with msgpack ``memoryview`` zero-copy
   serialisation and ZMQ ``send(copy=False)``.

*  Resolve path: TQ delivers a flat ``uint8`` tensor → slice + reshape to
   ``(C, H, W)`` numpy arrays (channels-first, the format vLLM expects for
   ``multi_modal_data["image"]``).  **No PIL round-trip** — the numpy arrays
   are passed directly to the vLLM engine, avoiding two unnecessary copies
   (``Image.fromarray`` + HF processor's internal ``np→tensor``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from transfer_queue import AsyncTransferQueueClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default partition name used for multi-modal image storage in TQ.
TQ_MM_IMAGE_PARTITION = "mm_images"

# Module-level TQ client reference (set externally by the trainer or worker init).
_tq_client = None


def set_tq_client(client: AsyncTransferQueueClient) -> None:
    """Set the module-level TQ client (called during worker/trainer init)."""
    global _tq_client
    _tq_client = client


def get_tq_client() -> AsyncTransferQueueClient | None:
    """Get the module-level TQ client, or ``None`` if not initialised."""
    return _tq_client


# ---------------------------------------------------------------------------
# tq:// URL helpers
# ---------------------------------------------------------------------------


def make_tq_image_url(partition_id: str, batch_key: str, index: int) -> str:
    """Build a ``tq://`` URL for a stored image sample.

    Format: ``tq://<partition_id>/<batch_key>/<index>``
    """
    return f"tq://{partition_id}/{batch_key}/{index}"


def parse_tq_url(url: str) -> tuple[str, str, int]:
    """Parse a ``tq://`` URL into ``(partition_id, batch_key, index)``.

    Raises:
        ValueError: If the URL is not a valid ``tq://`` URL.
    """
    if not url.startswith("tq://"):
        raise ValueError(f"Not a tq:// URL: {url}")
    parts = url[len("tq://") :].split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid tq:// URL format (expected 3 path segments): {url}")
    return parts[0], parts[1], int(parts[2])


def is_tq_url(url: str) -> bool:
    """Check whether *url* uses the ``tq://`` scheme."""
    return isinstance(url, str) and url.startswith("tq://")


# ---------------------------------------------------------------------------
# Store images into TQ (batched, single round-trip per call)
# ---------------------------------------------------------------------------


async def store_images_to_tq(
    tq_client: AsyncTransferQueueClient,
    images: list,
    partition_id: str = TQ_MM_IMAGE_PARTITION,
) -> list[str]:
    """Store a list of PIL images into TransferQueue and return ``tq://`` URLs.

    All images from a single call are batched into **one** TQ entry: pixel data
    is flattened and written into a single pre-allocated contiguous buffer,
    alongside shape and offset metadata.  This avoids N separate TQ
    round-trips and eliminates intermediate tensor allocations.

    The entry is stored under an explicit KV key (``uuid4().hex``) so that
    consumers can retrieve it deterministically.

    Args:
        tq_client: An initialised ``AsyncTransferQueueClient``.
        images: PIL images to store.
        partition_id: TQ partition that holds the image data.

    Returns:
        A list of ``tq://<partition_id>/<batch_key>/<index>`` URL strings,
        one per input image, suitable for passing as ``image_data`` items.
    """
    if not images:
        return []

    batch_key = uuid4().hex

    # First pass: compute shapes, offsets, and total buffer size.
    np_views: list[np.ndarray] = []
    shapes: list[list[int]] = []
    offsets: list[int] = []
    total_elements = 0

    for img in images:
        pixel_np = np.asarray(img)  # zero-copy view of PIL buffer
        if pixel_np.ndim == 2:
            pixel_np = pixel_np[:, :, np.newaxis]  # (H, W) -> (H, W, 1) for grayscale
        np_views.append(pixel_np)
        shapes.append(list(pixel_np.shape))  # [H, W, C]
        offsets.append(total_elements)
        total_elements += pixel_np.size

    # Second pass: pre-allocate one contiguous buffer and copy directly into it.
    pixel_flat = torch.empty(total_elements, dtype=torch.uint8)
    for np_view, offset in zip(np_views, offsets, strict=False):
        n = np_view.size
        # Direct copy from PIL's numpy view into the pre-allocated buffer.
        # This is the single unavoidable memcpy (PIL buffer is read-only).
        pixel_flat[offset : offset + n].copy_(torch.from_numpy(np_view.reshape(-1)))

    shapes_tensor = torch.tensor(shapes, dtype=torch.int64)  # [N, 3]
    offsets_tensor = torch.tensor(offsets, dtype=torch.int64)  # [N]

    td = TensorDict(
        {
            "pixel_flat": pixel_flat.unsqueeze(0),  # [1, total_elements]
            "shapes": shapes_tensor.unsqueeze(0),  # [1, N, 3]
            "offsets": offsets_tensor.unsqueeze(0),  # [1, N]
        },
        batch_size=(1,),
    )
    await tq_client.async_kv_put(
        data=td,
        keys=[batch_key],
        partition_id=partition_id,
    )

    return [make_tq_image_url(partition_id, batch_key, i) for i in range(len(images))]


# ---------------------------------------------------------------------------
# Resolve tq:// URLs back to numpy arrays (channels-first for vLLM)
# ---------------------------------------------------------------------------


async def resolve_tq_images(
    tq_client: AsyncTransferQueueClient,
    urls: list[str],
) -> list[np.ndarray]:
    """Fetch images from TransferQueue given a list of ``tq://`` URLs.

    URLs sharing the same batch key are fetched in a single TQ call, so a
    batch of N images stored by :func:`store_images_to_tq` costs only one
    round-trip regardless of N.

    Returns numpy arrays in **channels-first** ``(C, H, W)`` format, which
    vLLM's ``multi_modal_data["image"]`` accepts directly — no PIL
    round-trip needed.

    Args:
        tq_client: An initialised ``AsyncTransferQueueClient``.
        urls: ``tq://`` URL strings to resolve.

    Returns:
        A list of ``numpy.ndarray`` in ``(C, H, W)`` uint8 format, in the
        same order as *urls*.
    """
    # Group by (partition, batch_key) to fetch each batch only once.
    batch_groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for url_idx, url in enumerate(urls):
        partition_id, batch_key, img_index = parse_tq_url(url)
        batch_groups[(partition_id, batch_key)].append((url_idx, img_index))

    results: list[np.ndarray | None] = [None] * len(urls)

    async def _fetch_batch(
        partition_id: str,
        batch_key: str,
        indices: list[tuple[int, int]],
    ) -> None:
        metadata = await tq_client.async_kv_retrieve_meta(
            keys=[batch_key],
            partition_id=partition_id,
            create=False,
        )
        td = await tq_client.async_get_data(metadata)

        pixel_flat = td["pixel_flat"][0]  # [total_elements]
        shapes = td["shapes"][0]  # [N, 3]
        offsets_t = td["offsets"][0]  # [N]

        for url_idx, img_index in indices:
            h, w, c = (int(v) for v in shapes[img_index].tolist())
            offset = int(offsets_t[img_index].item())
            num_elements = h * w * c
            # Slice → reshape to (H, W, C) → transpose to (C, H, W) channels-first.
            # vLLM expects (C, H, W) for numpy/tensor image inputs and uses
            # ``_, h, w = image.shape`` to detect image size.
            pixel_hwc = pixel_flat[offset : offset + num_elements].reshape(h, w, c)
            pixel_chw = pixel_hwc.numpy().transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
            results[url_idx] = pixel_chw

    await asyncio.gather(*[_fetch_batch(pid, bkey, idxs) for (pid, bkey), idxs in batch_groups.items()])

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Convenience wrapper for agent loops
# ---------------------------------------------------------------------------


async def maybe_store_media_to_tq(
    images: list | None,
    videos: list | None,
    request_id: str,
) -> tuple[list | None, list | None]:
    """Store images/videos into TransferQueue when TQ is enabled.

    If TQ is not available or not enabled, return the original data unchanged
    so that the existing PIL-Image-over-Ray-RPC path remains functional.

    Returns:
        ``(image_data, video_data)`` — either ``tq://`` URL lists or the
        original PIL Image lists.
    """
    tq_enabled = os.environ.get("TRANSFER_QUEUE_ENABLE", False)
    if not tq_enabled:
        return images, videos

    tq_client = get_tq_client()
    if tq_client is None:
        return images, videos

    image_data = images
    if images:
        image_data = await store_images_to_tq(tq_client, images)

    video_data = videos
    # Video TQ storage can be added here following the same pattern.

    return image_data, video_data


# ---------------------------------------------------------------------------
# TQ connection info serialisation helpers (for env-var injection)
# ---------------------------------------------------------------------------


def serialize_tq_info(obj: Any) -> str:
    """JSON-serialize *obj* and return a base64-encoded string (safe for env vars).

    Uses JSON instead of pickle for security — avoids arbitrary code execution
    risk when the env var crosses process boundaries.
    """
    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")


def deserialize_tq_info(b64_str: str) -> Any:
    """Reverse of :func:`serialize_tq_info`."""
    return json.loads(base64.b64decode(b64_str))
