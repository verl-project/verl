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

*  ``PIL.Image`` → ``np.asarray`` (zero-copy, shares PIL buffer)
   → ``torch.from_numpy`` (zero-copy, shares numpy buffer)
   → TQ ``async_put`` which internally uses msgpack ``memoryview`` zero-copy
   serialisation and ZMQ ``send(copy=False)``.

*  On the consumer side the reverse path is equally lean: TQ delivers a
   ``uint8`` tensor whose underlying buffer is turned back into a
   ``PIL.Image`` via ``Image.fromarray`` (one unavoidable copy since PIL
   needs to own its buffer).
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

if TYPE_CHECKING:
    from transfer_queue import AsyncTransferQueueClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default partition name used for multi-modal image storage in TQ.
TQ_MM_IMAGE_PARTITION = "mm_images"

# Module-level TQ client reference (set externally by the trainer or worker init).
_tq_client = None


def set_tq_client(client: "AsyncTransferQueueClient") -> None:
    """Set the module-level TQ client (called during worker/trainer init)."""
    global _tq_client
    _tq_client = client


def get_tq_client() -> "AsyncTransferQueueClient | None":
    """Get the module-level TQ client, or ``None`` if not initialised."""
    return _tq_client


# ---------------------------------------------------------------------------
# Store images into TQ
# ---------------------------------------------------------------------------


async def store_images_to_tq(
    tq_client: "AsyncTransferQueueClient",
    images: list[Image.Image],
    partition_id: str = TQ_MM_IMAGE_PARTITION,
    base_global_index: int = 0,
) -> list[str]:
    """Store a list of PIL images into TransferQueue and return ``tq://`` URLs.

    Each image is converted to a uint8 tensor (zero-copy via numpy) and
    written as a separate sample in a dedicated TQ partition.

    Args:
        tq_client: An initialised ``AsyncTransferQueueClient``.
        images: PIL images to store.
        partition_id: TQ partition that holds the image data.
        base_global_index: Starting global index for the stored samples.
            The caller must ensure uniqueness across concurrent writers.

    Returns:
        A list of ``tq://<partition_id>/<global_index>`` URL strings, one
        per input image, suitable for passing as ``image_data`` items.
    """
    if not images:
        return []

    urls: list[str] = []
    for i, img in enumerate(images):
        global_index = base_global_index + i
        pixel_np = np.asarray(img)  # zero-copy view of PIL buffer
        pixel_tensor = torch.from_numpy(pixel_np.copy())  # copy once: PIL buffer is read-only

        td = TensorDict(
            {"pixel_data": pixel_tensor.unsqueeze(0)},  # [1, H, W, C]
            batch_size=(1,),
        )
        await tq_client.async_put(
            data=td,
            partition_id=partition_id,
        )

        urls.append(make_tq_image_url(partition_id, global_index))

    return urls


def make_tq_image_url(partition_id: str, global_index: int) -> str:
    """Build a ``tq://`` URL for a stored image sample."""
    return f"tq://{partition_id}/{global_index}"


def is_tq_url(url: str) -> bool:
    """Check whether *url* uses the ``tq://`` scheme."""
    return isinstance(url, str) and url.startswith("tq://")


# ---------------------------------------------------------------------------
# Resolve tq:// URLs back to PIL Images (for the RPC path in vLLMHttpServer)
# ---------------------------------------------------------------------------


async def resolve_tq_images(
    tq_client: "AsyncTransferQueueClient",
    urls: list[str],
) -> list[Image.Image]:
    """Fetch images from TransferQueue given a list of ``tq://`` URLs.

    This is the consumer counterpart of :func:`store_images_to_tq`.

    Args:
        tq_client: An initialised ``AsyncTransferQueueClient``.
        urls: ``tq://`` URL strings to resolve.

    Returns:
        A list of PIL images in the same order as *urls*.
    """
    import asyncio

    async def _resolve_one(url: str) -> Image.Image:
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        partition_id, global_index = _parse_tq_url(url)
        metadata = await tq_client.async_kv_retrieve_meta(
            keys=[str(global_index)],
            partition_id=partition_id,
            create=False,
        )
        td = await tq_client.async_get_data(metadata)
        pixel_tensor = td["pixel_data"][0]
        pixel_np = pixel_tensor.numpy()
        channels = pixel_np.shape[2] if pixel_np.ndim == 3 else 1
        mode = "RGB" if channels == 3 else ("RGBA" if channels == 4 else "L")
        return Image.fromarray(pixel_np, mode=mode)

    return list(await asyncio.gather(*[_resolve_one(u) for u in urls]))


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
    try:
        tq_enabled = os.environ.get("TRANSFER_QUEUE_ENABLE", False)
        if not tq_enabled:
            return images, videos

        tq_client = get_tq_client()
        if tq_client is None:
            return images, videos
    except ImportError:
        return images, videos

    image_data = images
    if images:
        base_idx = abs(hash(request_id + "_img")) % (2**31)
        image_data = await store_images_to_tq(tq_client, images, base_global_index=base_idx)

    video_data = videos
    # Video TQ storage can be added here following the same pattern.

    return image_data, video_data


# ---------------------------------------------------------------------------
# TQ connection info serialisation helpers (for env-var injection)
# ---------------------------------------------------------------------------


def serialize_tq_info(obj: Any) -> str:
    """Pickle *obj* and return a base64-encoded string (safe for env vars)."""
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def deserialize_tq_info(b64_str: str) -> Any:
    """Reverse of :func:`serialize_tq_info`."""
    return pickle.loads(base64.b64decode(b64_str))
