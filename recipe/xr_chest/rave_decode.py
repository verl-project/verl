"""RAVE XR decode helper.

Byte-identical to ``Qwen3VLRavePlugin._load_rave_image`` at
``llama-factory-voio/src/llamafactory/data/mm_plugin.py:1763-1776``.

Factored into its own module so the dataset, the preprocessing script, and
the pixel-parity test can all share one source of truth.
"""

import os
from pathlib import Path
from typing import Union

import av
import numpy as np
from PIL import Image


def load_rave_image(path: Union[str, Path]) -> Image.Image:
    """Decode one frame of RAVE XR ``volume.mp4`` into an RGB PIL image.

    Pipeline (matches llama-factory's ``Qwen3VLRavePlugin``):
        HEVC decode -> gray16le uint16 ndarray
        -> float32 / 65535 * 255
        -> clip(0, 255) -> uint8
        -> ``Image.fromarray(..., mode="L").convert("RGB")``
    """
    mp4_path = os.path.join(str(path), "volume.mp4")
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frame = next(container.decode(stream))
    arr = frame.to_ndarray(format="gray16le")
    container.close()

    arr_f = arr.astype(np.float32) / 65535.0
    arr_u8 = (arr_f * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L").convert("RGB")
