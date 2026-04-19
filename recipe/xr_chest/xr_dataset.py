"""Custom verl ``RLHFDataset`` for XR-chest GRPO.

The parquets produced by ``prepare_xr_data.py`` store paths to RAVE series
directories in ``example["images"]`` rather than embedded image bytes. We
subclass ``RLHFDataset`` with a minimal override that decodes each path
into a PIL image byte-identically to llama-factory's
``Qwen3VLRavePlugin._load_rave_image`` before the stock
``_build_messages`` substitutes them into the chat content.

The ``filter_overlong_prompts`` code path in the base class calls
``process_image`` directly on ``example[image_key]`` entries and does not
go through our override; we therefore require ``filter_overlong_prompts``
to be disabled and rely on offline filtering at parquet-build time (see
``verl/docs/xr_grpo.md``).
"""

from pathlib import Path
from typing import Union

from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset

# Absolute import (not relative): verl loads this file standalone via
# ``importlib.util.spec_from_file_location`` where relative imports don't
# resolve. The repo root is on ``sys.path`` at launch time.
from recipe.xr_chest.rave_decode import load_rave_image


DEFAULT_IMAGE_MAX_PIXELS = 131072  # matches SFT checkpoint (cfg: full_all_unlocked_lowres.yaml)
DEFAULT_IMAGE_MIN_PIXELS = 32 * 32


class XRChestRLHFDataset(RLHFDataset):
    """XR-chest dataset subclass that decodes RAVE series paths on the fly.

    Each decoded PIL image is wrapped in a dict with ``max_pixels`` and
    ``min_pixels`` so that ``qwen_vl_utils.fetch_image`` (called downstream
    by the processor in ``process_vision_info``) matches the image-budget
    of the SFT checkpoint.
    """

    def _build_messages(self, example: dict, key: str):
        images = example.get(self.image_key, None)
        if images:
            max_pixels = int(
                self.config.get("image_max_pixels", DEFAULT_IMAGE_MAX_PIXELS)
            )
            min_pixels = int(
                self.config.get("image_min_pixels", DEFAULT_IMAGE_MIN_PIXELS)
            )
            example = dict(example)
            example[self.image_key] = [
                _ensure_image_dict(img, max_pixels, min_pixels) for img in images
            ]
        return super()._build_messages(example, key)

    def maybe_filter_out_long_prompts(self, dataframe):
        # The base implementation calls `process_image` which doesn't accept
        # raw string paths. We intentionally disallow it; rely on offline
        # filtering at parquet-build time instead.
        if self.config.get("filter_overlong_prompts", False):
            raise NotImplementedError(
                "XRChestRLHFDataset expects data.filter_overlong_prompts=False; "
                "pre-filter at parquet-build time (see verl/docs/xr_grpo.md)."
            )
        return dataframe


def _ensure_image_dict(
    image: Union[str, Path, Image.Image, dict],
    max_pixels: int,
    min_pixels: int,
) -> dict:
    """Normalize to ``{"image": PIL, "max_pixels": ..., "min_pixels": ...}``.

    The base ``_build_messages`` passes dicts straight into
    ``{"type": "image", **image}`` and ``qwen_vl_utils.fetch_image``
    honours the ``max_pixels`` / ``min_pixels`` keys for smart-resize.
    """
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (str, Path)):
        pil = load_rave_image(image)
    elif isinstance(image, dict):
        # Assume caller already set their own pixel budget.
        return image
    else:
        raise TypeError(
            f"Unsupported image type for XRChestRLHFDataset: {type(image)}"
        )
    return {"image": pil, "max_pixels": max_pixels, "min_pixels": min_pixels}
