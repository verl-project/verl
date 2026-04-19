"""Pixel-parity test: our RAVE XR decode vs llama-factory's.

Must pass before any training run — guarantees that GRPO sees pixels
byte-identical to the SFT pipeline that produced the initial checkpoint.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

LF_SRC = Path("/mnt/nfs/home/michael/llama-factory-voio/src")
if str(LF_SRC) not in sys.path:
    sys.path.insert(0, str(LF_SRC))

from llamafactory.data.mm_plugin import Qwen3VLRavePlugin  # noqa: E402

from recipe.xr_chest.rave_decode import load_rave_image  # noqa: E402

PARQUET = Path("/mnt/nfs/home/michael/data/xr_chest_grpo/train_mini.parquet")
N_SAMPLES = 10
SEED = 42


def _collect_paths(parquet_path: Path, n: int, seed: int) -> list[str]:
    df = pd.read_parquet(parquet_path)
    all_paths: list[str] = []
    for row in df["images"]:
        all_paths.extend(list(row))
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:n]


@pytest.mark.parametrize("path", _collect_paths(PARQUET, N_SAMPLES, SEED))
def test_pixel_parity(path: str):
    ours: Image.Image = load_rave_image(path)
    theirs: Image.Image = Qwen3VLRavePlugin._load_rave_image(path)

    assert ours.mode == theirs.mode == "RGB", (ours.mode, theirs.mode)
    assert ours.size == theirs.size, (ours.size, theirs.size)

    ours_arr = np.asarray(ours)
    theirs_arr = np.asarray(theirs)
    assert ours_arr.shape == theirs_arr.shape
    assert ours_arr.dtype == theirs_arr.dtype
    assert np.array_equal(ours_arr, theirs_arr), (
        f"pixel mismatch at {path}: "
        f"max |diff| = {np.abs(ours_arr.astype(int) - theirs_arr.astype(int)).max()}"
    )
