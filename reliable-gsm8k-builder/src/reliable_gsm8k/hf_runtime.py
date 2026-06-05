from __future__ import annotations

import importlib.util
import os


def disable_hf_transfer_if_unavailable() -> bool:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        return False
    if importlib.util.find_spec("hf_transfer") is not None:
        return False
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return True
