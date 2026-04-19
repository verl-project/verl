"""Convert the llama-factory XR-chest study-grouped JSONL into verl-ready parquet.

Replicates the prompt construction of ``VoioSegmedQwen3VLConverter`` from
``llama-factory-voio/src/llamafactory/data/converter.py:972-1036`` with two
deliberate deviations:

1. The task phrase is fixed to ``"Write a radiology report for this study."``
   instead of sampled each call from a 10-phrase bag. GRPO groups must see
   the same prompt across rollouts.
2. Prompt metadata stripping is hard-coded ON (matches the XR SFT configs
   which set ``strip_prompt_metadata: true``).

Output columns:
    - data_source: str  (always "xr_chest", used as reward_fn_key)
    - prompt: list[{role, content}]  (content has <image> placeholders)
    - images: list[str]  (abs paths to RAVE series dirs)
    - ability: str  ("report_gen")
    - reward_model: {"style": "rule", "ground_truth": <report>}
    - extra_info: {study_id, index, split}
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

TASK_PHRASE = "Write a radiology report for this study."
DATA_SOURCE = "xr_chest"
MODALITY = "xr_chest"
MAX_IMAGES = 3
MEDIA_DIR_DEFAULT = "/data/cache/voio-segmed/rave/xr_chest"

LF_DATA_DIR_DEFAULT = Path("/mnt/nfs/home/michael/llama-factory-voio/data")
SOURCE_JSONLS = {
    "train": "xr_chest-v0_1_3-v4-train.jsonl",
    "valid_mini": "xr_chest-v0_1_3-v4-valid_mini.jsonl",
    "test": "xr_chest-v0_1_3-v4-test.jsonl",
}


def _series_to_path(series_id: str, media_dir: str) -> str:
    """Resolve a RAVE series_id to its on-disk directory.

    Matches ``VoioSegmedQwen3VLConverter``: XR series_ids end in ``_XX...``
    and already point directly at the cache dir (no ``.1.0`` suffix is
    added). CT/MR would take the ``.1.0`` fallback but we never hit it here.
    """
    direct_path = os.path.join(media_dir, series_id)
    if series_id.endswith(".1.0") or os.path.isdir(direct_path):
        return direct_path
    # Fallback for legacy CT-style ids. Not expected for XR but keeps parity.
    return os.path.join(media_dir, f"{series_id}.1.0")


def _build_user_content(series: list[dict]) -> str:
    """Exact text produced by ``VoioSegmedQwen3VLConverter`` with
    ``strip_prompt_metadata=True`` and the fixed task phrase."""
    content = "The following series are obtained from the study:\n\n"
    for s in series:
        content += f"Series {s['series_number']}: <image> \n\n"
    content += "\n\n" + TASK_PHRASE
    return content


def _select_and_order_series(series_list: list[dict], max_imgs: int) -> list[dict]:
    """Replicate the converter's slice-count-descending sort + head cut."""
    if len(series_list) > max_imgs:
        series_list = sorted(
            series_list,
            key=lambda s: (
                s.get("slice_count") if isinstance(s.get("slice_count"), (int, float)) else 0
            ),
            reverse=True,
        )
    return series_list[:max_imgs]


def convert_split(
    src_jsonl: Path,
    out_parquet: Path,
    split_name: str,
    media_dir: str,
    max_images: int,
    verify_paths: bool,
    limit: int | None,
) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    dropped_no_series = 0
    dropped_missing_paths = 0

    with open(src_jsonl) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            series_list = raw.get("series", [])
            if not series_list:
                dropped_no_series += 1
                continue
            series_list = _select_and_order_series(series_list, max_images)

            image_paths = [
                _series_to_path(s["series_id"], media_dir) for s in series_list
            ]

            if verify_paths:
                missing = [p for p in image_paths if not os.path.isdir(p)]
                if missing:
                    dropped_missing_paths += 1
                    continue

            user_content = _build_user_content(series_list)

            rows.append({
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": user_content}],
                "images": image_paths,
                "ability": "report_gen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": raw.get("report", ""),
                },
                "extra_info": {
                    "study_id": raw["study_id"],
                    "index": idx,
                    "split": split_name,
                },
            })

            if limit is not None and len(rows) >= limit:
                break

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(
        f"[{split_name}] wrote {len(df)} rows to {out_parquet} "
        f"(dropped: no-series={dropped_no_series}, missing-paths={dropped_missing_paths})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lf-data-dir",
        type=Path,
        default=LF_DATA_DIR_DEFAULT,
        help="Directory holding the llama-factory xr_chest-v0_1_3-v4-*.jsonl files.",
    )
    ap.add_argument(
        "--media-dir",
        default=MEDIA_DIR_DEFAULT,
        help="RAVE cache root holding <series_id>/volume.mp4 subdirs.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "data" / "xr_chest_grpo",
        help="Output directory for {train,valid_mini,train_mini}.parquet.",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
    )
    ap.add_argument(
        "--verify-paths",
        action="store_true",
        help=(
            "If set, stat each series dir and drop rows with missing paths. "
            "Only safe on a node with the RAVE cache (e.g. b200-9)."
        ),
    )
    ap.add_argument(
        "--train-mini-size",
        type=int,
        default=32,
        help="Number of rows to carve from train into train_mini.parquet for smoke runs.",
    )
    args = ap.parse_args()

    # Main splits
    for split_name, jsonl_name in SOURCE_JSONLS.items():
        src = args.lf_data_dir / jsonl_name
        if not src.exists():
            print(f"[skip] {split_name}: {src} not found")
            continue
        out = args.out_dir / f"{split_name}.parquet"
        convert_split(
            src_jsonl=src,
            out_parquet=out,
            split_name=split_name,
            media_dir=args.media_dir,
            max_images=args.max_images,
            verify_paths=args.verify_paths,
            limit=None,
        )

    # Smoke-run train_mini carved from train
    train_src = args.lf_data_dir / SOURCE_JSONLS["train"]
    if train_src.exists():
        out_mini = args.out_dir / "train_mini.parquet"
        convert_split(
            src_jsonl=train_src,
            out_parquet=out_mini,
            split_name="train_mini",
            media_dir=args.media_dir,
            max_images=args.max_images,
            verify_paths=args.verify_paths,
            limit=args.train_mini_size,
        )


if __name__ == "__main__":
    main()
