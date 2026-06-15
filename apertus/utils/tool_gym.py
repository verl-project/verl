"""Convert Tool Gym verl.jsonl files to train/val parquet files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import datasets


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_toolgym_parquets(
    source_dir: str | Path,
    *,
    val_per_task: int,
    seed: int,
) -> None:
    source_dir = Path(source_dir).expanduser()
    output_dir = source_dir / "dataset"
    files = sorted(source_dir.glob("*/verl.jsonl"))
    if not files:
        raise FileNotFoundError(f"No */verl.jsonl files found under {source_dir}")

    train_rows, val_rows = [], []
    for path in files:
        task_name = path.parent.name
        rows = read_jsonl(path)
        if len(rows) <= val_per_task:
            raise ValueError(
                f"{task_name} has {len(rows)} rows, fewer than "
                f"{val_per_task + 1} required for a train/val split."
            )

        for row in rows:
            row["data_source"] = f"tool_gym:{task_name}"

        random.Random(f"{seed}:{task_name}").shuffle(rows)
        val_rows.extend(rows[:val_per_task])
        train_rows.extend(rows[val_per_task:])
        print(
            f"{task_name}: {len(rows) - val_per_task} train, {val_per_task} val",
            flush=True,
        )

    random.Random(f"{seed}:train").shuffle(train_rows)
    random.Random(f"{seed}:val").shuffle(val_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    datasets.Dataset.from_list(train_rows).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_rows).to_parquet(str(val_path))
    print(
        f"Wrote {len(train_rows)} train rows to {train_path} and "
        f"{len(val_rows)} val rows to {val_path}.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Tool Gym verl.jsonl files to train/val parquet files."
    )
    parser.add_argument(
        "--source-dir",
        default="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/toolgym_test_v3",
        help="Directory containing per-subset */verl.jsonl files.",
    )
    parser.add_argument(
        "--val-per-task",
        type=int,
        default=50,
        help="Number of validation rows reserved from each discovered task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_toolgym_parquets(
        source_dir=args.source_dir,
        val_per_task=args.val_per_task,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
