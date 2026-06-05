# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Load a dataset, add question_id=md5(question), assert global uniqueness,
and save train/test parquet files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from typing import Dict, Iterable, List, Tuple

import datasets


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def _parse_splits(raw: str) -> List[str]:
    splits = [s.strip() for s in raw.split(",") if s.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split name.")
    return splits


def _default_output_dir(dataset_name: str, config_name: str | None) -> str:
    dataset_tag = _sanitize_name(dataset_name)
    config_tag = _sanitize_name(config_name) if config_name else "default"
    return os.path.expanduser(f"~/data/primitive/{dataset_tag}_{config_tag}_with_question_id")


def _load_dataset_dict(local_dataset_path: str | None, dataset_name: str, config_name: str | None):
    if local_dataset_path:
        expanded = os.path.expanduser(local_dataset_path)
        if os.path.exists(expanded):
            try:
                ds = datasets.load_from_disk(expanded)
            except Exception:
                ds = datasets.load_dataset(expanded, config_name)
        else:
            ds = datasets.load_dataset(local_dataset_path, config_name)
    else:
        ds = datasets.load_dataset(dataset_name, config_name)

    if isinstance(ds, datasets.DatasetDict):
        return ds
    if isinstance(ds, datasets.Dataset):
        return datasets.DatasetDict({"train": ds})
    raise TypeError(f"Unsupported dataset object type: {type(ds)}")


def _require_split(ds_dict: datasets.DatasetDict, split: str) -> datasets.Dataset:
    if split not in ds_dict:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(ds_dict.keys())}")
    return ds_dict[split]


def _add_question_id(ds: datasets.Dataset, question_field: str) -> datasets.Dataset:
    if question_field not in ds.column_names:
        raise KeyError(f"Required field '{question_field}' not found. Columns: {ds.column_names}")

    def process_fn(example):
        question = example[question_field]
        if not isinstance(question, str) or not question:
            raise ValueError(f"Invalid question value for field '{question_field}': {question!r}")
        qid = hashlib.md5(question.encode("utf-8")).hexdigest()
        example["question_id"] = qid
        return example

    return ds.map(process_fn)


def _find_duplicate_ids(ds_by_split: Dict[str, datasets.Dataset]) -> Dict[str, List[Tuple[str, int]]]:
    seen: Dict[str, Tuple[str, int]] = {}
    duplicates: Dict[str, List[Tuple[str, int]]] = {}
    for split, ds in ds_by_split.items():
        for idx, qid in enumerate(ds["question_id"]):
            if qid in seen:
                if qid not in duplicates:
                    duplicates[qid] = [seen[qid]]
                duplicates[qid].append((split, idx))
            else:
                seen[qid] = (split, idx)
    return duplicates


def _format_duplicate_report(dupes: Dict[str, List[Tuple[str, int]]], limit: int = 10) -> str:
    lines: List[str] = []
    for i, (qid, rows) in enumerate(dupes.items()):
        if i >= limit:
            lines.append(f"... and {len(dupes) - limit} more duplicated question_id values")
            break
        where = ", ".join(f"{split}[{idx}]" for split, idx in rows[:6])
        suffix = " ..." if len(rows) > 6 else ""
        lines.append(f"{qid}: {where}{suffix}")
    return "\n".join(lines)


def _save_parquet(ds_by_split: Dict[str, datasets.Dataset], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split, ds in ds_by_split.items():
        out_path = os.path.join(output_dir, f"{split}.parquet")
        ds.to_parquet(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add question_id=md5(question) to dataset rows and save parquet splits."
    )
    parser.add_argument("--dataset", default="openai/gsm8k", help="Hugging Face dataset name.")
    parser.add_argument("--config", default="main", help="Hugging Face dataset config.")
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to process and save (default: train,test).",
    )
    parser.add_argument(
        "--question_field",
        default="question",
        help="Strict top-level field to hash into question_id.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for parquet files. If omitted, a clear default name is used.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local HF dataset path (load_from_disk or load_dataset path).",
    )
    args = parser.parse_args()

    split_names = _parse_splits(args.splits)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else _default_output_dir(args.dataset, args.config)

    ds_dict = _load_dataset_dict(
        local_dataset_path=args.local_dataset_path,
        dataset_name=args.dataset,
        config_name=args.config,
    )

    processed: Dict[str, datasets.Dataset] = {}
    total_rows = 0
    for split in split_names:
        ds = _require_split(ds_dict, split)
        ds = _add_question_id(ds, args.question_field)
        processed[split] = ds
        total_rows += len(ds)

    duplicates = _find_duplicate_ids(processed)
    if duplicates:
        report = _format_duplicate_report(duplicates)
        raise AssertionError(
            f"question_id uniqueness assertion failed: {len(duplicates)} duplicated values found.\n{report}"
        )

    _save_parquet(processed, output_dir)

    print("Saved dataset with question_id column.")
    print(f"Dataset: {args.dataset} (config={args.config})")
    print(f"Splits: {split_names}")
    print(f"Rows: {total_rows}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
