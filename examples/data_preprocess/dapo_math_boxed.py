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
"""Prepare pinned filtered & deduplicated DAPO-Math boxed training data.

Upstream lineage:
1. ``aaabiao/dapo_filter`` (used in SimpleTIR, arXiv:2509.02479) filters
   ``BytedTsinghua-SIA/DAPO-Math-17k`` by initial-model solve rate to discard overly
   easy or overly hard tasks, formats prompts with the ``\\boxed{}`` suffix, and stores
   13,044 rows (12,691 unique prompts plus 353 duplicate-prompt entries).
2. ``Jiawei415/DPAO_filter`` (linked in ``docs/algo/otb.md``, arXiv:2602.07078) applies
   first-occurrence ``prompt`` deduplication to ``aaabiao/dapo_filter``, yielding the
   exact same 12,691 rows in the same order (``train/dapo_filter.parquet``).

Both upstream files store ``data_source="dapo_filter"``, which raises
``NotImplementedError`` in ``verl.utils.reward_score.default_compute_score``. This
standalone script verifies pinned source bytes/SHA-256, deduplicates by ``prompt`` when
loading ``aaabiao/dapo_filter`` so both sources produce the exact same 12,691 rows,
rewrites only ``data_source`` to ``"math_dapo"`` while preserving prompts, answers,
UUID indices, row order, and metadata verbatim, verifies reward dispatch on every row,
and writes ``train.parquet`` plus a provenance ``manifest.json``.

Validation data is intentionally kept separate from training preparation (for example,
use dedicated AIME validation preprocessing rather than ``Jiawei415/DPAO_filter``'s
``test/`` files, where ``test/aime24.parquet`` and ``test/aime25.parquet`` are
byte-identical).

    python examples/data_preprocess/dapo_math_boxed.py \
        --output-dir ~/data/dapo-math-boxed
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score import default_compute_score

BOXED_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."
EXPECTED_UNIQUE_ROWS = 12691
RAW_DATA_SOURCE = "dapo_filter"
TARGET_DATA_SOURCE = "math_dapo"
DEFAULT_SOURCE = "Jiawei415/DPAO_filter"

DAPO_FILTER_SOURCES = {
    "Jiawei415/DPAO_filter": {
        "repo_id": "Jiawei415/DPAO_filter",
        "revision": "d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd",
        "filename": "train/dapo_filter.parquet",
        "sha256": "47a9806fdb10be56336651384f0c1a7d2c7de6be2883b1b64a7d559bf6f9e39c",
        "bytes": 2428814,
        "raw_rows": 12691,
        "deduplicate_prompts": False,
        "license": None,
        "license_status": "Not declared in the pinned Hugging Face dataset metadata; no usage rights inferred",
        "repo_url": (
            "https://huggingface.co/datasets/Jiawei415/DPAO_filter/tree/d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd"
        ),
    },
    "aaabiao/dapo_filter": {
        "repo_id": "aaabiao/dapo_filter",
        "revision": "0b10fea35812902fe4f6985f2064424ec460f9c0",
        "filename": "train.parquet",
        "sha256": "78b9c05dfa36982ac4b44c76c9a05335979c1d37371425271fa7f29fd2e7e436",
        "bytes": 2395267,
        "raw_rows": 13044,
        "deduplicate_prompts": True,
        "license": None,
        "license_status": "Not declared in the pinned Hugging Face dataset metadata; no usage rights inferred",
        "repo_url": (
            "https://huggingface.co/datasets/aaabiao/dapo_filter/tree/0b10fea35812902fe4f6985f2064424ec460f9c0"
        ),
    },
}


def digest(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    """Serialize nested row values deterministically for semantic hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_dedupe_key(prompt) -> str:
    """Compute deterministic JSON key matching Jiawei415/DPAO_filter dedupe_downsample.py."""
    return json.dumps(prompt, ensure_ascii=True, sort_keys=True)


def dedupe_rows_by_prompt(raw_rows: list[dict]) -> list[dict]:
    """Keep the first occurrence of each normalized prompt while preserving row order."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in raw_rows:
        key = prompt_dedupe_key(row.get("prompt"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def fetch_source(source: dict, local_dir: Path) -> Path:
    """Download a pinned Hugging Face dataset file and verify its byte size and SHA-256."""
    path = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            filename=source["filename"],
            local_dir=local_dir,
        )
    )
    if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
        raise ValueError(f"Source size/hash mismatch: {source['repo_id']}@{source['revision']}:{source['filename']}")
    return path


def iter_parquet(path: Path):
    """Yield Python row dicts from a Parquet file without materializing all batches at once."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def verify_reward(rows: list[dict]) -> None:
    """Ensure every prepared row routes to math_dapo and scores boxed answers as expected."""
    for row in rows:
        answer = row["reward_model"]["ground_truth"]
        correct = default_compute_score(row["data_source"], f"\\boxed{{{answer}}}", answer)
        wrong = default_compute_score(row["data_source"], r"\boxed{INCORRECT-SENTINEL}", answer)
        if correct["acc"] is not True or correct["score"] != 1.0 or wrong["acc"] is not False or wrong["score"] != -1.0:
            raise ValueError(f"Unexpected reward dispatch or answer normalization for index={row['extra_info']}")


def write_parquet(rows: list[dict], path: Path) -> dict:
    """Write rows to Parquet, verify lossless round-trip, and return artifact provenance."""
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    if pq.read_table(path).to_pylist() != rows:
        raise ValueError("Parquet roundtrip changed prepared rows")
    semantic = hashlib.sha256()
    for row in rows:
        semantic.update((canonical(row) + "\n").encode())
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "canonical_rows_sha256": semantic.hexdigest(),
    }


def prepare_train_rows(
    raw_rows: list[dict],
    *,
    deduplicate_prompts: bool = False,
    expected_raw_rows: int | None = None,
    expected_rows: int | None = None,
) -> list[dict]:
    """Validate raw dapo_filter training rows, optionally dedupe prompts, and rewrite data_source."""
    if expected_rows is None:
        expected_rows = EXPECTED_UNIQUE_ROWS
    if expected_raw_rows is not None and len(raw_rows) != expected_raw_rows:
        raise ValueError(f"Expected {expected_raw_rows} raw training rows, got {len(raw_rows)}")
    if not raw_rows:
        raise ValueError("Training dataset cannot be empty")

    candidate_rows = dedupe_rows_by_prompt(raw_rows) if deduplicate_prompts else raw_rows
    if len(candidate_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} unique training rows, got {len(candidate_rows)}")

    seen_indices: set[str] = set()
    seen_prompts: set[str] = set()
    prepared: list[dict] = []

    for row in candidate_rows:
        if row.get("data_source") != RAW_DATA_SOURCE:
            raise ValueError(f"Expected raw data_source={RAW_DATA_SOURCE!r}, got {row.get('data_source')!r}")

        prompt = row.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
            or not prompt[0]["content"].endswith(BOXED_SUFFIX)
        ):
            raise ValueError("Each training row must have a single user message ending with the boxed suffix")

        body = prompt[0]["content"][: -len(BOXED_SUFFIX)]
        if not body.strip() or "\x00" in prompt[0]["content"]:
            raise ValueError("Invalid empty or NUL-corrupted mathematical prompt text")

        reward_model = row.get("reward_model")
        if (
            not isinstance(reward_model, dict)
            or reward_model.get("style") != "rule"
            or not isinstance(reward_model.get("ground_truth"), str)
            or not reward_model["ground_truth"].strip()
        ):
            raise ValueError("Each training row must specify a non-empty rule-based ground_truth string")

        extra_info = row.get("extra_info")
        index = extra_info.get("index") if isinstance(extra_info, dict) else None
        if not isinstance(index, str) or not index.strip():
            raise ValueError("Each training row must preserve a non-empty string extra_info.index")
        if index in seen_indices:
            raise ValueError(f"Duplicate extra_info.index detected: {index}")
        seen_indices.add(index)

        prompt_key = canonical(prompt)
        if prompt_key in seen_prompts:
            raise ValueError(f"Duplicate prompt detected at index={index}")
        seen_prompts.add(prompt_key)

        updated = deepcopy(row)
        updated["data_source"] = TARGET_DATA_SOURCE
        if {k: v for k, v in updated.items() if k != "data_source"} != {
            k: v for k, v in row.items() if k != "data_source"
        }:
            raise ValueError("Unexpected mutation outside data_source")
        prepared.append(updated)

    verify_reward(prepared)
    return prepared


def main() -> None:
    """Parse CLI arguments, verify pinned dapo_filter source, and write prepared Parquet + manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--local_save_dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Local directory where train.parquet and manifest.json will be written.",
    )
    parser.add_argument(
        "--source",
        choices=tuple(DAPO_FILTER_SOURCES),
        default=DEFAULT_SOURCE,
        help="Pinned Hugging Face dataset repository to prepare (both produce the exact same 12,691 rows).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional offline checked source cache (<source-dir>/<publisher>--<dataset>/<filename>).",
    )
    parser.add_argument(
        "--hdfs-dir",
        "--hdfs_dir",
        dest="hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS destination directory to mirror prepared outputs.",
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = DAPO_FILTER_SOURCES[args.source]
    directory = source["repo_id"].replace("/", "--")
    if args.source_dir is not None:
        source_path = args.source_dir.expanduser().resolve() / directory / source["filename"]
        if source_path.stat().st_size != source["bytes"] or digest(source_path) != source["sha256"]:
            raise ValueError(f"Offline source size/hash mismatch: {source_path}")
    else:
        source_path = fetch_source(source, output / "raw" / directory)

    raw_rows = list(iter_parquet(source_path))
    prepared_rows = prepare_train_rows(
        raw_rows,
        deduplicate_prompts=source["deduplicate_prompts"],
        expected_raw_rows=source["raw_rows"],
        expected_rows=EXPECTED_UNIQUE_ROWS,
    )
    train_artifact = write_parquet(prepared_rows, output / "train.parquet")

    manifest = {
        "schema_version": 1,
        "source": source,
        "lineage": {
            "upstream_base_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
            "difficulty_filtered_source": {
                "repo_id": "aaabiao/dapo_filter",
                "revision": DAPO_FILTER_SOURCES["aaabiao/dapo_filter"]["revision"],
                "raw_rows": DAPO_FILTER_SOURCES["aaabiao/dapo_filter"]["raw_rows"],
                "description": (
                    "Curated in the SimpleTIR setup by filtering DAPO-Math-17k according to initial-model "
                    "accuracy to remove tasks with overly high or overly low solve rates, formatted with "
                    "the boxed answer instruction."
                ),
            },
            "prompt_deduplicated_source": {
                "repo_id": "Jiawei415/DPAO_filter",
                "revision": DAPO_FILTER_SOURCES["Jiawei415/DPAO_filter"]["revision"],
                "rows": EXPECTED_UNIQUE_ROWS,
                "description": (
                    "Exact first-occurrence prompt deduplication of aaabiao/dapo_filter (removing 353 "
                    "duplicate-prompt entries while preserving all 12,691 unique rows and order verbatim)."
                ),
            },
        },
        "transform": {
            "raw_data_source": RAW_DATA_SOURCE,
            "target_data_source": TARGET_DATA_SOURCE,
            "deduplicated_prompts": source["deduplicate_prompts"],
            "raw_rows": len(raw_rows),
            "prepared_rows": len(prepared_rows),
            "description": (
                "Deduplicate by prompt if loading aaabiao/dapo_filter (13,044 -> 12,691 rows), then "
                "rewrite only data_source from 'dapo_filter' to 'math_dapo' for reward dispatch while "
                "preserving prompt, reward_model, extra_info, UUID indices, and row order verbatim."
            ),
            "prompt_suffix": BOXED_SUFFIX,
        },
        "validation_policy": (
            "Training preparation only. Do not use test/aime24.parquet or test/aime25.parquet from "
            "Jiawei415/DPAO_filter (both files have identical bytes/SHA-256); prepare validation splits separately."
        ),
        "reward": {
            "dispatcher": "verl.utils.reward_score.default_compute_score",
            "scorer": "verl.utils.reward_score.math_dapo.compute_score",
        },
        "artifacts": {"train": train_artifact},
        "generator": {
            "path": "examples/data_preprocess/dapo_math_boxed.py",
            "sha256": digest(__file__),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        if not copy(src=str(output), dst=args.hdfs_dir):
            raise RuntimeError(f"Failed to copy prepared outputs from {output} to {args.hdfs_dir}")

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "source": source["repo_id"],
                "rows": train_artifact["rows"],
                "artifacts": {"train": train_artifact},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
