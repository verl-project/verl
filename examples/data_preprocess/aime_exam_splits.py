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
"""Prepare pinned AIME 2024/2025/2026 validation, with distinct I/II metrics.

Existing boxed AIME24/25 files can be supplied to preserve their prompts, answers,
source indices, and metadata exactly while adding exam identity. Each session
contains all 15 problems. No training data or model weights are downloaded.
AIME26 retains MathArena's published regional variant, including II question 10
(sum, answer 850). The publisher explains the US/international variant difference
at https://huggingface.co/datasets/MathArena/aime_2026/discussions/2.

    python examples/data_preprocess/aime_exam_splits.py \
        --output-dir /tmp/aime-validation \
        --existing-aime2024 /path/to/aime2024.parquet \
        --existing-aime2025 /path/to/aime2025.parquet
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from verl.utils.reward_score import default_compute_score

AIME2024_SOURCE = {
    "repo_id": "HuggingFaceH4/aime_2024",
    "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    "filename": "data/train-00000-of-00001.parquet",
    "sha256": "26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf",
    "bytes": 81670,
    "license": None,
    "license_status": "Not declared in the pinned dataset card; no usage rights inferred",
    "dataset_card_url": (
        "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/blob/"
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a/README.md"
    ),
}
AIME2025_SOURCE = {
    "repo_id": "MathArena/aime_2025",
    "revision": "c94da77eb22bbd6439e62a323bec18493a421302",
    "filename": "data/train-00000-of-00001.parquet",
    "sha256": "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d",
    "bytes": 14313,
    "license": "cc-by-nc-sa-4.0",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fetch_source(source, local_dir):
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
        raise ValueError(f"Source size/hash mismatch: {source['repo_id']}@{source['revision']}")
    return path


def iter_parquet(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def verify_reward(rows):
    for row in rows:
        answer = row["reward_model"]["ground_truth"]
        correct = default_compute_score(row["data_source"], f"Answer: {answer}", answer)
        wrong = default_compute_score(row["data_source"], "Answer: [INCORRECT-SENTINEL]", answer)
        if correct["acc"] is not True or correct["score"] != 1.0 or wrong["acc"] is not False or wrong["score"] != -1.0:
            raise ValueError("Unexpected reward dispatch or answer normalization")


def write_parquet(rows, path):
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


BOXED_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."
YEARS = (2024, 2025, 2026)
SESSIONS = ("I", "II")
FILENAME = "data/train-00000-of-00001.parquet"
LICENSE = "cc-by-nc-sa-4.0"
AIME26_II_REFERENCE = {
    "publisher_discussion_url": "https://huggingface.co/datasets/MathArena/aime_2026/discussions/2",
    "publisher_mapping": "Publisher discussion explicitly identifies combined #16 as II #1 and combined #25 as II #10",
    "version": "Pinned MathArena published variant; publisher confirms US/international contest variants differ",
    "problems_url": "https://live.poshenloh.com/past-contests/aime/2026II",
    "answers_url": "https://live.poshenloh.com/past-contests/aime/2026II/answers",
    "publisher": "LIVE by Po-Shen Loh; MAA problems reproduced with permission",
    "combined_source_indices": list(range(16, 31)),
    "exam_question_numbers": list(range(1, 16)),
    "answer_agreement": "All 14 unmodified question answers agree with the licensed exam copy",
    "question_10_variant": (
        "MathArena asks for the sum of all possible BC values (850); licensed copy asks for greatest BC (340). "
        "MathArena's publisher confirms the scraped variant and answer are correct in discussion #2. "
        "Preserve MathArena's question and answer together."
    ),
}
VALIDATION_SOURCES = {
    "aime2024": AIME2024_SOURCE,
    "aime2025": AIME2025_SOURCE,
    "aime2026": {
        "repo_id": "MathArena/aime_2026",
        "revision": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
        "filename": FILENAME,
        "sha256": "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957",
        "bytes": 10065,
        "license": LICENSE,
    },
    "aime2024-I": {
        "repo_id": "MathArena/aime_2024_I",
        "revision": "ea5b061c3e8039dc9858defaafc407d04b995e9f",
        "filename": FILENAME,
        "sha256": "e4033c704609cc7cdfe712ed410357b190733dec75aa5b54a39adc55add49393",
        "bytes": 6187,
        "license": LICENSE,
    },
    "aime2024-II": {
        "repo_id": "MathArena/aime_2024_II",
        "revision": "29d5d31e9b46e215fc24d9b2a3047506823dd101",
        "filename": FILENAME,
        "sha256": "eab2b6a77c048ec4efb7d5f91d1f7548b7fc8a08566b9fcddd1b014699f30dbf",
        "bytes": 7559,
        "license": LICENSE,
    },
    "aime2025-I": {
        "repo_id": "MathArena/aime_2025_I",
        "revision": "65981f72a5ac5bd4d9aaab2cb3adcc8955c95873",
        "filename": FILENAME,
        "sha256": "68f8040bec1d1a609f12b57fa008848ac93d235c2aaead78daaba0e1c4f6e30c",
        "bytes": 9302,
        "license": LICENSE,
    },
    "aime2025-II": {
        "repo_id": "MathArena/aime_2025_II",
        "revision": "68558a4580ae339ef7473f588f44ac41571749b8",
        "filename": FILENAME,
        "sha256": "55008c9da95da684d54d6db0049477632107c31fe6a88a272fb686811cdaa02f",
        "bytes": 10605,
        "license": LICENSE,
    },
    "aime2026-I": {
        "repo_id": "MathArena/aime_2026_I",
        "revision": "37806d403fa0c9ae914ee7d7da215cf2a3c7a94d",
        "filename": FILENAME,
        "sha256": "224257fa2c4ceb0796205047b25f124c610bb72277cd558f68a9ee15d875c688",
        "bytes": 6879,
        "license": LICENSE,
    },
}


def exam_identity(row, year):
    """Use H4's source URL, never its lexicographically ordered ID as a question number."""
    if year == 2024:
        match = re.fullmatch(
            r"https://artofproblemsolving\.com/wiki/index\.php/2024_AIME_(I|II)_Problems/Problem_(\d+)",
            row.get("url", ""),
        )
        if str(row.get("year")) != "2024" or match is None:
            raise ValueError("AIME2024 requires its original year/session/problem source URL")
        session, number = match[1], int(match[2])
    elif year in (2025, 2026):
        index = row.get("problem_idx")
        if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= 30:
            raise ValueError(f"AIME{year} source problem_idx must be in 1 through 30")
        session = "I" if index <= 15 else "II"
        number = index if session == "I" else index - 15
    else:
        raise ValueError(f"Unsupported AIME year: {year}")
    if not 1 <= number <= 15:
        raise ValueError("Each AIME session must contain question numbers 1 through 15")
    return session, number


def source_index(row, year):
    index = row.get("id" if year == 2024 else "problem_idx")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("AIME source index must be an integer")
    return str(index)


def source_answer(row, year):
    answer = row["answer"]
    if year == 2024:
        if not isinstance(answer, str) or not answer.isascii() or not answer.isdigit():
            raise ValueError("Invalid AIME2024 source answer")
        answer = int(answer)
    if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer <= 999:
        raise ValueError("AIME answer must be an integer between 0 and 999")
    return str(answer)


def check_coverage(rows, *, year):
    identities = [exam_identity(row, year) for row in rows]
    expected = {(session, number) for session in SESSIONS for number in range(1, 16)}
    if len(rows) != 30 or set(identities) != expected:
        raise ValueError(f"AIME{year} must contain exactly 15 distinct questions in each of I and II")
    indices = [source_index(row, year) for row in rows]
    if len(set(indices)) != 30:
        raise ValueError(f"AIME{year} has duplicate source indices")


def verify_membership(sources):
    """Corroborate URLs/combined indexing with the publisher's separate exam files."""
    for year in YEARS:
        rows = sources[f"aime{year}"]
        check_coverage(rows, year=year)
        by_exam = {exam_identity(row, year): row for row in rows}
        for session in SESSIONS:
            key = f"aime{year}-{session}"
            if key not in VALIDATION_SOURCES:
                # The publisher provides AIME26 and AIME26-I. Combined 16..30
                # correspond to AIME II, confirmed by the publisher's discussion
                # and the licensed AIME II copy linked in the manifest,
                # with the documented MathArena question 10 variant preserved.
                continue
            reference = sources[key]
            numbers = [row["problem_idx"] for row in reference]
            if len(reference) != 15 or set(numbers) != set(range(1, 16)):
                raise ValueError(f"Invalid publisher session coverage: {key}")
            for ref in reference:
                row = by_exam[session, ref["problem_idx"]]
                if source_answer(row, year) != source_answer(ref, 2025):
                    raise ValueError(f"Publisher session answer disagrees: {key}/{ref['problem_idx']}")
                # H4 and MathArena transcriptions of AIME24 differ in LaTeX and
                # whitespace. Preserve H4 verbatim and corroborate URL/number/GT.
                if year != 2024 and row["problem"] != ref["problem"]:
                    raise ValueError(f"Publisher session problem disagrees: {key}/{ref['problem_idx']}")


def prepare_validation(sources, existing=None):
    """Preserve legacy rows and create independently named validation metric groups."""
    existing = existing or {}
    verify_membership(sources)
    grouped = {f"aime{year}-{session}": [] for year in YEARS for session in SESSIONS}
    for year in YEARS:
        dataset = f"aime{year}"
        old_rows = existing.get(dataset)
        if old_rows is not None:
            old = {row["extra_info"]["index"]: row for row in old_rows}
            expected = {source_index(row, year) for row in sources[dataset]}
            if len(old_rows) != 30 or set(old) != expected:
                raise ValueError(f"Existing {dataset} does not contain all 30 original source indices")
        for raw in sources[dataset]:
            problem = raw["problem"]
            if not isinstance(problem, str) or not problem or any(ord(c) < 32 and c not in "\n\r" for c in problem):
                raise ValueError(f"Invalid {dataset} mathematical text/control characters")
            index, answer = source_index(raw, year), source_answer(raw, year)
            prompt = [{"role": "user", "content": problem + BOXED_SUFFIX}]
            if old_rows is not None:
                row = deepcopy(old[index])
                if row["data_source"] != dataset or row["prompt"] != prompt:
                    raise ValueError(f"Existing {dataset}/{index} prompt differs from pinned boxed source")
                if row["reward_model"] != {"style": "rule", "ground_truth": answer}:
                    raise ValueError(f"Existing {dataset}/{index} ground truth differs from pinned source")
            else:
                row = {
                    "data_source": dataset,
                    "ability": "math",
                    "prompt": prompt,
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"index": index, "split": "validation", "source_split": "train"},
                }
            session, number = exam_identity(raw, year)
            key = f"{dataset}-{session}"
            row["data_source"] = key
            row["extra_info"].update(year=year, session=session, problem_number=number)
            grouped[key].append(row)
    for key, rows in grouped.items():
        if len(rows) != 15 or {row["extra_info"]["problem_number"] for row in rows} != set(range(1, 16)):
            raise ValueError(f"Incomplete prepared exam: {key}")
        verify_reward(rows)
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="Offline checked source cache, named publisher--dataset/data/..."
    )
    parser.add_argument("--existing-aime2024", type=Path)
    parser.add_argument("--existing-aime2025", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for key, source in VALIDATION_SOURCES.items():
        directory = source["repo_id"].replace("/", "--")
        if args.source_dir:
            path = args.source_dir / directory / source["filename"]
            if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
                raise ValueError(f"Offline source size/hash mismatch: {key}")
        else:
            path = fetch_source(source, output / "raw" / directory)
        sources[key] = list(iter_parquet(path))
    existing, inputs = {}, {}
    for year in (2024, 2025):
        path = getattr(args, f"existing_aime{year}")
        if path:
            existing[f"aime{year}"] = pq.read_table(path).to_pylist()
            inputs[f"aime{year}"] = {"path": str(path.resolve()), "sha256": digest(path)}
    grouped = prepare_validation(sources, existing)
    artifacts = {key: write_parquet(rows, output / f"{key}.parquet") for key, rows in grouped.items()}
    all_rows = [row for rows in grouped.values() for row in rows]
    artifacts["all"] = write_parquet(all_rows, output / "aime2024-2026.parquet")
    manifest = {
        "schema_version": 1,
        "sources": VALIDATION_SOURCES,
        "preserved_inputs": inputs,
        "prompt_policy": (
            "Unmodified mathematical source text plus the identical boxed suffix; existing prompts preserved"
        ),
        "prompt_suffix": BOXED_SUFFIX,
        "session_verification": {
            "aime2024": "Original H4 URL contains exam and question number; all 30 answers match publisher I/II files",
            "aime2025": "All 30 problem texts and answers match publisher I/II files exactly",
            "aime2026": (
                "First 15 texts/answers match publisher I exactly; publisher discussion #2 identifies #16 as II #1 "
                "and #25 as II #10, confirming combined II indexing"
            ),
        },
        "aime2026_II_exam_reference": AIME26_II_REFERENCE,
        "groups": dict(sorted(Counter(row["data_source"] for row in all_rows).items())),
        "questions": {
            key: [
                {field: row["extra_info"][field] for field in ("index", "year", "session", "problem_number")}
                for row in rows
            ]
            for key, rows in grouped.items()
        },
        "artifacts": artifacts,
        "generator": {"path": "examples/data_preprocess/aime_exam_splits.py", "sha256": digest(__file__)},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "groups": manifest["groups"], "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
